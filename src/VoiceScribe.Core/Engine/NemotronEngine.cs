using Microsoft.Extensions.Logging;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

using NAudio.Wave;
using System.Diagnostics;
using System.Threading.Channels;

using VoiceScribe.Core.Audio;
using VoiceScribe.Core.Configuration;
using VoiceScribe.Core.ModelAssets;


/// <summary>
/// NemotronEngine es la clase central que maneja la inferencia de los modelos ONNX para el reconocimiento de voz en tiempo real. 
/// Se encarga de cargar los modelos, procesar los datos de audio, mantener los estados de los búferes acústicos y lingüísticos, 
/// y emitir los tokens reconocidos. Además, incluye funciones de diagnóstico para inspeccionar las sesiones de ONNX y los tensores
/// de activación, lo que facilita la depuración y optimización del proceso de inferencia. 
/// La clase implementa IDisposable para asegurar la liberación adecuada de recursos como las sesiones de ONNX y los escritores de archivos.
/// </summary>  
namespace VoiceScribe.Core.Engine
{
    public sealed class NemotronEngine : IDisposable, IAsyncDisposable
    {
        private readonly ILogger<NemotronEngine> _logger;
        private readonly AudioCaptureOptions _audioOptions;
        private readonly NemotronModelOptions _modelOptions;
        private readonly NemotronModelDefinition _modelDefinition;
        private readonly NemotronOnnxSessionFactories _sessionFactories;
        private readonly Channel<byte[]> _audioQueue;
        private readonly Task _audioWorker;
        private readonly CancellationTokenSource _workerCancellation = new();
        private const int MetricsReportIntervalChunks = 20;

        private readonly TextWriter? _transcriptWriter;

        private bool _isDisposed;
        private int _queueCompleted;
        private long _metricsChunkCount;
        private long _metricsDecoderRunCount;
        private long _metricsJointRunCount;
        private long _metricsEmittedTokenCount;
        private double _metricsFeatureExtractionMs;
        private double _metricsEncoderMs;
        private double _metricsDecoderMs;
        private double _metricsJointMs;
        private double _metricsDecodeLoopMs;
        private double _metricsTotalChunkMs;
        private int _trailingSilenceChunksProcessed;
        private bool _hasProcessedSpeechChunk;

        // Sesiones de ONNX Runtime para cada modelo
        private InferenceSession _encoderSession = null!;
        private InferenceSession _decoderSession = null!;
        private InferenceSession _jointSession = null!;

        // Estados del búfer lingüístico (Decoder / Transducer)
        private DenseTensor<long> _decoderTargets = null!;
        private DenseTensor<float> _decoderHIn = null!;
        private DenseTensor<float> _decoderCIn = null!;        
               
        // Estados del búfer acústico (Cachés de streaming)
        private DenseTensor<float> _cacheLastChannel = null!;
        private DenseTensor<float> _cacheLastTime = null!;
        private DenseTensor<long> _cacheLastChannelLen = null!;

        // Vocabulario mapeado desde tokenizer.json (Índice de token → String del token)
        private Dictionary<long, string> _vocab = null!;
        
        /// <summary>
        /// Extractor acústico para convertir los fragmentos de audio PCM en características log-mel, que son las entradas esperadas por el modelo
        /// encoder. Se inicializa a partir de un archivo de configuración específico que define los parámetros del extractor, como el número de 
        /// filtros mel, la ventana de análisis, etc.
        /// Este extractor es crucial para el preprocesamiento del audio y la generación de las características que alimentan el modelo de reconocimiento
        /// de voz. 
        /// </summary>
        private AudioFeatureExtractor _featureExtractor = null!;


        /// <summary>
        /// Último token predicho por el modelo. Se utiliza para mantener el estado del decoder y evitar emitir tokens repetidos o no deseados. 
        /// Este campo se actualiza cada vez que se emite un token reconocido, y se puede utilizar para implementar lógica adicional de filtrado o
        /// post-procesamiento de tokens si es necesario.
        /// </summary>
        /// <remarks> 
        /// No se está usando actualmente pero está para usarlo en diagnósticos.
        /// </remarks>    
        private long _lastPredictedToken = -1;


        /// <summary>
        /// Constructor principal que recibe la ruta de los modelos y un StreamWriter opcional para guardar la transcripción en un archivo.
        /// </summary>
        /// <param name="logger"></param>
        /// <param name="modelFolderPath"></param>
        /// <param name="fileWriter"></param>
        public NemotronEngine(
            ILogger<NemotronEngine> logger,
            string modelFolderPath,
            AudioCaptureOptions audioOptions,
            NemotronModelOptions modelOptions,
            NemotronModelDefinition modelDefinition,
            NemotronOnnxSessionFactories sessionFactories,
            TextWriter? transcriptWriter)
        {
            _logger = logger;
            _audioOptions = audioOptions;
            _modelOptions = modelOptions;
            _modelDefinition = modelDefinition;
            _sessionFactories = sessionFactories;
            _transcriptWriter = transcriptWriter;

            try
            {
                string tokenizerJsonPath = Path.Combine(
                    modelFolderPath,
                    NemotronModelFiles.Tokenizer);
                Initialize(modelFolderPath, tokenizerJsonPath);

                _audioQueue = Channel.CreateBounded<byte[]>(
                    new BoundedChannelOptions(_audioOptions.QueueCapacity)
                    {
                        FullMode = BoundedChannelFullMode.Wait,
                        SingleReader = true,
                        SingleWriter = false
                    });
                _audioWorker = Task.Run(
                    () => ProcessAudioQueueAsync(_workerCancellation.Token));
            }
            catch
            {
                DisposeSessions();
                _workerCancellation.Dispose();
                throw;
            }
        }


        /// <summary>
        /// Inicializa las sesiones de ONNX Runtime y ejecuta los diagnósticos.
        /// </summary>
        private void Initialize(string modelFolderPath, string tokenizerJsonPath)
        {
            _logger.LogInformation("Bootstrapping ONNX Execution Runtimes...");

            var encoderPath = Path.Combine(modelFolderPath, _modelDefinition.Encoder.FileName);
            var decoderPath = Path.Combine(modelFolderPath, _modelDefinition.Decoder.FileName);
            var jointPath = Path.Combine(modelFolderPath, _modelDefinition.Joiner.FileName);

            _logger.LogInformation(
                "Using ONNX execution providers: {ExecutionProviders}.",
                _sessionFactories.Describe());

            _encoderSession = _sessionFactories.Encoder.CreateSession(encoderPath);
            _decoderSession = _sessionFactories.Decoder.CreateSession(decoderPath);
            _jointSession = _sessionFactories.Joiner.CreateSession(jointPath);

            ValidateSessionContracts();

            // Ejecución de los nuevos diagnósticos reutilizables antes de abrir micrófonos
            DiagnoseSessionMetadata("ENCODER", _encoderSession);
            DiagnoseSessionMetadata("DECODER", _decoderSession);
            DiagnoseSessionMetadata("JOINT / JOINER", _jointSession);

            _logger.LogInformation("Cargando y mapeando el vocabulario desde tokenizer.json...");
            LoadTokenizer(tokenizerJsonPath);

            var audioProcessorConfigPath = Path.Combine(modelFolderPath, NemotronModelFiles.AudioProcessorConfig);
            var genaiConfigPath = Path.Combine(modelFolderPath, NemotronModelFiles.GenAiConfig);

            _logger.LogInformation("Inicializando extractor acústico desde {Path}", audioProcessorConfigPath);
            _featureExtractor = AudioFeatureExtractor.FromConfig(
                audioProcessorConfigPath,
                genaiConfigPath
            );

            _logger.LogInformation("Allocating streaming architecture tensors...");
            InitializeTensors();

            _logger.LogInformation("NemotronEngine inicializado correctamente.");
        }


        /// <summary>
        /// Método principal que se acopla directamente al evento DataAvailable de NAudio.
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        public void ProcessAudioChunk(object? sender, WaveInEventArgs e)
        {
            if (_isDisposed || Volatile.Read(ref _queueCompleted) != 0)
                return;

            byte[] audioBytes = new byte[e.BytesRecorded];
            Buffer.BlockCopy(e.Buffer, 0, audioBytes, 0, e.BytesRecorded);

            if (!_audioQueue.Writer.TryWrite(audioBytes))
            {
                _logger.LogWarning(
                    "Audio inference queue is full. Dropping {ByteCount} bytes.",
                    e.BytesRecorded);
            }
        }

        private async Task ProcessAudioQueueAsync(
            CancellationToken cancellationToken)
        {
            int bytesPerSample = _audioOptions.BitsPerSample / 8;
            int bytesPerFrame = bytesPerSample * _audioOptions.Channels;
            int expectedBytes = _modelDefinition.ChunkSamples * bytesPerFrame;
            byte[] pending = new byte[expectedBytes * 2];
            float[] pcmChunk = new float[_modelDefinition.ChunkSamples];
            int bufferedBytes = 0;

            try
            {
                await foreach (byte[] audioBytes in
                    _audioQueue.Reader.ReadAllAsync(cancellationToken))
                {
                    if (pending.Length < bufferedBytes + audioBytes.Length)
                    {
                        Array.Resize(
                            ref pending,
                            Math.Max(
                                pending.Length * 2,
                                bufferedBytes + audioBytes.Length));
                    }

                    Buffer.BlockCopy(
                        audioBytes,
                        0,
                        pending,
                        bufferedBytes,
                        audioBytes.Length);
                    bufferedBytes += audioBytes.Length;

                    int offset = 0;
                    while (bufferedBytes - offset >= expectedBytes)
                    {
                        cancellationToken.ThrowIfCancellationRequested();
                        ProcessPcmChunk(
                            pending,
                            offset,
                            bytesPerFrame,
                            pcmChunk,
                            forceProcessSilence: false);
                        offset += expectedBytes;
                    }

                    if (offset > 0)
                    {
                        bufferedBytes -= offset;
                        Buffer.BlockCopy(
                            pending,
                            offset,
                            pending,
                            0,
                            bufferedBytes);
                    }
                }

                FlushEndOfStream(
                    pending,
                    bufferedBytes,
                    expectedBytes,
                    bytesPerFrame,
                    pcmChunk,
                    cancellationToken);
            }
            catch (OperationCanceledException)
                when (cancellationToken.IsCancellationRequested)
            {
                _logger.LogInformation("Audio inference worker cancelled.");
            }
        }

        private void ProcessPcmChunk(
            byte[] buffer,
            int bufferOffset,
            int bytesPerFrame,
            float[] pcmChunk,
            bool forceProcessSilence)
        {
            try
            {
                long chunkStart = Stopwatch.GetTimestamp();
                int expectedSamples = _modelDefinition.ChunkSamples;
                float maxAmp = 0f;

                for (int i = 0; i < expectedSamples; i++)
                {
                    int sampleOffset = bufferOffset + i * bytesPerFrame;
                    float sample = ReadPcmSample(buffer, sampleOffset, _audioOptions.BitsPerSample);

                    pcmChunk[i] = sample;
                    maxAmp = Math.Max(maxAmp, Math.Abs(sample));
                }

                bool isSilent = maxAmp < _audioOptions.SilenceThreshold;
                if (!ShouldProcessChunk(isSilent, forceProcessSilence))
                    return;

                // IMPORTANTE:
                // Aquí NO se debe hacer reshape directo del PCM.
                // Aquí debe ir PCM -> log-mel.
                long featureStart = Stopwatch.GetTimestamp();
                var features = _featureExtractor.Extract(pcmChunk);
                double featureExtractionMs = GetElapsedMilliseconds(featureStart);

                _logger.LogDebug("Features extraídas: Frames={Frames}, MelBins={MelBins}, Total={Total}",
                    features.Frames, features.MelBins, features.Data.Length);

                var audioTensor = new DenseTensor<float>(
                    features.Data,
                    new[] { 1, features.Frames, features.MelBins }
                );

                var lengthTensor = new DenseTensor<long>(
                    new long[] { features.Frames },
                    new[] { 1 }
                );

                var langIdTensor = new DenseTensor<long>(
                    new long[] { _modelOptions.LanguageId },
                    new[] { 1 }
                );

                var encoderInputs = new List<NamedOnnxValue>
                {
                    NamedOnnxValue.CreateFromTensor(_modelDefinition.Encoder.AudioFeaturesInput, audioTensor),
                    NamedOnnxValue.CreateFromTensor(_modelDefinition.Encoder.InputLengthsInput, lengthTensor),
                    NamedOnnxValue.CreateFromTensor(_modelDefinition.Encoder.CacheLastChannelInput, _cacheLastChannel),
                    NamedOnnxValue.CreateFromTensor(_modelDefinition.Encoder.CacheLastTimeInput, _cacheLastTime),
                    NamedOnnxValue.CreateFromTensor(_modelDefinition.Encoder.CacheLastChannelLengthInput, _cacheLastChannelLen),
                    NamedOnnxValue.CreateFromTensor(_modelDefinition.Encoder.LanguageIdInput, langIdTensor)
                };

                long encoderStart = Stopwatch.GetTimestamp();
                using var encoderResults = _encoderSession.Run(encoderInputs);
                double encoderMs = GetElapsedMilliseconds(encoderStart);

                var acoustic = encoderResults
                    .First(x => x.Name == _modelDefinition.Encoder.EncoderOutputsOutput)
                    .AsTensor<float>();

                int T = acoustic.Dimensions[1];

                UpdateCache(encoderResults);

                long decodeLoopStart = Stopwatch.GetTimestamp();
                RnntDecodeMetrics decodeMetrics = DecodeRnntFrames(acoustic, T);
                double decodeLoopMs = GetElapsedMilliseconds(decodeLoopStart);

                _transcriptWriter?.Flush();

                RecordPerformanceMetrics(
                    featureExtractionMs,
                    encoderMs,
                    decodeMetrics,
                    decodeLoopMs,
                    GetElapsedMilliseconds(chunkStart));
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error processing queued audio chunk.");
            }
        }

        private bool ShouldProcessChunk(
            bool isSilent,
            bool forceProcessSilence)
        {
            if (forceProcessSilence)
                return true;

            if (!isSilent)
            {
                _hasProcessedSpeechChunk = true;
                _trailingSilenceChunksProcessed = 0;
                return true;
            }

            if (!_hasProcessedSpeechChunk ||
                _trailingSilenceChunksProcessed >= _audioOptions.TrailingSilenceChunks)
            {
                return false;
            }

            _trailingSilenceChunksProcessed++;
            _logger.LogDebug(
                "Processing trailing silence chunk {Chunk}/{Total}.",
                _trailingSilenceChunksProcessed,
                _audioOptions.TrailingSilenceChunks);
            return true;
        }

        private void FlushEndOfStream(
            byte[] pending,
            int bufferedBytes,
            int expectedBytes,
            int bytesPerFrame,
            float[] pcmChunk,
            CancellationToken cancellationToken)
        {
            if (bufferedBytes > 0)
            {
                cancellationToken.ThrowIfCancellationRequested();
                Array.Clear(pending, bufferedBytes, expectedBytes - bufferedBytes);
                _logger.LogInformation(
                    "Flushing final partial audio chunk: {BufferedBytes}/{ExpectedBytes} bytes.",
                    bufferedBytes,
                    expectedBytes);
                ProcessPcmChunk(
                    pending,
                    0,
                    bytesPerFrame,
                    pcmChunk,
                    forceProcessSilence: true);
            }

            if (_audioOptions.FinalSilencePaddingChunks == 0)
                return;

            byte[] silence = new byte[expectedBytes];
            for (int i = 0; i < _audioOptions.FinalSilencePaddingChunks; i++)
            {
                cancellationToken.ThrowIfCancellationRequested();
                _logger.LogDebug(
                    "Flushing final silence padding chunk {Chunk}/{Total}.",
                    i + 1,
                    _audioOptions.FinalSilencePaddingChunks);
                ProcessPcmChunk(
                    silence,
                    0,
                    bytesPerFrame,
                    pcmChunk,
                    forceProcessSilence: true);
            }
        }

        /// <summary>
        /// Reads the first channel of an integer PCM frame and normalizes it to [-1, 1].
        /// </summary>
        private static float ReadPcmSample(byte[] buffer, int offset, int bitsPerSample)
        {
            return bitsPerSample switch
            {
                8 => (buffer[offset] - 128) / 128f,
                16 => BitConverter.ToInt16(buffer, offset) / 32768f,
                24 => ReadPcm24(buffer, offset) / 8388608f,
                32 => BitConverter.ToInt32(buffer, offset) / 2147483648f,
                _ => throw new NotSupportedException(
                    $"PCM bit depth '{bitsPerSample}' is not supported. Use 8, 16, 24 or 32 bits.")
            };
        }

        private static int ReadPcm24(byte[] buffer, int offset)
        {
            int sample = buffer[offset]
                | buffer[offset + 1] << 8
                | buffer[offset + 2] << 16;

            return (sample & 0x00800000) != 0
                ? sample | unchecked((int)0xFF000000)
                : sample;
        }


        /// <summary>
        /// Decodifica los frames acústicos utilizando el mecanismo RNN-T. Para cada frame, se ejecuta el decoder y el joint para obtener
        /// el token más probable.
        /// </summary>
        /// <param name="acoustic"></param>
        /// <param name="frameCount"></param>
        private RnntDecodeMetrics DecodeRnntFrames(
            Tensor<float> acoustic,
            int frameCount)
        {
            int encoderEmbeddingSize = acoustic.Dimensions[^1];
            if (encoderEmbeddingSize != _modelDefinition.EncoderHiddenSize)
                throw new InvalidOperationException(
                    $"Encoder produced embedding size {encoderEmbeddingSize}; expected {_modelDefinition.EncoderHiddenSize}.");

            float[] encoderFrameBuffer = new float[encoderEmbeddingSize];
            var encoderFrame = new DenseTensor<float>(
                encoderFrameBuffer,
                new[] { 1, 1, encoderEmbeddingSize });
            RnntDecodeMetrics metrics = new();

            for (int t = 0; t < frameCount; t++)
            {
                CopySingleFrame(acoustic, t, encoderFrameBuffer);

                int maxSymbolsPerStep =
                    _modelOptions.MaxSymbolsPerStep ?? _modelDefinition.MaxSymbolsPerStep;

                for (int u = 0; u < maxSymbolsPerStep; u++)
                {
                    var decInputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor(_modelDefinition.Decoder.TargetsInput, _decoderTargets),
                NamedOnnxValue.CreateFromTensor(_modelDefinition.Decoder.HiddenStateInput, _decoderHIn),
                NamedOnnxValue.CreateFromTensor(_modelDefinition.Decoder.CellStateInput, _decoderCIn)
            };

                    long decoderStart = Stopwatch.GetTimestamp();
                    using var decResults = _decoderSession.Run(decInputs);
                    metrics.DecoderMs += GetElapsedMilliseconds(decoderStart);
                    metrics.DecoderRuns++;

                    var decOutTensor = decResults
                        .First(x => x.Name == _modelDefinition.Decoder.DecoderOutput)
                        .AsTensor<float>();

                    var hOut = decResults
                        .First(x => x.Name == _modelDefinition.Decoder.HiddenStateOutput)
                        .AsTensor<float>();
                    var cOut = decResults
                        .First(x => x.Name == _modelDefinition.Decoder.CellStateOutput)
                        .AsTensor<float>();

                    int decoderEmbeddingSize = checked((int)decOutTensor.Length);
                    if (decoderEmbeddingSize != _modelDefinition.DecoderHiddenSize)
                        throw new InvalidOperationException(
                            $"Decoder produced {decoderEmbeddingSize} values for one step; expected {_modelDefinition.DecoderHiddenSize}.");

                    Tensor<float> decOutReshaped = decOutTensor.Reshape(
                        new[] { 1, 1, decoderEmbeddingSize });

                    var jointInputs = new List<NamedOnnxValue>
                    {
                        NamedOnnxValue.CreateFromTensor(_modelDefinition.Joiner.EncoderOutputInput, encoderFrame),
                        NamedOnnxValue.CreateFromTensor(_modelDefinition.Joiner.DecoderOutputInput, decOutReshaped)
                    };

                    long jointStart = Stopwatch.GetTimestamp();
                    using var jointResults = _jointSession.Run(jointInputs);
                    metrics.JointMs += GetElapsedMilliseconds(jointStart);
                    metrics.JointRuns++;

                    var jointOut = jointResults
                        .First(x => x.Name == _modelDefinition.Joiner.LogitsOutput)
                        .AsTensor<float>();

                    long token = ArgMax(jointOut);

                    long blankId = ResolveBlankId(jointOut);
                    if (token == blankId)
                        break;

                    CopyTensorValues(hOut, _decoderHIn, "decoder hidden state");
                    CopyTensorValues(cOut, _decoderCIn, "decoder cell state");
                    _decoderTargets[0, 0] = token;

                    EmitToken(token);
                    metrics.EmittedTokens++;
                }
            }

            return metrics;
        }

        private void RecordPerformanceMetrics(
            double featureExtractionMs,
            double encoderMs,
            RnntDecodeMetrics decodeMetrics,
            double decodeLoopMs,
            double totalChunkMs)
        {
            _metricsChunkCount++;
            _metricsDecoderRunCount += decodeMetrics.DecoderRuns;
            _metricsJointRunCount += decodeMetrics.JointRuns;
            _metricsEmittedTokenCount += decodeMetrics.EmittedTokens;
            _metricsFeatureExtractionMs += featureExtractionMs;
            _metricsEncoderMs += encoderMs;
            _metricsDecoderMs += decodeMetrics.DecoderMs;
            _metricsJointMs += decodeMetrics.JointMs;
            _metricsDecodeLoopMs += decodeLoopMs;
            _metricsTotalChunkMs += totalChunkMs;

            if (_metricsChunkCount % MetricsReportIntervalChunks != 0)
                return;

            double chunks = _metricsChunkCount;
            double decoderRuns = Math.Max(1, _metricsDecoderRunCount);
            double jointRuns = Math.Max(1, _metricsJointRunCount);

            Console.ForegroundColor = ConsoleColor.DarkGray;
            Console.WriteLine();
            Console.WriteLine(
                "[Metrics] chunks={0} provider={1} avg_ms: total={2:F1}, features={3:F1}, encoder={4:F1}, decode_loop={5:F1}, decoder_run={6:F2}, joint_run={7:F2}; runs: decoder={8}, joint={9}, tokens={10}",
                _metricsChunkCount,
                _sessionFactories.Describe(),
                _metricsTotalChunkMs / chunks,
                _metricsFeatureExtractionMs / chunks,
                _metricsEncoderMs / chunks,
                _metricsDecodeLoopMs / chunks,
                _metricsDecoderMs / decoderRuns,
                _metricsJointMs / jointRuns,
                _metricsDecoderRunCount,
                _metricsJointRunCount,
                _metricsEmittedTokenCount);
            Console.ResetColor();
        }

        private static double GetElapsedMilliseconds(long startTimestamp) =>
            (Stopwatch.GetTimestamp() - startTimestamp) * 1000.0 /
            Stopwatch.Frequency;

        private struct RnntDecodeMetrics
        {
            public double DecoderMs;
            public double JointMs;
            public long DecoderRuns;
            public long JointRuns;
            public long EmittedTokens;
        }

        private long ResolveBlankId(Tensor<float> jointOutput)
        {
            int classCount = jointOutput.Dimensions[^1];
            long blankId = _modelOptions.BlankId ?? _modelDefinition.BlankId;

            if (blankId < 0 || blankId >= classCount)
                throw new InvalidOperationException(
                    $"Blank token ID '{blankId}' is outside the joint output range [0, {classCount - 1}].");

            return blankId;
        }


        /// <summary>
        /// Emite un token reconocido. Primero se limpia el token para eliminar caracteres especiales o de control, luego se imprime en 
        /// consola y se escribe en el archivo si el StreamWriter está disponible. También se actualiza el último token predicho para 
        /// mantener el estado del decoder.
        /// </summary>
        /// <param name="token"></param>
        private void EmitToken(long token)
        {
            if (!_vocab.TryGetValue(token, out string? text))
                return;

            string clean = CleanToken(text);

            if (string.IsNullOrEmpty(clean))
                return;

            Console.Write(clean);
            _transcriptWriter?.Write(clean);

            _logger.LogDebug("Emitido: '{Clean}' (Token: {Token})", clean, token);

            _lastPredictedToken = token;
        }



        // ====================================================================
        // REPOSITORIO DE FUNCIONES DE DIAGNÓSTICO REUTILIZABLES
        // ====================================================================

        /// <summary>
        /// Escanea de forma exhaustiva los contratos de entrada y salida del grafo de una sesión ONNX.
        /// </summary>
        /// <param name="sessionLabel"></param>
        /// <param name="session"></param>
        private void DiagnoseSessionMetadata(string sessionLabel, InferenceSession session)
        {
            _logger.LogInformation("==================================================");
            _logger.LogInformation(" DIAGNÓSTICO DE GRAFO ONNX: [{Label}]", sessionLabel);
            _logger.LogInformation("==================================================");

            _logger.LogInformation(">>> INPUTS EXPECTED (Nodos de Entrada):");
            foreach (var input in session.InputMetadata)
            {
                string dims = input.Value.Dimensions != null ? string.Join(", ", input.Value.Dimensions) : "Dinámico";
                // CORREGIDO: Usamos ElementDataType para evitar el error CS1061
                _logger.LogInformation(" -> Name: '{Key}' | Type: {Type} | Dimensions: [{Dims}]",
                    input.Key, input.Value.ElementDataType, dims);
            }

            _logger.LogInformation(">>> OUTPUTS AVAILABLE (Nodos de Salida):");
            foreach (var output in session.OutputMetadata)
            {
                string dims = output.Value.Dimensions != null ? string.Join(", ", output.Value.Dimensions) : "Dinámico";
                // CORREGIDO: Usamos ElementDataType para evitar el error CS1061
                _logger.LogInformation(" -> Name: '{Key}' | Type: {Type} | Dimensions: [{Dims}]",
                    output.Key, output.Value.ElementDataType, dims);
            }

            _logger.LogInformation("==================================================\n");
        }

        private void ValidateSessionContracts()
        {
            ValidateSessionContract(
                "encoder",
                _encoderSession,
                [
                    _modelDefinition.Encoder.AudioFeaturesInput,
                    _modelDefinition.Encoder.InputLengthsInput,
                    _modelDefinition.Encoder.CacheLastChannelInput,
                    _modelDefinition.Encoder.CacheLastTimeInput,
                    _modelDefinition.Encoder.CacheLastChannelLengthInput,
                    _modelDefinition.Encoder.LanguageIdInput
                ],
                [
                    _modelDefinition.Encoder.EncoderOutputsOutput,
                    _modelDefinition.Encoder.CacheLastChannelOutput,
                    _modelDefinition.Encoder.CacheLastTimeOutput,
                    _modelDefinition.Encoder.CacheLastChannelLengthOutput
                ]);
            ValidateNodeTypes(
                "encoder",
                _encoderSession,
                new Dictionary<string, TensorElementType>
                {
                    [_modelDefinition.Encoder.AudioFeaturesInput] = TensorElementType.Float,
                    [_modelDefinition.Encoder.InputLengthsInput] = TensorElementType.Int64,
                    [_modelDefinition.Encoder.CacheLastChannelInput] = TensorElementType.Float,
                    [_modelDefinition.Encoder.CacheLastTimeInput] = TensorElementType.Float,
                    [_modelDefinition.Encoder.CacheLastChannelLengthInput] = TensorElementType.Int64,
                    [_modelDefinition.Encoder.LanguageIdInput] = TensorElementType.Int64
                },
                new Dictionary<string, TensorElementType>
                {
                    [_modelDefinition.Encoder.EncoderOutputsOutput] = TensorElementType.Float,
                    [_modelDefinition.Encoder.CacheLastChannelOutput] = TensorElementType.Float,
                    [_modelDefinition.Encoder.CacheLastTimeOutput] = TensorElementType.Float,
                    [_modelDefinition.Encoder.CacheLastChannelLengthOutput] = TensorElementType.Int64
                });

            ValidateSessionContract(
                "decoder",
                _decoderSession,
                [
                    _modelDefinition.Decoder.TargetsInput,
                    _modelDefinition.Decoder.HiddenStateInput,
                    _modelDefinition.Decoder.CellStateInput
                ],
                [
                    _modelDefinition.Decoder.DecoderOutput,
                    _modelDefinition.Decoder.HiddenStateOutput,
                    _modelDefinition.Decoder.CellStateOutput
                ]);
            ValidateNodeTypes(
                "decoder",
                _decoderSession,
                new Dictionary<string, TensorElementType>
                {
                    [_modelDefinition.Decoder.TargetsInput] = TensorElementType.Int64,
                    [_modelDefinition.Decoder.HiddenStateInput] = TensorElementType.Float,
                    [_modelDefinition.Decoder.CellStateInput] = TensorElementType.Float
                },
                new Dictionary<string, TensorElementType>
                {
                    [_modelDefinition.Decoder.DecoderOutput] = TensorElementType.Float,
                    [_modelDefinition.Decoder.HiddenStateOutput] = TensorElementType.Float,
                    [_modelDefinition.Decoder.CellStateOutput] = TensorElementType.Float
                });

            ValidateSessionContract(
                "joiner",
                _jointSession,
                [
                    _modelDefinition.Joiner.EncoderOutputInput,
                    _modelDefinition.Joiner.DecoderOutputInput
                ],
                [_modelDefinition.Joiner.LogitsOutput]);
            ValidateNodeTypes(
                "joiner",
                _jointSession,
                new Dictionary<string, TensorElementType>
                {
                    [_modelDefinition.Joiner.EncoderOutputInput] = TensorElementType.Float,
                    [_modelDefinition.Joiner.DecoderOutputInput] = TensorElementType.Float
                },
                new Dictionary<string, TensorElementType>
                {
                    [_modelDefinition.Joiner.LogitsOutput] = TensorElementType.Float
                });
        }

        private static void ValidateSessionContract(
            string sessionName,
            InferenceSession session,
            IEnumerable<string> requiredInputs,
            IEnumerable<string> requiredOutputs)
        {
            string[] missingInputs = requiredInputs
                .Where(name => !session.InputMetadata.ContainsKey(name))
                .ToArray();
            string[] missingOutputs = requiredOutputs
                .Where(name => !session.OutputMetadata.ContainsKey(name))
                .ToArray();

            if (missingInputs.Length == 0 && missingOutputs.Length == 0)
                return;

            throw new InvalidOperationException(
                $"The {sessionName} ONNX graph does not match genai_config.json. " +
                $"Missing inputs: [{string.Join(", ", missingInputs)}]. " +
                $"Missing outputs: [{string.Join(", ", missingOutputs)}].");
        }

        private static void ValidateNodeTypes(
            string sessionName,
            InferenceSession session,
            IReadOnlyDictionary<string, TensorElementType> expectedInputs,
            IReadOnlyDictionary<string, TensorElementType> expectedOutputs)
        {
            foreach ((string name, TensorElementType expectedType) in expectedInputs)
            {
                TensorElementType actualType = session.InputMetadata[name].ElementDataType;
                if (actualType != expectedType)
                    throw new InvalidOperationException(
                        $"The {sessionName} input '{name}' has type {actualType}; expected {expectedType}.");
            }

            foreach ((string name, TensorElementType expectedType) in expectedOutputs)
            {
                TensorElementType actualType = session.OutputMetadata[name].ElementDataType;
                if (actualType != expectedType)
                    throw new InvalidOperationException(
                        $"The {sessionName} output '{name}' has type {actualType}; expected {expectedType}.");
            }
        }


        /// <summary>
        /// Imprime la anatomía y métricas de activación de un Tensor en tiempo de ejecución.
        /// </summary>
        /// <param name="tensorLabel"></param>
        /// <param name="tensor"></param>
        private void DiagnoseTensorOutput(string tensorLabel, Tensor<float> tensor)
        {
            float[] data = [.. tensor];
            float min = float.MaxValue;
            float max = float.MinValue;
            float sum = 0f;

            for (int i = 0; i < data.Length; i++)
            {
                if (data[i] < min) min = data[i];
                if (data[i] > max) max = data[i];
                sum += data[i];
            }
            float avg = sum / data.Length;

            _logger.LogDebug("[Diag-Tensor] '{Label}' -> Dims: [{Dims}] | TotalElements: {Len} | Min: {Min:F4} | Max: {Max:F4} | Avg: {Avg:F4}",
                tensorLabel, string.Join(", ", tensor.Dimensions.ToArray()), data.Length, min, max, avg);
        }


        /// <summary>
        /// Inicializa los tensores de caché para la arquitectura de streaming. Estos tensores mantienen el estado entre las inferencias del 
        /// encoder y el decoder, lo que permite procesar el audio en tiempo real sin perder contexto. Se asignan con las dimensiones y tipos 
        /// esperados por los modelos ONNX.
        /// </summary>
        private void InitializeTensors()
        {
            int[] cacheChannelDimensions = GetRequiredStateInputDimensions(
                _encoderSession,
                _modelDefinition.Encoder.CacheLastChannelInput);
            int[] cacheTimeDimensions = GetRequiredStateInputDimensions(
                _encoderSession,
                _modelDefinition.Encoder.CacheLastTimeInput);
            int[] cacheLengthDimensions = GetRequiredStateInputDimensions(
                _encoderSession,
                _modelDefinition.Encoder.CacheLastChannelLengthInput);
            int[] decoderTargetDimensions = GetRequiredStateInputDimensions(
                _decoderSession,
                _modelDefinition.Decoder.TargetsInput);
            int[] decoderHiddenDimensions = GetRequiredStateInputDimensions(
                _decoderSession,
                _modelDefinition.Decoder.HiddenStateInput);
            int[] decoderCellDimensions = GetRequiredStateInputDimensions(
                _decoderSession,
                _modelDefinition.Decoder.CellStateInput);

            ValidateInitialStateDimensions(
                cacheChannelDimensions,
                cacheTimeDimensions,
                decoderTargetDimensions,
                decoderHiddenDimensions,
                decoderCellDimensions);

            _cacheLastChannel = new DenseTensor<float>(
                new float[GetElementCount(cacheChannelDimensions)],
                cacheChannelDimensions);
            _cacheLastTime = new DenseTensor<float>(
                new float[GetElementCount(cacheTimeDimensions)],
                cacheTimeDimensions);
            _cacheLastChannelLen = new DenseTensor<long>(
                new long[GetElementCount(cacheLengthDimensions)],
                cacheLengthDimensions);

            _decoderTargets = new DenseTensor<long>(
                new long[GetElementCount(decoderTargetDimensions)],
                decoderTargetDimensions);
            _decoderHIn = new DenseTensor<float>(
                new float[GetElementCount(decoderHiddenDimensions)],
                decoderHiddenDimensions);
            _decoderCIn = new DenseTensor<float>(
                new float[GetElementCount(decoderCellDimensions)],
                decoderCellDimensions);

            _lastPredictedToken = 0;
        }

        private static int[] GetRequiredStateInputDimensions(
            InferenceSession session,
            string inputName)
        {
            if (!session.InputMetadata.TryGetValue(inputName, out NodeMetadata? metadata))
                throw new InvalidOperationException(
                    $"Required ONNX input '{inputName}' was not found.");

            // These calls are restricted to initial streaming state tensors.
            // Dynamic batch/sequence axes start at one.
            return OnnxStateShape.ResolveInitialDimensions(
                metadata.Dimensions,
                inputName);
        }

        private void ValidateInitialStateDimensions(
            int[] cacheChannel,
            int[] cacheTime,
            int[] decoderTargets,
            int[] decoderHidden,
            int[] decoderCell)
        {
            if (cacheChannel.Length != 4 ||
                cacheChannel[^1] != _modelDefinition.EncoderHiddenSize)
            {
                throw new InvalidOperationException(
                    "Encoder cache_last_channel dimensions do not match the model definition.");
            }

            if (cacheTime.Length != 4 ||
                cacheTime[2] != _modelDefinition.EncoderHiddenSize)
            {
                throw new InvalidOperationException(
                    "Encoder cache_last_time dimensions do not match the model definition.");
            }

            if (decoderTargets.Length != 2)
                throw new InvalidOperationException(
                    "Decoder targets input must be a rank-2 tensor.");

            ValidateDecoderStateDimensions(decoderHidden, "hidden");
            ValidateDecoderStateDimensions(decoderCell, "cell");
        }

        private void ValidateDecoderStateDimensions(int[] dimensions, string stateName)
        {
            if (dimensions.Length != 3 ||
                dimensions[0] != _modelDefinition.DecoderLayerCount ||
                dimensions[^1] != _modelDefinition.DecoderHiddenSize)
            {
                throw new InvalidOperationException(
                    $"Decoder {stateName} state dimensions do not match the model definition.");
            }
        }

        private static int GetElementCount(int[] dimensions)
        {
            int count = 1;
            foreach (int dimension in dimensions)
                count = checked(count * dimension);

            return count;
        }


        /// <summary>
        /// Carga el vocabulario desde el archivo tokenizer.json y lo mapea a un diccionario de índice a token. El método es robusto para manejar 
        /// diferentes estructuras de tokenización, como listas de strings, listas de sub-arreglos o listas de objetos con propiedades. 
        /// Se espera que el archivo tokenizer.json tenga una estructura que incluya un nodo "model" con un sub-nodo "vocab" que contenga la lista
        /// de tokens. El vocabulario se almacena en un diccionario donde la clave es el índice del token (long) y el valor es el string del token 
        /// correspondiente.
        /// Si hay errores al cargar o parsear el archivo, se registrará un error crítico y se lanzará la excepción para evitar continuar con un 
        /// estado inconsistente.    
        /// </summary>
        /// <param name="path"></param>
        private void LoadTokenizer(string path)
        {
            _vocab = new Dictionary<long, string>();
            try
            {
                string jsonContent = File.ReadAllText(path);
                using (var doc = System.Text.Json.JsonDocument.Parse(jsonContent))
                {
                    var modelNode = doc.RootElement.GetProperty("model");
                    var vocabNode = modelNode.GetProperty("vocab");

                    long index = 0;
                    foreach (var item in vocabNode.EnumerateArray())
                    {
                        string token = "";

                        // Si el elemento es un Sub-Arreglo (Caso común: ["token", score])
                        if (item.ValueKind == System.Text.Json.JsonValueKind.Array)
                        {
                            if (item.GetArrayLength() > 0)
                            {
                                token = item[0].GetString() ?? "";
                            }
                        }
                        // Si el elemento es un Objeto (Caso: { "piece": "token" })
                        else if (item.ValueKind == System.Text.Json.JsonValueKind.Object)
                        {
                            if (item.TryGetProperty("piece", out var pieceNode))
                            {
                                token = pieceNode.GetString() ?? "";
                            }
                        }
                        // Si el elemento es un String directo
                        else if (item.ValueKind == System.Text.Json.JsonValueKind.String)
                        {
                            token = item.GetString() ?? "";
                        }

                        _vocab[index] = token;
                        index++;
                    }
                }
                _logger.LogInformation("Vocabulario mapeado con éxito. Total tokens: {Count}", _vocab.Count);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error crítico cargando el archivo de tokenización tokenizer.json");
                throw;
            }
        }


        /// <summary>
        /// Copies one acoustic frame from [1, T, D] into a reusable D-sized buffer.
        /// </summary>
        /// <param name="acousticOutputs"></param>
        /// <param name="frameIndex"></param>
        private static void CopySingleFrame(
            Tensor<float> acousticOutputs,
            int frameIndex,
            float[] destination)
        {
            int embeddingSize = acousticOutputs.Dimensions[^1];
            if (destination.Length != embeddingSize)
                throw new ArgumentException(
                    "Destination length must match the encoder embedding size.",
                    nameof(destination));

            for (int i = 0; i < embeddingSize; i++)
                destination[i] = acousticOutputs[0, frameIndex, i];
        }


        /// <summary>
        /// Limpia el token reconocido para eliminar caracteres especiales o de control que no deberían emitirse en la transcripción final.
        /// </summary>
        /// <param name="token"></param>
        /// <returns></returns>
        private string CleanToken(string token)
        {
            if (string.IsNullOrWhiteSpace(token))
                return "";

            if (token.StartsWith("<") || token.StartsWith("["))
                return "";

            return token
                .Replace("▁", " ")
                .Replace("##", "");
        }


        /// <summary>
        /// Devuelve el índice del valor máximo (ArgMax) en el tensor de logits
        /// </summary>
        /// <param name="logitsTensor"></param>
        private static long ArgMax(Tensor<float> logitsTensor)
        {
            bool hasValues = false;
            long maxIndex = 0;
            long index = 0;
            float maxValue = float.NegativeInfinity;

            foreach (float value in logitsTensor)
            {
                hasValues = true;
                if (value > maxValue)
                {
                    maxValue = value;
                    maxIndex = index;
                }

                index++;
            }

            return hasValues ? maxIndex : 0;
        }


        /// <summary>
        /// Actualiza los tensores de caché con los resultados del encoder para mantener el estado entre inferencias en la arquitectura de streaming.
        /// Estos tensores se pasan como entradas en la siguiente inferencia del encoder para que el modelo pueda mantener el contexto temporal y 
        /// acústico necesario para el reconocimiento de voz en tiempo real.
        /// El método extrae los tensores de caché de los resultados del encoder, los convierte a DenseTensor con las dimensiones y tipos correctos, 
        /// y los asigna a las variables de estado correspondientes en la clase. Esto es crucial para que el modelo de streaming funcione correctamente, 
        /// ya que sin esta actualización de caché, el modelo perdería el contexto entre los fragmentos de audio y no podría realizar un reconocimiento
        /// coherente. 
        /// </summary>
        /// <param name="encoderResults"></param>
        private void UpdateCache(IDisposableReadOnlyCollection<DisposableNamedOnnxValue> encoderResults)
        {
            var cacheChannelNext = encoderResults
                .First(x => x.Name == _modelDefinition.Encoder.CacheLastChannelOutput)
                .AsTensor<float>();
            var cacheTimeNext = encoderResults
                .First(x => x.Name == _modelDefinition.Encoder.CacheLastTimeOutput)
                .AsTensor<float>();
            var cacheLenNext = encoderResults
                .First(x => x.Name == _modelDefinition.Encoder.CacheLastChannelLengthOutput)
                .AsTensor<long>();

            CopyTensorValues(
                cacheChannelNext,
                _cacheLastChannel,
                "encoder channel cache");
            CopyTensorValues(
                cacheTimeNext,
                _cacheLastTime,
                "encoder time cache");
            CopyTensorValues(
                cacheLenNext,
                _cacheLastChannelLen,
                "encoder cache length");
        }


        /// <summary>
        /// Libera los recursos utilizados por las sesiones de ONNX y el StreamWriter. Es importante llamar a este método cuando se haya terminado
        ///  de usar el NemotronEngine para asegurar que no haya fugas de memoria o recursos bloqueados.
        /// El método verifica si ya se ha llamado a Dispose para evitar liberar recursos múltiples veces, lo que podría causar errores.
        /// Se disponen de las sesiones de ONNX y del StreamWriter, y se marca el estado como dispuesto para prevenir futuras llamadas a este método.
        /// </summary>
        public async Task StopAsync(
            CancellationToken cancellationToken = default)
        {
            bool firstStop =
                Interlocked.Exchange(ref _queueCompleted, 1) == 0;
            if (firstStop)
                _audioQueue.Writer.TryComplete();

            try
            {
                await _audioWorker
                    .WaitAsync(cancellationToken)
                    .ConfigureAwait(false);
            }
            catch (OperationCanceledException)
                when (cancellationToken.IsCancellationRequested)
            {
                _workerCancellation.Cancel();
                await _audioWorker.ConfigureAwait(false);
                throw;
            }
        }

        public void Dispose()
        {
            if (_isDisposed) return;
            if (Volatile.Read(ref _queueCompleted) == 0)
            {
                _audioQueue.Writer.TryComplete();
                _audioWorker.GetAwaiter().GetResult();
                Interlocked.Exchange(ref _queueCompleted, 1);
            }

            DisposeSessions();
            _workerCancellation.Dispose();
            _isDisposed = true;
        }

        public async ValueTask DisposeAsync()
        {
            if (_isDisposed) return;
            await StopAsync().ConfigureAwait(false);
            Dispose();
        }

        private void DisposeSessions()
        {
            _encoderSession?.Dispose();
            _decoderSession?.Dispose();
            _jointSession?.Dispose();
        }

        private static void CopyTensorValues<T>(
            Tensor<T> source,
            DenseTensor<T> destination,
            string stateName)
        {
            if (source.Length != destination.Length ||
                !source.Dimensions.SequenceEqual(destination.Dimensions))
            {
                throw new InvalidOperationException(
                    $"The {stateName} shape changed from " +
                    $"[{string.Join(", ", destination.Dimensions.ToArray())}] to " +
                    $"[{string.Join(", ", source.Dimensions.ToArray())}].");
            }

            Span<T> destinationValues = destination.Buffer.Span;
            int index = 0;
            foreach (T value in source)
                destinationValues[index++] = value;
        }
    }
}
