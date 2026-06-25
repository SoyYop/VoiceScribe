using Microsoft.Extensions.Logging;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

using NAudio.Wave;

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
    public class NemotronEngine : IDisposable
    {
        private readonly ILogger<NemotronEngine> _logger;
        private readonly AudioCaptureOptions _audioOptions;
        private readonly NemotronModelOptions _modelOptions;

        private StreamWriter? _fileWriter;

        private bool _isDisposed;

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


        public NemotronEngine(ILogger<NemotronEngine> logger)
        {
            _logger = logger;
            _audioOptions = new AudioCaptureOptions();
            _modelOptions = new NemotronModelOptions();
        }


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
            StreamWriter? fileWriter)
        {
            _logger = logger;
            _audioOptions = audioOptions;
            _modelOptions = modelOptions;
            _fileWriter = fileWriter; // Asignamos el escritor que viene desde Main

            // Calculamos de forma automática la ruta del tokenizador e inicializamos el grafo
            string tokenizerJsonPath = Path.Combine(modelFolderPath, NemotronModelFiles.Tokenizer);
            Initialize(modelFolderPath, tokenizerJsonPath);
        }


        /// <summary>
        /// Inicializa las sesiones de ONNX Runtime y ejecuta los diagnósticos.
        /// </summary>
        public void Initialize(string modelFolderPath, string tokenizerJsonPath)
        {
            _logger.LogInformation("Bootstrapping ONNX Execution Runtimes...");

            var encoderPath = Path.Combine(modelFolderPath, NemotronModelFiles.Encoder);
            var decoderPath = Path.Combine(modelFolderPath, NemotronModelFiles.Decoder);
            var jointPath = Path.Combine(modelFolderPath, NemotronModelFiles.Joint);

            var sessionOptions = new SessionOptions
            {
                GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
                ExecutionMode = ExecutionMode.ORT_SEQUENTIAL
            };

            _encoderSession = new InferenceSession(encoderPath, sessionOptions);
            _decoderSession = new InferenceSession(decoderPath, sessionOptions);
            _jointSession = new InferenceSession(jointPath, sessionOptions);

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
            if (_isDisposed || _encoderSession == null) return;

            try
            {
                int expectedSamples = _audioOptions.SamplesPerBuffer;

                int bytesPerSample = _audioOptions.BitsPerSample / 8;
                int bytesPerFrame = bytesPerSample * _audioOptions.Channels;
                int expectedBytes = expectedSamples * bytesPerFrame;

                if (e.BytesRecorded < expectedBytes)
                    return;

                float[] pcmChunk = new float[expectedSamples];
                float maxAmp = 0f;

                for (int i = 0; i < expectedSamples; i++)
                {
                    int sampleOffset = i * bytesPerFrame;
                    float sample = ReadPcmSample(e.Buffer, sampleOffset, _audioOptions.BitsPerSample);

                    pcmChunk[i] = sample;
                    maxAmp = Math.Max(maxAmp, Math.Abs(sample));
                }

                if (maxAmp < _audioOptions.SilenceThreshold)
                    return;

                // IMPORTANTE:
                // Aquí NO se debe hacer reshape directo del PCM.
                // Aquí debe ir PCM -> log-mel.
                var features = _featureExtractor.Extract(pcmChunk);

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
                    NamedOnnxValue.CreateFromTensor("audio_signal", audioTensor),
                    NamedOnnxValue.CreateFromTensor("length", lengthTensor),
                    NamedOnnxValue.CreateFromTensor("cache_last_channel", _cacheLastChannel),
                    NamedOnnxValue.CreateFromTensor("cache_last_time", _cacheLastTime),
                    NamedOnnxValue.CreateFromTensor("cache_last_channel_len", _cacheLastChannelLen),
                    NamedOnnxValue.CreateFromTensor("lang_id", langIdTensor)
                };

                using var encoderResults = _encoderSession.Run(encoderInputs);

                var acoustic = encoderResults
                    .First(x => x.Name == "outputs")
                    .AsTensor<float>();

                int T = acoustic.Dimensions[1];

                UpdateCache(encoderResults);

                DecodeRnntFrames(acoustic, T);

                _fileWriter?.Flush();
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error en ProcessAudioChunk");
                Console.WriteLine("ERROR: " + ex.Message);
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
        private void DecodeRnntFrames(Tensor<float> acoustic, int frameCount)
        {
            for (int t = 0; t < frameCount; t++)
            {
                var encFrame = ExtractSingleFrame(acoustic, t);

                for (int u = 0; u < _modelOptions.MaxSymbolsPerStep; u++)
                {
                    var decInputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("targets", _decoderTargets),
                NamedOnnxValue.CreateFromTensor("h_in", _decoderHIn),
                NamedOnnxValue.CreateFromTensor("c_in", _decoderCIn)
            };

                    using var decResults = _decoderSession.Run(decInputs);

                    var decOutTensor = decResults
                        .First(x => x.Name == "decoder_output")
                        .AsTensor<float>();

                    var hOut = decResults.First(x => x.Name == "h_out").AsTensor<float>();
                    var cOut = decResults.First(x => x.Name == "c_out").AsTensor<float>();

                    int decoderEmbeddingSize = checked((int)decOutTensor.Length);
                    var decOutReshaped = new DenseTensor<float>(
                        decOutTensor.ToArray(),
                        new[] { 1, 1, decoderEmbeddingSize }
                    );

                    var jointInputs = new List<NamedOnnxValue>
                    {
                        NamedOnnxValue.CreateFromTensor("encoder_output", encFrame),
                        NamedOnnxValue.CreateFromTensor("decoder_output", decOutReshaped)
                    };

                    using var jointResults = _jointSession.Run(jointInputs);

                    var jointOut = jointResults
                        .First(x => x.Name == "joint_output")
                        .AsTensor<float>();

                    long token = ArgMax(jointOut);

                    long blankId = ResolveBlankId(jointOut);
                    if (token == blankId)
                        break;

                    _decoderHIn = new DenseTensor<float>(hOut.ToArray(), hOut.Dimensions.ToArray());
                    _decoderCIn = new DenseTensor<float>(cOut.ToArray(), cOut.Dimensions.ToArray());
                    _decoderTargets = new DenseTensor<long>(
                        new long[] { token },
                        _decoderTargets.Dimensions.ToArray());

                    EmitToken(token);
                }
            }
        }

        private long ResolveBlankId(Tensor<float> jointOutput)
        {
            int classCount = jointOutput.Dimensions[^1];
            long blankId = _modelOptions.BlankId ?? classCount - 1;

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
            _fileWriter?.Write(clean);

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
            int[] cacheChannelDimensions = GetRequiredInputDimensions(
                _encoderSession,
                "cache_last_channel");
            int[] cacheTimeDimensions = GetRequiredInputDimensions(
                _encoderSession,
                "cache_last_time");
            int[] cacheLengthDimensions = GetRequiredInputDimensions(
                _encoderSession,
                "cache_last_channel_len");
            int[] decoderTargetDimensions = GetRequiredInputDimensions(
                _decoderSession,
                "targets");
            int[] decoderHiddenDimensions = GetRequiredInputDimensions(
                _decoderSession,
                "h_in");
            int[] decoderCellDimensions = GetRequiredInputDimensions(
                _decoderSession,
                "c_in");

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

        private static int[] GetRequiredInputDimensions(
            InferenceSession session,
            string inputName)
        {
            if (!session.InputMetadata.TryGetValue(inputName, out NodeMetadata? metadata))
                throw new InvalidOperationException(
                    $"Required ONNX input '{inputName}' was not found.");

            int[] dimensions = metadata.Dimensions.ToArray();
            if (dimensions.Length == 0)
                throw new InvalidOperationException(
                    $"Required ONNX input '{inputName}' has no tensor dimensions.");

            for (int i = 0; i < dimensions.Length; i++)
            {
                if (dimensions[i] <= 0)
                    // Streaming inference starts with one batch item and one
                    // sequence/state item for every symbolic ONNX dimension.
                    dimensions[i] = 1;
            }

            return dimensions;
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
        /// Extrae un solo frame acústico del encoder [1, T, 1024] → [1, 1, 1024]
        /// </summary>
        /// <param name="acousticOutputs"></param>
        /// <param name="frameIndex"></param>
        private DenseTensor<float> ExtractSingleFrame(Tensor<float> acousticOutputs, int frameIndex)
        {
            int embeddingSize = acousticOutputs.Dimensions[^1];
            var full = acousticOutputs.ToArray();
            float[] frame = new float[embeddingSize];
            int offset = frameIndex * embeddingSize;
            Array.Copy(full, offset, frame, 0, embeddingSize);
            return new DenseTensor<float>(frame, new[] { 1, 1, embeddingSize });
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
        private long ArgMax(Tensor<float> logitsTensor)
        {
            var data = logitsTensor.ToArray();
            if (data.Length == 0) return 0;

            long maxIndex = 0;
            float maxValue = data[0];

            for (int i = 1; i < data.Length; i++)
            {
                if (data[i] > maxValue)
                {
                    maxValue = data[i];
                    maxIndex = i;
                }
            }
            return maxIndex;
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
            var cacheChannelNext = encoderResults.First(x => x.Name == "cache_last_channel_next").AsTensor<float>();
            var cacheTimeNext = encoderResults.First(x => x.Name == "cache_last_time_next").AsTensor<float>();
            var cacheLenNext = encoderResults.First(x => x.Name == "cache_last_channel_len_next").AsTensor<long>();

            _cacheLastChannel = new DenseTensor<float>(cacheChannelNext.ToArray(), cacheChannelNext.Dimensions.ToArray());
            _cacheLastTime = new DenseTensor<float>(cacheTimeNext.ToArray(), cacheTimeNext.Dimensions.ToArray());
            _cacheLastChannelLen = new DenseTensor<long>(cacheLenNext.ToArray(), cacheLenNext.Dimensions.ToArray());
        }


        /// <summary>
        /// Libera los recursos utilizados por las sesiones de ONNX y el StreamWriter. Es importante llamar a este método cuando se haya terminado
        ///  de usar el NemotronEngine para asegurar que no haya fugas de memoria o recursos bloqueados.
        /// El método verifica si ya se ha llamado a Dispose para evitar liberar recursos múltiples veces, lo que podría causar errores.
        /// Se disponen de las sesiones de ONNX y del StreamWriter, y se marca el estado como dispuesto para prevenir futuras llamadas a este método.
        /// </summary>
        public void Dispose()
        {
            if (_isDisposed) return;
            _fileWriter?.Dispose();
            _encoderSession?.Dispose();
            _decoderSession?.Dispose();
            _jointSession?.Dispose();
            _isDisposed = true;
        }
    }
}
