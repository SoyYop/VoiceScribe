using Microsoft.Extensions.Logging;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

using NAudio.Wave;

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

using VoiceScribe.Core.Audio;

namespace VoiceScribe.Core.Engine
{
    public class NemotronEngine : IDisposable
    {
        private readonly ILogger<NemotronEngine> _logger;
        private InferenceSession _encoderSession = null!;
        private InferenceSession _decoderSession = null!;
        private InferenceSession _jointSession = null!;
        private Dictionary<long, string> _vocab = null!;
        private StreamWriter? _fileWriter;
        private AudioFeatureExtractor _featureExtractor = null!;
        private bool _isDisposed;

        // Estados del búfer acústico (Cachés de streaming)
        private DenseTensor<float> _cacheLastChannel = null!;
        private DenseTensor<float> _cacheLastTime = null!;
        private DenseTensor<long> _cacheLastChannelLen = null!;

        // Estados del búfer lingüístico (Decoder / Transducer)
        private DenseTensor<long> _decoderTargets = null!;
        private DenseTensor<float> _decoderHIn = null!;
        private DenseTensor<float> _decoderCIn = null!;
        private long _lastPredictedToken = -1;
        private const long BlankId = 13087;
        private const int MaxSymbolsPerStep = 10;

        public NemotronEngine(ILogger<NemotronEngine> logger)
        {
            _logger = logger;
        }



        public NemotronEngine(ILogger<NemotronEngine> logger, string modelFolderPath, StreamWriter? fileWriter)
        {
            _logger = logger;
            _fileWriter = fileWriter; // Asignamos el escritor que viene desde Main

            // Calculamos de forma automática la ruta del tokenizador e inicializamos el grafo
            string tokenizerJsonPath = Path.Combine(modelFolderPath, "tokenizer.json");
            Initialize(modelFolderPath, tokenizerJsonPath);
        }



        /// <summary>
        /// Inicializa las sesiones de ONNX Runtime y ejecuta los diagnósticos.
        /// </summary>
        public void Initialize(string modelFolderPath, string tokenizerJsonPath)
        {
            _logger.LogInformation("Bootstrapping ONNX Execution Runtimes...");

            var encoderPath = Path.Combine(modelFolderPath, "encoder.onnx");
            var decoderPath = Path.Combine(modelFolderPath, "decoder.onnx");
            var jointPath = Path.Combine(modelFolderPath, "joint.onnx");

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


            var audioProcessorConfigPath = Path.Combine(modelFolderPath, "audio_processor_config.json");
            var genaiConfigPath = Path.Combine(modelFolderPath, "genai_config.json");

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
        public void ProcessAudioChunk(object? sender, WaveInEventArgs e)
        {
            if (_isDisposed || _encoderSession == null) return;

            try
            {
                const int expectedSamples = 8960;

                if (e.BytesRecorded < expectedSamples * 2)
                    return;

                float[] pcmChunk = new float[expectedSamples];
                float maxAmp = 0f;

                for (int i = 0; i < expectedSamples; i++)
                {
                    short pcm = BitConverter.ToInt16(e.Buffer, i * 2);
                    float sample = pcm / 32768f;

                    pcmChunk[i] = sample;
                    maxAmp = Math.Max(maxAmp, Math.Abs(sample));
                }

                if (maxAmp < 0.003f)
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
                    new long[] { 101 },
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



        private void DecodeRnntFrames(Tensor<float> acoustic, int frameCount)
        {
            for (int t = 0; t < frameCount; t++)
            {
                var encFrame = ExtractSingleFrame(acoustic, t);

                for (int u = 0; u < MaxSymbolsPerStep; u++)
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

                    var decOutReshaped = new DenseTensor<float>(
                        decOutTensor.ToArray(),
                        new[] { 1, 1, 640 }
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

                    if (token == BlankId)
                        break;

                    _decoderHIn = new DenseTensor<float>(hOut.ToArray(), hOut.Dimensions.ToArray());
                    _decoderCIn = new DenseTensor<float>(cOut.ToArray(), cOut.Dimensions.ToArray());
                    _decoderTargets = new DenseTensor<long>(new long[] { token }, new[] { 1, 1 });

                    EmitToken(token);
                }
            }
        }



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
        private void DiagnoseTensorOutput(string tensorLabel, Tensor<float> tensor)
        {
            float[] data = tensor.ToArray();
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

        // ====================================================================
        // MÉTODOS DE INFRAESTRUCTURA DE DATOS INTERNOS
        // ====================================================================
        private void InitializeTensors()
        {
            _cacheLastChannel = new DenseTensor<float>(new float[1 * 24 * 56 * 1024], new[] { 1, 24, 56, 1024 });
            _cacheLastTime = new DenseTensor<float>(new float[1 * 24 * 1024 * 8], new[] { 1, 24, 1024, 8 });
            _cacheLastChannelLen = new DenseTensor<long>(new long[] { 0 }, new[] { 1 });

            // Estados del decoder - más robusto
            _decoderTargets = new DenseTensor<long>(new long[] { 0 }, new[] { 1, 1 });
            _decoderHIn = new DenseTensor<float>(new float[2 * 1 * 640], new[] { 2, 1, 640 });
            _decoderCIn = new DenseTensor<float>(new float[2 * 1 * 640], new[] { 2, 1, 640 });

            _lastPredictedToken = 0;
        }


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
        private DenseTensor<float> ExtractSingleFrame(Tensor<float> acousticOutputs, int frameIndex)
        {
            var full = acousticOutputs.ToArray();
            float[] frame = new float[1024];
            int offset = frameIndex * 1024;
            Array.Copy(full, offset, frame, 0, 1024);
            return new DenseTensor<float>(frame, new[] { 1, 1, 1024 });
        }



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


        private void UpdateCache(IDisposableReadOnlyCollection<DisposableNamedOnnxValue> encoderResults)
        {
            var cacheChannelNext = encoderResults.First(x => x.Name == "cache_last_channel_next").AsTensor<float>();
            var cacheTimeNext = encoderResults.First(x => x.Name == "cache_last_time_next").AsTensor<float>();
            var cacheLenNext = encoderResults.First(x => x.Name == "cache_last_channel_len_next").AsTensor<long>();

            _cacheLastChannel = new DenseTensor<float>(cacheChannelNext.ToArray(), cacheChannelNext.Dimensions.ToArray());
            _cacheLastTime = new DenseTensor<float>(cacheTimeNext.ToArray(), cacheTimeNext.Dimensions.ToArray());
            _cacheLastChannelLen = new DenseTensor<long>(cacheLenNext.ToArray(), cacheLenNext.Dimensions.ToArray());
        }


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
