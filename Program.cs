using System;
using System.IO;
using System.Net.Http;
using System.Text;
using System.Threading.Tasks;

using Microsoft.Extensions.Logging;

using NAudio.Wave;

using VoiceScribe;

class Program
{
    private static readonly SpeechAppConfig DefaultConfig = new()
    {
        ModelDownloadsPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "artifacts", "models", "nemotron-3.5-asr"),
        RepoUrl = "https://huggingface.co/onnx-community/nemotron-3.5-asr-streaming-0.6b-onnx-int4/resolve/main",
        ModelFiles = [
            "encoder.onnx",
            "encoder.onnx.data",
            "decoder.onnx", "decoder.onnx.data",
            "joint.onnx", "joint.onnx.data",
            "tokenizer.json",
            "genai_config.json",
            "audio_processor_config.json",
            "model_config.json"
        ]
    };
    
    
    private static StreamWriter? _fileWriter;

    private static readonly HttpClient _httpClient = new HttpClient();

    static async Task Main(string[] args)
    {
        Console.OutputEncoding = Encoding.UTF8;

        // Inicialización de Fábrica de Logging portable
        using var loggerFactory = LoggerFactory.Create(builder =>
        {
            builder.AddConsole().SetMinimumLevel(LogLevel.Warning);
        });
        ILogger<NemotronEngine> engineLogger = loggerFactory.CreateLogger<NemotronEngine>();

        Console.WriteLine("==================================================");
        Console.WriteLine("  NVIDIA Nemotron-3.5-ASR Real-Time C# Engine     ");
        Console.WriteLine("==================================================");

        if (args.Length > 0)
        {
            try
            {
                _fileWriter = new StreamWriter(args[0], append: true, Encoding.UTF8);
                engineLogger.LogWarning($"[Config] Transcripts target file: {Path.GetFullPath(args[0])}");
            }
            catch (Exception ex)
            {
                engineLogger.LogWarning($"[Warning] File init failed: {ex.Message}. Screen output only.");
            }
        }

        var config = await SpeechAppConfig.FromJsonFileAsync(engineLogger, "SpeechAppConfig.json", defaultConfig: DefaultConfig);

        if (config == null)
        {
            engineLogger.LogError("[Error] Failed to load configuration. Exiting.");
            Environment.Exit(1);
        }

        if (config.ModelFiles == null || config.ModelFiles.Count == 0)
        {
            engineLogger.LogWarning("[Warning] No model files specified in config. Using default list.");
            Environment.Exit(2);
        }

        if (string.IsNullOrWhiteSpace(config.ModelDownloadsPath))
        {
            engineLogger.LogWarning($"[Config] Model downloads path set to: {config.ModelDownloadsPath}");
            Environment.Exit(3);
        }


        ModelDownloader md = new(_httpClient, config.RepoUrl, config.ModelDownloadsPath);
        if (!md.VerifyLocalWeights(config.ModelFiles))
        {
            engineLogger.LogWarning($"[Config] Model files are missing, asking to download.");

            Console.ForegroundColor = ConsoleColor.Yellow;
            Console.Write("\n[Missing Assets] Nemotron model layers missing. Download? (y/n): ");
            Console.ResetColor();

            if (char.ToLower(Console.ReadKey().KeyChar) == 'y')
            {
                Console.WriteLine();
                await md.HandleModelDownload(config.ModelFiles);
                engineLogger.LogInformation($"[Config] Model files downloaded.");
            }
            else {
                engineLogger.LogInformation($"[Config] User declined to download Model files, exiting.");
                return;
            }
        }

        // Ejecución encapsulada del motor
        using var engine = new NemotronEngine(engineLogger, config.ModelDownloadsPath, _fileWriter);

        using var waveSource = new WaveInEvent
        {
            DeviceNumber = 0,
            WaveFormat = new WaveFormat(16000, 16, 1),
            BufferMilliseconds = 560
        };

        waveSource.DataAvailable += engine.ProcessAudioChunk;

        Console.ForegroundColor = ConsoleColor.Cyan;
        Console.WriteLine("\n>>> Microphones active. Speak clearly. Press [ENTER] to exit pipeline <<<\n");
        Console.ResetColor();

        engineLogger.LogInformation($"[Application] Starting audio capture and processing loop.");

        waveSource.StartRecording();
        Console.ReadLine();

        // Apagado síncrono e integrado
        waveSource.StopRecording();
        System.Threading.Thread.Sleep(250); // Tiempo de drenaje de tramas pendientes
        _fileWriter?.Close();

        engineLogger.LogInformation($"[Application] Ending application. Resources released, file closed.");
    }
}
