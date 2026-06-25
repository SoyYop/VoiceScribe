using System;
using System.IO;
using System.Net.Http;
using System.Text;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;

using NAudio.Wave;

using VoiceScribe.Core.Configuration;
using VoiceScribe.Core.Engine;
using VoiceScribe.Core.ModelAssets;


class Program
{
    /// <summary>
    /// Default configuration instance for the application. This provides fallback values for model file management and repository URL
    /// in case the configuration file is missing or invalid. The ModelDownloadsPath is set to a relative path within the application's
    /// base directory, and the RepoUrl points to a predefined location where the model files can be accessed.
    /// The ModelFiles list includes the expected files that the ModelDownloader will check for and download if necessary.
    /// </summary>
    private static readonly VoiceAppConfig DefaultConfig = new()
    {
        ModelDownloadsPath = Path.GetFullPath(Path.Combine(
            AppContext.BaseDirectory,
            "..", "..", "..", "..", "..",
            "artifacts", "models", "nemotron-3.5-asr")),
        RepoUrl = "https://huggingface.co/onnx-community/nemotron-3.5-asr-streaming-0.6b-onnx-int4/resolve/main",
        ModelFiles = NemotronModelFiles.CreateRequiredFileList(),
        Audio = new AudioCaptureOptions(),
        Nemotron = new NemotronModelOptions()
    };


    /// <summary>
    /// Optional StreamWriter for outputting transcripts to a file. If not initialized, transcripts will only be printed to the console.
    /// The file path can be provided as a command-line argument when starting the application. The StreamWriter is managed with proper
    /// disposal to ensure file integrity and resource cleanup.
    /// </summary>
    private static StreamWriter? _fileWriter;


    static async Task Main(string[] args)
    {
        Console.OutputEncoding = Encoding.UTF8;
        Console.WriteLine("============================================================");
        Console.WriteLine("  A 'Simple' NVIDIA Nemotron-3.5-ASR Real-Time C# Engine    ");
        Console.WriteLine("============================================================");

        // Inicialización del logging factory stand-alone
        using var loggerFactory = LoggerFactory.Create(builder =>
        {
            builder.AddConsole().SetMinimumLevel(LogLevel.Warning);
        });
        ILogger<NemotronEngine> engineLogger = loggerFactory.CreateLogger<NemotronEngine>();


        if (args.Length > 0)
        {
            try
            {
                _fileWriter = new StreamWriter(args[0], append: true, Encoding.UTF8);
                engineLogger.LogInformation($"[Config] Transcripts target file: {Path.GetFullPath(args[0])}");
            }
            catch (Exception ex)
            {
                engineLogger.LogWarning($"[Warning] File init failed: {ex.Message}. Screen output only.");
            }
        }


        var configPath = Path.Combine(AppContext.BaseDirectory, "VoiceAppConfig.json");
        var config = await VoiceAppConfig.FromJsonFileAsync(engineLogger, configPath, defaultConfig: DefaultConfig);

        if (config == null)
        {
            engineLogger.LogError("[Error] Failed to load configuration. Exiting.");
            Environment.Exit(1);
        }

        config.Audio ??= new AudioCaptureOptions();
        config.Nemotron ??= new NemotronModelOptions();

        if (config.ModelFiles == null || config.ModelFiles.Count == 0)
        {
            engineLogger.LogError("[Error] No model files specified in config. Using default list.");
            Environment.Exit(2);
        }

        if (string.IsNullOrWhiteSpace(config.ModelDownloadsPath))
        {
            engineLogger.LogError($"[Error] Model downloads path set to: {config.ModelDownloadsPath}");
            Environment.Exit(3);
        }


        if (!await EnsureModelsDownoadedAsync(engineLogger, config))
        {
            engineLogger.LogError($"[Error] Model files are missing and were not downloaded. Exiting.");
            Environment.Exit(4);
        }


        // Ejecución encapsulada del motor
        using var engine = new NemotronEngine(
            engineLogger,
            config.ModelDownloadsPath,
            config.Audio,
            config.Nemotron,
            _fileWriter);

        using var waveSource = CreateWaveSource(config.Audio);

        waveSource.DataAvailable += engine.ProcessAudioChunk;

        Console.ForegroundColor = ConsoleColor.Cyan;
        Console.WriteLine("\n>>> Microphones active. Speak clearly. Press [ENTER] to exit pipeline <<<\n");
        Console.ResetColor();

        engineLogger.LogInformation($"[Application] Starting audio capture and processing loop.");

        waveSource.StartRecording();
        Console.ReadLine();

        // Apagado síncrono e integrado
        waveSource.StopRecording();
        System.Threading.Thread.Sleep(config.Audio.ShutdownDrainMilliseconds);
        _fileWriter?.Close();

        engineLogger.LogInformation($"[Application] Ending application. Resources released, file closed.");
    }


    /// <summary>
    /// Ensures that all required model files are present in the local directory. If any files are missing, it prompts the user to 
    /// download them from the specified repository URL. If the user agrees, it uses the ModelDownloader to fetch the missing files.
    /// If the user declines, it logs an error and returns false, indicating that the application cannot proceed without the necessary
    ///  model assets.
    /// </summary>
    /// <param name="engineLogger"></param>
    /// <param name="config"></param>
    /// <returns></returns>
    private static async Task<bool> EnsureModelsDownoadedAsync(ILogger<NemotronEngine> engineLogger, VoiceAppConfig config)
    {
        ModelDownloader md = new(config.RepoUrl, config.ModelDownloadsPath!);
        if (!md.VerifyLocalWeights(config.ModelFiles))
        {
            engineLogger.LogError($"[Error] Model files are missing, asking to download.");

            Console.ForegroundColor = ConsoleColor.Yellow;
            Console.Write("\n[Missing Assets] Nemotron model layers missing. Download? (y/n): ");
            Console.ResetColor();

            if (char.ToLower(Console.ReadKey().KeyChar) == 'y')
            {
                Console.WriteLine();

                using var httpClient = new HttpClient();
                await md.HandleModelDownload(httpClient, config.ModelFiles);
                engineLogger.LogInformation($"[Config] Model files downloaded.");
            }
            else
            {
                engineLogger.LogInformation($"[Config] User declined to download Model files, exiting.");
                return false;
            }

        }
        return true;
    }


    /// <summary>
    /// Creates and configures a WaveInEvent audio source for capturing microphone input. It first checks the available audio input devices
    /// and prompts the user to select one if multiple devices are found. The selected device is then configured with a sample rate of 16 kHz,
    /// 16-bit depth, mono channel, and a buffer size of 560 milliseconds to optimize for real-time processing with the Nemotron engine. 
    /// If no devices are found, it logs an error and exits the application.
    /// </summary>
    /// <returns></returns>
    private static WaveInEvent CreateWaveSource(AudioCaptureOptions audioOptions)
    {
        int deviceNumber = SelectMicrophoneDeviceNumber();

        var waveSource = new WaveInEvent
        {
            DeviceNumber = deviceNumber,
            WaveFormat = new WaveFormat(
                audioOptions.SampleRate,
                audioOptions.BitsPerSample,
                audioOptions.Channels),
            BufferMilliseconds = audioOptions.BufferMilliseconds
        };

        return waveSource;
    }


    /// <summary>
    /// Prompts the user to select an audio input device if multiple microphones are detected. It lists all available devices with their names
    /// and channel counts, and validates the user's selection to ensure it corresponds to a valid device number. 
    /// If only one device is found, it is automatically selected. If no devices are found, it logs an error and exits the application.
    /// The method returns the selected device number for use in configuring the audio source.
    /// </summary>
    /// <returns>Input device ID</returns>
    private static int SelectMicrophoneDeviceNumber()
    {
        int deviceCount = WaveIn.DeviceCount;

        if (deviceCount == 0)
        {
            Console.ForegroundColor = ConsoleColor.Red;
            Console.WriteLine("[Audio] No input microphones found.");
            Console.ResetColor();

            Environment.Exit(10);
        }

        if (deviceCount == 1)
        {
            var onlyDevice = WaveIn.GetCapabilities(0);

            Console.ForegroundColor = ConsoleColor.Green;
            Console.WriteLine($"[Audio] Microphone selected: 0 - {onlyDevice.ProductName}");
            Console.ResetColor();

            return 0;
        }

        Console.ForegroundColor = ConsoleColor.Yellow;
        Console.WriteLine("\n[Audio] Multiple input microphones found:");
        Console.ResetColor();

        for (int i = 0; i < deviceCount; i++)
        {
            var device = WaveIn.GetCapabilities(i);
            Console.WriteLine($"  {i}: {device.ProductName} - Channels: {device.Channels}");
        }

        while (true)
        {
            Console.ForegroundColor = ConsoleColor.Yellow;
            Console.Write("\nSelect microphone number: ");
            Console.ResetColor();

            string? input = Console.ReadLine();

            if (int.TryParse(input, out int selectedDevice)
                && selectedDevice >= 0
                && selectedDevice < deviceCount)
            {
                var device = WaveIn.GetCapabilities(selectedDevice);

                Console.ForegroundColor = ConsoleColor.Green;
                Console.WriteLine($"[Audio] Microphone selected: {selectedDevice} - {device.ProductName}");
                Console.ResetColor();

                return selectedDevice;
            }

            Console.ForegroundColor = ConsoleColor.Red;
            Console.WriteLine("[Audio] Invalid microphone number. Try again.");
            Console.ResetColor();
        }
    }

}
