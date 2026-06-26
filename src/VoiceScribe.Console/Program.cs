using System;
using System.IO;
using System.Net.Http;
using System.Text;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;

using NAudio.Wave;

using VoiceScribe.Console.Audio;
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
        Nemotron = new NemotronModelOptions(),
        Inference = new OnnxRuntimeOptions()
    };


    /// <summary>
    /// Optional StreamWriter for outputting transcripts to a file. If not initialized, transcripts will only be printed to the console.
    /// The file path can be provided as a command-line argument when starting the application. The StreamWriter is managed with proper
    /// disposal to ensure file integrity and resource cleanup.
    /// </summary>
    private static StreamWriter? _fileWriter;


    static async Task<int> Main(string[] args)
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
            return 1;
        }

        config.Audio ??= new AudioCaptureOptions();
        config.Nemotron ??= new NemotronModelOptions();
        config.Inference ??= new OnnxRuntimeOptions();

        if (config.ModelFiles == null || config.ModelFiles.Count == 0)
        {
            engineLogger.LogError("[Error] No model files specified in config. Using default list.");
            return 2;
        }

        if (string.IsNullOrWhiteSpace(config.ModelDownloadsPath))
        {
            engineLogger.LogError($"[Error] Model downloads path set to: {config.ModelDownloadsPath}");
            return 3;
        }


        if (!await EnsureModelsDownloadedAsync(engineLogger, config))
        {
            engineLogger.LogError($"[Error] Model files are missing and were not downloaded. Exiting.");
            return 4;
        }

        NemotronModelDefinition modelDefinition;
        try
        {
            string modelConfigPath = Path.Combine(
                config.ModelDownloadsPath,
                NemotronModelFiles.GenAiConfig);
            modelDefinition = NemotronModelDefinition.Load(modelConfigPath);
        }
        catch (Exception ex)
        {
            engineLogger.LogError(ex, "Failed to load Nemotron model definition.");
            return 5;
        }

        IReadOnlyList<string> configurationErrors =
            VoiceAppConfigValidator.Validate(config, modelDefinition);
        if (configurationErrors.Count > 0)
        {
            foreach (string error in configurationErrors)
                engineLogger.LogError("[Configuration] {Error}", error);

            return 6;
        }

        IReadOnlyList<AudioInputDevice> audioDevices = ConsoleAudioInput.GetDevices();
        if (audioDevices.Count == 0)
        {
            Console.ForegroundColor = ConsoleColor.Red;
            Console.WriteLine("[Audio] No input microphones found.");
            Console.ResetColor();
            return 10;
        }

        using var shutdownCts = new CancellationTokenSource();
        ConsoleCancelEventHandler cancelHandler = (_, eventArgs) =>
        {
            eventArgs.Cancel = true;
            shutdownCts.Cancel();
        };
        Console.CancelKeyPress += cancelHandler;
        Task<NemotronEngine>? engineLoadTask = null;

        try
        {
            IOnnxSessionFactory sessionFactory =
                OnnxSessionFactoryResolver.Create(
                    config.Inference,
                    engineLogger);

            Console.ForegroundColor = ConsoleColor.DarkGray;
            Console.WriteLine(
                $"[Model] Loading ONNX models with " +
                $"{sessionFactory.ExecutionProvider} in background...");
            Console.ResetColor();

            engineLoadTask = Task.Run(
                () => new NemotronEngine(
                    engineLogger,
                    config.ModelDownloadsPath,
                    config.Audio,
                    config.Nemotron,
                    modelDefinition,
                    sessionFactory,
                    _fileWriter));

            int deviceNumber = ConsoleAudioInput.SelectDeviceNumber(
                audioDevices,
                shutdownCts.Token);
            await using NemotronEngine engine =
                await engineLoadTask.ConfigureAwait(false);
            shutdownCts.Token.ThrowIfCancellationRequested();

            Console.ForegroundColor = ConsoleColor.DarkGray;
            Console.WriteLine("[Model] ONNX models ready.");
            Console.ResetColor();

            using WaveInEvent waveSource =
                ConsoleAudioInput.CreateWaveSource(deviceNumber, config.Audio);

            waveSource.DataAvailable += engine.ProcessAudioChunk;

            Console.ForegroundColor = ConsoleColor.Cyan;
            Console.WriteLine("\n>>> Microphones active. Speak clearly. Press [ENTER] to exit pipeline <<<\n");
            Console.ResetColor();

            engineLogger.LogInformation("[Application] Starting audio capture and processing loop.");

            waveSource.StartRecording();
            try
            {
                await Console.In.ReadLineAsync(shutdownCts.Token);
            }
            catch (OperationCanceledException)
            {
                engineLogger.LogInformation(
                    "[Application] Cancellation requested.");
            }

            waveSource.DataAvailable -= engine.ProcessAudioChunk;
            waveSource.StopRecording();
            await engine.StopAsync(shutdownCts.Token);

            engineLogger.LogInformation("[Application] Ending application. Resources released.");
            return 0;
        }
        catch (OperationCanceledException)
        {
            if (engineLoadTask != null)
            {
                try
                {
                    await using NemotronEngine loadedEngine =
                        await engineLoadTask.ConfigureAwait(false);
                }
                catch
                {
                    // The original cancellation result is returned below.
                }
            }

            engineLogger.LogInformation("[Application] Shutdown cancelled.");
            return 130;
        }
        finally
        {
            Console.CancelKeyPress -= cancelHandler;
            if (_fileWriter != null)
                await _fileWriter.DisposeAsync();
        }
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
    private static async Task<bool> EnsureModelsDownloadedAsync(ILogger<NemotronEngine> engineLogger, VoiceAppConfig config)
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

                using var downloadCts = new CancellationTokenSource();
                ConsoleCancelEventHandler cancelDownload = (_, eventArgs) =>
                {
                    eventArgs.Cancel = true;
                    downloadCts.Cancel();
                };
                Console.CancelKeyPress += cancelDownload;

                using var httpClient = new HttpClient();
                try
                {
                    await md.HandleModelDownload(
                        httpClient,
                        config.ModelFiles,
                        downloadCts.Token);
                    engineLogger.LogInformation("[Config] Model files downloaded.");
                }
                catch (OperationCanceledException)
                {
                    engineLogger.LogWarning("[Config] Model download cancelled.");
                    return false;
                }
                finally
                {
                    Console.CancelKeyPress -= cancelDownload;
                }
            }
            else
            {
                engineLogger.LogInformation($"[Config] User declined to download Model files, exiting.");
                return false;
            }

        }
        return true;
    }

}
