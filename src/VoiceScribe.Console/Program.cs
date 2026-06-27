using System;
using System.IO;
using System.Net.Http;
using System.Text;
using System.Threading.Tasks;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Logging;

using NAudio.Wave;

using VoiceScribe.Console.Audio;
using VoiceScribe.Console.Benchmark;
using VoiceScribe.Console.CommandLine;
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


    static async Task<int> Main(string[] args)
    {
        Console.OutputEncoding = Encoding.UTF8;
        RunOptions runOptions = RunOptionsParser.Parse(args);
        Console.WriteLine("============================================================");
        Console.WriteLine("  A 'Simple' NVIDIA Nemotron-3.5-ASR Real-Time C# Engine    ");
        Console.WriteLine("============================================================");

        IConfiguration appSettings = new ConfigurationBuilder()
            .SetBasePath(AppContext.BaseDirectory)
            .AddJsonFile("appsettings.json", optional: true, reloadOnChange: false)
            .Build();

        // Inicialización del logging factory stand-alone
        using var loggerFactory = LoggerFactory.Create(builder =>
        {
            builder.AddConfiguration(appSettings.GetSection("Logging"));
            builder.AddConsole();
        });
        ILogger<NemotronEngine> engineLogger = loggerFactory.CreateLogger<NemotronEngine>();

        StreamWriter? transcriptWriter = null;

        if (runOptions.TranscriptPath is not null)
        {
            try
            {
                transcriptWriter = new StreamWriter(runOptions.TranscriptPath, append: true, Encoding.UTF8);
                engineLogger.LogInformation($"[Config] Transcripts target file: {Path.GetFullPath(runOptions.TranscriptPath)}");
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

        if (runOptions.Benchmark)
            config.Audio.QueueCapacity = Math.Max(
                config.Audio.QueueCapacity,
                runOptions.BenchmarkChunks);

        IReadOnlyList<AudioInputDevice> audioDevices = ConsoleAudioInput.GetDevices();
        if (!runOptions.Benchmark && audioDevices.Count == 0)
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

        try
        {
            if (!runOptions.Benchmark &&
                DirectMlAdapterSelector.IsDirectMlRequested(config.Inference) &&
                OnnxRuntimeVariant.Supports(OnnxExecutionProvider.DirectMl))
            {
                IReadOnlyList<DirectMlAdapter> adapters =
                    DirectMlAdapterSelector.GetAdapters();
                config.Inference.DeviceId =
                    DirectMlAdapterSelector.SelectDeviceId(
                        adapters,
                        config.Inference.DeviceId,
                        shutdownCts.Token);
            }

            NemotronOnnxSessionFactories sessionFactories =
                OnnxSessionFactoryResolver.CreateForNemotron(
                    config.Inference,
                    engineLogger);

            PrintInferenceConfiguration(config.Inference, sessionFactories);

            Console.ForegroundColor = ConsoleColor.DarkGray;
            Console.WriteLine(
                $"[Model] Loading ONNX models with " +
                $"{sessionFactories.Describe()}...");
            Console.ResetColor();

            await using NemotronEngine engine =
                new(
                    engineLogger,
                    config.ModelDownloadsPath,
                    config.Audio,
                    config.Nemotron,
                    modelDefinition,
                    sessionFactories,
                    transcriptWriter);
            shutdownCts.Token.ThrowIfCancellationRequested();

            Console.ForegroundColor = ConsoleColor.DarkGray;
            Console.WriteLine("[Model] ONNX models ready.");
            Console.ResetColor();

            if (runOptions.Benchmark)
            {
                await SyntheticBenchmarkRunner.RunAsync(
                    engine,
                    modelDefinition,
                    config.Audio,
                    runOptions.BenchmarkChunks,
                    shutdownCts.Token);

                engineLogger.LogInformation("[Application] Synthetic benchmark complete.");
                return 0;
            }

            int deviceNumber = ConsoleAudioInput.SelectDeviceNumber(
                audioDevices,
                shutdownCts.Token);

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
            engineLogger.LogInformation("[Application] Shutdown cancelled.");
            return 130;
        }
        finally
        {
            Console.CancelKeyPress -= cancelHandler;
            if (transcriptWriter != null)
                await transcriptWriter.DisposeAsync();
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

    private static void PrintInferenceConfiguration(
        OnnxRuntimeOptions options,
        NemotronOnnxSessionFactories sessionFactories)
    {
        Console.ForegroundColor = ConsoleColor.DarkGray;
        Console.WriteLine("[Inference] Current configuration:");
        Console.WriteLine($"  Runtime flavor     : {OnnxRuntimeVariant.Name}");
        Console.WriteLine($"  Default provider   : {options.ExecutionProvider}");
        Console.WriteLine($"  Encoder provider   : {sessionFactories.Encoder.ExecutionProvider}");
        Console.WriteLine($"  Decoder provider   : {sessionFactories.Decoder.ExecutionProvider}");
        Console.WriteLine($"  Joiner provider    : {sessionFactories.Joiner.ExecutionProvider}");
        Console.WriteLine($"  Device id          : {options.DeviceId}");
        Console.WriteLine($"  CPU fallback       : {options.AllowCpuFallback}");
        Console.WriteLine($"  Profiling          : {options.EnableProfiling}");
        Console.WriteLine($"  ORT log severity   : {options.LogSeverityLevel ?? "default"}");
        Console.WriteLine(
            $"  ORT log verbosity  : " +
            $"{(options.LogVerbosityLevel.HasValue ? options.LogVerbosityLevel.Value.ToString() : "default")}");
        Console.WriteLine(
            $"  GPU memory limit   : " +
            $"{(options.GpuMemoryLimitMiB.HasValue ? $"{options.GpuMemoryLimitMiB} MiB" : "not set")}");

        if ((IsGpuProvider(sessionFactories.Encoder.ExecutionProvider) ||
             IsGpuProvider(sessionFactories.Decoder.ExecutionProvider) ||
             IsGpuProvider(sessionFactories.Joiner.ExecutionProvider)) &&
            options.AllowCpuFallback)
        {
            Console.WriteLine(
                "  Note               : individual models may fall back to CPU if the GPU provider cannot initialize them.");
        }

        Console.ResetColor();
    }

    private static bool IsGpuProvider(OnnxExecutionProvider provider) =>
        provider is OnnxExecutionProvider.DirectMl or OnnxExecutionProvider.Cuda;

}
