using System;
using System.IO;
using System.Text;
using System.Threading.Tasks;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Logging;

using NAudio.Wave;

using VoiceScribe.Console;
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

        using ILoggerFactory loggerFactory = CreateLoggerFactory();
        ILogger<NemotronEngine> engineLogger = loggerFactory.CreateLogger<NemotronEngine>();

        StreamWriter? transcriptWriter =
            OpenTranscriptWriter(runOptions, engineLogger);


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
            engineLogger.LogError(
                "[Error] Model downloads path set to: {Path}",
                config.ModelDownloadsPath);
            return 3;
        }


        if (!await ModelAssetBootstrapper.EnsureDownloadedAsync(engineLogger, config))
        {
            engineLogger.LogError(
                "[Error] Model files are missing and were not downloaded. Exiting.");
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
            ConsoleOutput.WriteLine("[Audio] No input microphones found.", ConsoleColor.Red);
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
                DirectMlAdapterSelector.IsDirectMlRequested(config.Inference))
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

            InferenceConfigurationPrinter.Print(config.Inference, sessionFactories);

            ConsoleOutput.WriteLine(
                $"[Model] Loading ONNX models with " +
                $"{sessionFactories.Describe()}...",
                ConsoleColor.DarkGray);

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

            ConsoleOutput.WriteLine("[Model] ONNX models ready.", ConsoleColor.DarkGray);

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

            ConsoleOutput.WriteLine(
                "\n>>> Microphones active. Speak clearly. Press [ENTER] to exit pipeline <<<\n",
                ConsoleColor.Cyan);

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

            waveSource.StopRecording();
            waveSource.DataAvailable -= engine.ProcessAudioChunk;
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

    private static ILoggerFactory CreateLoggerFactory()
    {
        IConfiguration appSettings = new ConfigurationBuilder()
            .SetBasePath(AppContext.BaseDirectory)
            .AddJsonFile("appsettings.json", optional: true, reloadOnChange: false)
            .Build();

        return LoggerFactory.Create(builder =>
        {
            builder.AddConfiguration(appSettings.GetSection("Logging"));
            builder.AddConsole();
        });
    }

    private static StreamWriter? OpenTranscriptWriter(
        RunOptions runOptions,
        ILogger logger)
    {
        if (runOptions.TranscriptPath is null)
            return null;

        try
        {
            var writer =
                new StreamWriter(runOptions.TranscriptPath, append: true, Encoding.UTF8);
            logger.LogInformation(
                "[Config] Transcripts target file: {Path}",
                Path.GetFullPath(runOptions.TranscriptPath));
            return writer;
        }
        catch (Exception ex)
        {
            logger.LogWarning(
                "[Warning] File init failed: {Reason}. Screen output only.",
                ex.Message);
            return null;
        }
    }

}
