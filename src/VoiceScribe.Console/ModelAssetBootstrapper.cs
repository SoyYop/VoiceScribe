using Microsoft.Extensions.Logging;
using VoiceScribe.Core.Configuration;
using VoiceScribe.Core.Engine;
using VoiceScribe.Core.ModelAssets;

namespace VoiceScribe.Console;

internal static class ModelAssetBootstrapper
{
    internal static async Task<bool> EnsureDownloadedAsync(
        ILogger<NemotronEngine> logger,
        VoiceAppConfig config)
    {
        ModelDownloader downloader = new(config.RepoUrl, config.ModelDownloadsPath!);
        if (downloader.VerifyLocalWeights(config.ModelFiles))
            return true;

        logger.LogError("[Error] Model files are missing, asking to download.");

        ConsoleOutput.Write(
            "\n[Missing Assets] Nemotron model layers missing. Download? (y/n): ",
            ConsoleColor.Yellow);

        if (char.ToLower(System.Console.ReadKey().KeyChar) != 'y')
        {
            logger.LogInformation("[Config] User declined to download Model files, exiting.");
            return false;
        }

        System.Console.WriteLine();
        using var downloadCts = new CancellationTokenSource();
        ConsoleCancelEventHandler cancelDownload = (_, eventArgs) =>
        {
            eventArgs.Cancel = true;
            downloadCts.Cancel();
        };

        System.Console.CancelKeyPress += cancelDownload;
        try
        {
            using var httpClient = new HttpClient();
            await downloader.HandleModelDownload(
                httpClient,
                config.ModelFiles,
                downloadCts.Token);
            logger.LogInformation("[Config] Model files downloaded.");
            return true;
        }
        catch (OperationCanceledException)
        {
            logger.LogWarning("[Config] Model download cancelled.");
            return false;
        }
        finally
        {
            System.Console.CancelKeyPress -= cancelDownload;
        }
    }
}
