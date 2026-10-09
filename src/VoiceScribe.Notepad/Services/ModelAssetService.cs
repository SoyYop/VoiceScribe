using VoiceScribe.Notepad.ModelAssets;

namespace VoiceScribe.Notepad.Services;

public sealed class ModelAssetService
{
    public static IReadOnlyList<string> MissingFiles(string folder) => NemotronModelFiles.CreateRequiredFileList()
        .Where(file => !File.Exists(Path.Combine(folder, file)) || new FileInfo(Path.Combine(folder, file)).Length == 0).ToArray();

    public async Task DownloadAsync(AppSettings settings, IProgress<string> progress, CancellationToken cancellationToken)
    {
        Directory.CreateDirectory(settings.ModelFolder);
        using var client = new HttpClient { Timeout = Timeout.InfiniteTimeSpan };
        foreach (string file in MissingFiles(settings.ModelFolder))
        {
            string path = Path.Combine(settings.ModelFolder, file);
            string temporary = path + ".download";
            try
            {
                using var response = await client.GetAsync($"{settings.Recognition.RepoUrl.TrimEnd('/')}/{file}", HttpCompletionOption.ResponseHeadersRead, cancellationToken);
                response.EnsureSuccessStatusCode();
                await using var input = await response.Content.ReadAsStreamAsync(cancellationToken);
                await using (var output = new FileStream(temporary, FileMode.Create, FileAccess.Write, FileShare.None, 131072, true))
                {
                    var buffer = new byte[131072];
                    long written = 0;
                    long lastReported = -1;
                    int count;
                    while ((count = await input.ReadAsync(buffer, cancellationToken)) > 0)
                    {
                        await output.WriteAsync(buffer.AsMemory(0, count), cancellationToken);
                        written += count;
                        long megabytes = written / 1048576;
                        if (megabytes != lastReported)
                        {
                            progress.Report($"{file} · {megabytes} MB");
                            lastReported = megabytes;
                        }
                    }
                    if (response.Content.Headers.ContentLength is long expected && written != expected)
                        throw new IOException($"Descarga incompleta: {file}");
                }
                File.Move(temporary, path, true);
            }
            finally { if (File.Exists(temporary)) File.Delete(temporary); }
        }
    }
}
