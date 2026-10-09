using System.Text.Json;
using VoiceScribe.Notepad.Configuration;

namespace VoiceScribe.Notepad.Services;

public sealed class AppSettings
{
    public string ModelFolder { get; set; } = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "VoiceScribe", "Models", "Nemotron");
    public int MicrophoneIndex { get; set; }
    public VoiceAppConfig Recognition { get; set; } = new()
    {
        RepoUrl = "https://huggingface.co/onnx-community/nemotron-3.5-asr-streaming-0.6b-onnx-int4/resolve/main",
        Audio = new() { TrailingSilenceChunks = 4 },
        Inference = new() { ExecutionProvider = OnnxExecutionProvider.DirectMl, AllowCpuFallback = true }
    };
}

public sealed class SettingsStore
{
    public string FolderPath { get; } = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "VoiceScribe");
    public async Task<AppSettings> LoadAsync()
    {
        string path = Path.Combine(FolderPath, "settings.json");
        if (!File.Exists(path)) return new();
        using var stream = File.OpenRead(path);
        return await JsonSerializer.DeserializeAsync<AppSettings>(stream) ?? new();
    }
    public async Task SaveAsync(AppSettings settings)
    {
        Directory.CreateDirectory(FolderPath);
        string path = Path.Combine(FolderPath, "settings.json");
        await File.WriteAllTextAsync(path + ".tmp", JsonSerializer.Serialize(settings, new JsonSerializerOptions { WriteIndented = true }));
        File.Move(path + ".tmp", path, true);
    }
}
