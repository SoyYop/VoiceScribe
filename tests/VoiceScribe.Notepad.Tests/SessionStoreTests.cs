using VoiceScribe.Notepad.Services;
using Xunit;

namespace VoiceScribe.Notepad.Tests;

public sealed class SessionStoreTests : IDisposable
{
    private readonly string _folder = Path.Combine(Path.GetTempPath(), "VoiceScribe.Tests", Guid.NewGuid().ToString("N"));

    [Fact]
    public async Task Session_RoundTripsUnicodeAndMarkdown()
    {
        var store = new SessionStore(_folder);
        var note = await store.CreateAsync();
        note.Title = "Reunión: diseño / ASR";
        note.Content = "\n\nUna idea con ñ y acentos.\n\n## Próximos pasos\n- Probar 🎙️\n";
        await store.SaveAsync(note);
        var reloaded = Assert.Single(await new SessionStore(_folder).LoadAsync());
        Assert.Equal(note.Title, reloaded.Title);
        Assert.Equal(note.Content, reloaded.Content);
        Assert.Equal(note.FilePath, reloaded.FilePath);
        Assert.Empty(Directory.EnumerateFiles(_folder, "*.tmp"));
    }

    [Fact]
    public async Task SameSecondSessions_HaveDistinctFilesAndRenamingPreservesPath()
    {
        var store = new SessionStore(_folder);
        var first = await store.CreateAsync();
        var second = await store.CreateAsync();
        Assert.NotEqual(first.FilePath, second.FilePath);
        first.Title = "Título nuevo\nOtra línea";
        await store.SaveAsync(first);
        Assert.Equal(2, (await store.LoadAsync()).Count);
        Assert.StartsWith("# Título nuevo Otra línea\n", await File.ReadAllTextAsync(first.FilePath));
    }

    [Fact]
    public async Task LoadsPlainMarkdownWithoutLosingItsContent()
    {
        var store = new SessionStore(_folder);
        const string text = "Texto externo\n\n- Uno\n- Dos\n";
        await File.WriteAllTextAsync(Path.Combine(_folder, "externa.md"), text);
        var note = Assert.Single(await store.LoadAsync());
        Assert.Equal("externa", note.Title);
        Assert.Equal(text, note.Content);
    }

    [Fact]
    public async Task ConcurrentSaves_LeaveLastCompleteSnapshot()
    {
        var store = new SessionStore(_folder);
        var note = await store.CreateAsync();
        var pending = new List<Task>();
        for (int i = 0; i < 20; i++)
        {
            note.Content = $"Texto {i}\n" + new string('á', 10000);
            pending.Add(store.SaveAsync(note));
        }
        await Task.WhenAll(pending);
        var reloaded = Assert.Single(await store.LoadAsync());
        Assert.Equal(note.Content, reloaded.Content);
    }

    public void Dispose()
    {
        if (Directory.Exists(_folder)) Directory.Delete(_folder, recursive: true);
    }
}
