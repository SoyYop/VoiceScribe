using System.Text;
using VoiceScribe.Notepad.Models;

namespace VoiceScribe.Notepad.Services;

public sealed class SessionStore
{
    public string FolderPath { get; }
    private readonly SemaphoreSlim _gate = new(1, 1);
    public SessionStore(string? folderPath = null)
    {
        FolderPath = folderPath ?? Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.MyDocuments), "VoiceScribe");
        Directory.CreateDirectory(FolderPath);
    }

    public async Task<IReadOnlyList<NoteSession>> LoadAsync()
    {
        var notes = new List<NoteSession>();
        foreach (string path in Directory.EnumerateFiles(FolderPath, "*.md").OrderByDescending(File.GetCreationTime))
        {
            string text = await File.ReadAllTextAsync(path);
            int newline = text.IndexOf('\n');
            bool hasTitle = text.StartsWith("# ", StringComparison.Ordinal);
            notes.Add(new NoteSession
            {
                FilePath = path,
                CreatedAt = File.GetCreationTime(path),
                Title = hasTitle ? (newline < 0 ? text[2..] : text[2..newline]).TrimEnd('\r') : Path.GetFileNameWithoutExtension(path),
                Content = hasTitle ? ReadBody(text, newline) : text
            });
        }
        return notes;
    }

    private static string ReadBody(string text, int newline)
    {
        if (newline < 0) return "";
        string body = text[(newline + 1)..];
        if (body.StartsWith("\r\n", StringComparison.Ordinal)) return body[2..];
        return body.StartsWith('\n') ? body[1..] : body;
    }

    public async Task<NoteSession> CreateAsync()
    {
        DateTime now = DateTime.Now;
        var note = new NoteSession
        {
            FilePath = Path.Combine(FolderPath, $"{now:yyyy-MM-dd_HH-mm-ss}_{Guid.NewGuid():N}.md"),
            CreatedAt = now,
            Title = $"Sesión {now:dd-MM-yyyy HH:mm}"
        };
        await SaveAsync(note);
        return note;
    }

    public async Task SaveAsync(NoteSession note)
    {
        // Snapshot on the caller's UI thread; serialized atomic replacement keeps a complete Markdown file.
        string title = note.Title.Replace('\r', ' ').Replace('\n', ' ').Trim();
        string markdown = $"# {title}\n\n{note.Content}";
        await _gate.WaitAsync();
        string temporary = note.FilePath + ".tmp";
        try
        {
            await File.WriteAllTextAsync(temporary, markdown, new UTF8Encoding(false));
            File.Move(temporary, note.FilePath, overwrite: true);
        }
        finally
        {
            if (File.Exists(temporary)) File.Delete(temporary);
            _gate.Release();
        }
    }
}
