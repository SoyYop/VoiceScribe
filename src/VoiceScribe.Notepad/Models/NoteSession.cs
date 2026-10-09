using System.ComponentModel;

namespace VoiceScribe.Notepad.Models;

public sealed class NoteSession : INotifyPropertyChanged
{
    public required string FilePath { get; init; }
    public DateTime CreatedAt { get; init; }
    private string _title = "Nueva sesión";
    private string _content = "";
    public string Title
    {
        get => _title;
        set { if (_title == value) return; _title = value; PropertyChanged?.Invoke(this, new(nameof(Title))); }
    }
    public string Content
    {
        get => _content;
        set { if (_content == value) return; _content = value; PropertyChanged?.Invoke(this, new(nameof(Content))); }
    }
    public string DateLabel => CreatedAt.ToString("dd MMM yyyy · HH:mm");
    public event PropertyChangedEventHandler? PropertyChanged;
}
