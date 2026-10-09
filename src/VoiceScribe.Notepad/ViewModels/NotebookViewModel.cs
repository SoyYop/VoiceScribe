using System.Collections.ObjectModel;
using System.ComponentModel;
using VoiceScribe.Notepad.Models;
using VoiceScribe.Notepad.Services;

namespace VoiceScribe.Notepad.ViewModels;

public sealed class NotebookViewModel : INotifyPropertyChanged
{
    public SessionStore Store { get; } = new();
    public ObservableCollection<NoteSession> Sessions { get; } = [];
    private NoteSession? _selected;
    public NoteSession? Selected
    {
        get => _selected;
        set { if (_selected == value) return; _selected = value; PropertyChanged?.Invoke(this, new(nameof(Selected))); }
    }
    public async Task InitializeAsync()
    {
        foreach (var session in await Store.LoadAsync()) Sessions.Add(session);
        Selected = Sessions.FirstOrDefault();
        if (Selected is null) await NewAsync();
    }
    public async Task NewAsync()
    {
        if (Selected is not null) await Store.SaveAsync(Selected);
        var note = await Store.CreateAsync();
        Sessions.Insert(0, note);
        Selected = note;
    }
    public event PropertyChangedEventHandler? PropertyChanged;
}
