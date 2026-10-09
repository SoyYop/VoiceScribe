using Microsoft.UI.Xaml;
using VoiceScribe.Notepad.Views;

namespace VoiceScribe.Notepad;

public partial class App : Application
{
    private Window? _window;
    public App() => InitializeComponent();
    protected override void OnLaunched(LaunchActivatedEventArgs args)
    {
        _window = new MainWindow();
        _window.Activate();
    }
}
