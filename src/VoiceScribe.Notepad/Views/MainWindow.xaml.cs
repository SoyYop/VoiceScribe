using System.Diagnostics;
using Microsoft.UI.Windowing;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using VoiceScribe.Notepad.Models;
using VoiceScribe.Notepad.Services;
using VoiceScribe.Notepad.ViewModels;
using Windows.Storage.Pickers;

namespace VoiceScribe.Notepad.Views;

public sealed partial class MainWindow : Window
{
    public NotebookViewModel ViewModel { get; } = new();
    private readonly SettingsStore _settingsStore = new();
    private readonly DictationService _dictation = new();
    private readonly DispatcherTimer _saveTimer = new() { Interval = TimeSpan.FromSeconds(1) };
    private AppSettings _settings = new();
    private bool _loading = true, _recording, _busy, _saving, _dirty, _closing, _allowClose;
    private long _revision;
    private CancellationTokenSource? _downloadCancellation;

    public MainWindow()
    {
        InitializeComponent();
        SystemBackdrop = new Microsoft.UI.Xaml.Media.MicaBackdrop();
        AppWindow.Resize(new Windows.Graphics.SizeInt32(1120, 780));
        AppWindow.Closing += Window_Closing;
        Root.Loaded += InitializeAsync;
        _dictation.TextRecognized += text => DispatcherQueue.TryEnqueue(() => AppendDictation(text));
        _dictation.Error += error => DispatcherQueue.TryEnqueue(async () =>
        {
            StateText.Text = $"Error de dictado: {error}";
            if (_recording && !_busy) await StopDictationAsync();
        });
        _saveTimer.Tick += async (_, _) =>
        {
            if (_dirty && !_saving && !_closing) await SaveCurrentAsync();
        };
    }

    private async void InitializeAsync(object sender, RoutedEventArgs args)
    {
        Root.Loaded -= InitializeAsync;
        SetBusy(true, "Abriendo cuaderno…");
        try
        {
            try { _settings = await _settingsStore.LoadAsync(); }
            catch (Exception ex) { await ShowErrorAsync(new IOException("No se pudieron cargar los ajustes. Se usarán los valores iniciales. " + ex.Message)); }
            await ViewModel.InitializeAsync();
            ShowSelected();
            _saveTimer.Start();
        }
        catch (Exception ex) { await ShowErrorAsync(ex); }
        finally { _loading = false; SetBusy(false, "Listo para escribir"); }
    }

    private void ShowSelected()
    {
        _loading = true;
        SessionList.SelectedItem = ViewModel.Selected;
        TitleEditor.Text = ViewModel.Selected?.Title ?? "";
        Editor.Text = ViewModel.Selected?.Content ?? "";
        _dirty = false;
        SaveState.Text = ViewModel.Selected is null ? "" : $"Guardado · {ViewModel.Selected.FilePath}";
        _loading = false;
    }

    private async void SessionList_SelectionChanged(object sender, SelectionChangedEventArgs args)
    {
        if (_loading || _busy || _recording || SessionList.SelectedItem is not NoteSession selected || selected == ViewModel.Selected) return;
        SetBusy(true, "Guardando sesión…");
        try
        {
            if (!await SaveCurrentAsync()) { SessionList.SelectedItem = ViewModel.Selected; return; }
            ViewModel.Selected = selected;
            ShowSelected();
        }
        finally { SetBusy(false, "Listo para escribir"); }
    }

    private void TitleEditor_TextChanged(object sender, TextChangedEventArgs args)
    {
        if (_loading || ViewModel.Selected is null) return;
        ViewModel.Selected.Title = TitleEditor.Text;
        MarkDirty();
    }
    private void Editor_TextChanged(object sender, TextChangedEventArgs args)
    {
        if (_loading || ViewModel.Selected is null) return;
        ViewModel.Selected.Content = Editor.Text;
        MarkDirty();
    }
    private void MarkDirty() { _dirty = true; _revision++; SaveState.Text = "Cambios pendientes…"; }

    private async Task<bool> SaveCurrentAsync()
    {
        if (ViewModel.Selected is null) return true;
        var note = ViewModel.Selected;
        long revision = _revision;
        _saving = true;
        try
        {
            await ViewModel.Store.SaveAsync(note);
            if (revision == _revision && ViewModel.Selected == note)
            {
                _dirty = false;
                SaveState.Text = $"Guardado · {note.FilePath}";
            }
            return true;
        }
        catch (Exception ex) { SaveState.Text = $"No se pudo guardar: {ex.Message}"; return false; }
        finally { _saving = false; }
    }

    private async void NewSession_Click(object sender, RoutedEventArgs args)
    {
        if (_busy || _recording) return;
        SetBusy(true, "Creando sesión…");
        try
        {
            if (!await SaveCurrentAsync()) return;
            await ViewModel.NewAsync();
            ShowSelected();
            Editor.Focus(FocusState.Programmatic);
        }
        catch (Exception ex) { await ShowErrorAsync(ex); }
        finally { SetBusy(false, "Listo para escribir"); }
    }
    private async void Save_Click(object sender, RoutedEventArgs args) => await SaveCurrentAsync();
    private async void OpenFolder_Click(object sender, RoutedEventArgs args)
    {
        try { Process.Start(new ProcessStartInfo("explorer.exe") { ArgumentList = { ViewModel.Store.FolderPath }, UseShellExecute = true }); }
        catch (Exception ex) { await ShowErrorAsync(ex); }
    }

    private void AppendDictation(string text)
    {
        if (ViewModel.Selected is null) return;
        Editor.Text += text;
        Editor.SelectionStart = Editor.Text.Length;
    }
    private async void Dictate_Click(object sender, RoutedEventArgs args)
    {
        if (_downloadCancellation is not null) { _downloadCancellation.Cancel(); return; }
        if (_busy || ViewModel.Selected is null) return;
        if (_recording) { await StopDictationAsync(); return; }
        SetBusy(true, "Cargando modelo…");
        try
        {
            await _dictation.StartAsync(_settings);
            _recording = true;
            DictateLabel.Text = "Detener dictado";
            Editor.Focus(FocusState.Programmatic);
        }
        catch (Exception ex) { await ShowErrorAsync(ex); }
        finally { SetBusy(false, _recording ? "Escuchando · el dictado se añade al final" : "Listo para escribir"); }
    }

    private async Task StopDictationAsync()
    {
        SetBusy(true, "Terminando las últimas palabras…");
        try
        {
            await _dictation.StopAsync();
            // Process all token callbacks queued by the worker before saving or switching notes.
            var drained = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
            if (DispatcherQueue.TryEnqueue(() => drained.SetResult())) await drained.Task;
            await SaveCurrentAsync();
        }
        catch (Exception ex) { await ShowErrorAsync(ex); }
        finally { _recording = false; DictateLabel.Text = "Iniciar dictado"; SetBusy(false, "Listo para escribir"); }
    }

    private void SetBusy(bool busy, string state)
    {
        _busy = busy;
        BusyRing.IsActive = busy;
        StateText.Text = state;
        NewButton.IsEnabled = NewMenu.IsEnabled = SettingsMenu.IsEnabled = SessionList.IsEnabled = !busy && !_recording && !_closing;
        DictateButton.IsEnabled = !busy && !_closing;
        TitleEditor.IsEnabled = Editor.IsEnabled = !busy;
    }

    private async void Settings_Click(object sender, RoutedEventArgs args)
    {
        if (_busy || _recording) return;
        var modelPath = new TextBox { Header = "Carpeta del modelo Nemotron ONNX", Text = _settings.ModelFolder, MinWidth = 400 };
        var choose = new Button { Content = "Elegir carpeta existente…" };
        var microphones = new ComboBox { Header = "Micrófono", ItemsSource = DictationService.Microphones(), HorizontalAlignment = HorizontalAlignment.Stretch };
        if (microphones.Items.Count > 0) microphones.SelectedIndex = Math.Clamp(_settings.MicrophoneIndex, 0, microphones.Items.Count - 1);
        var provider = new ComboBox { Header = "Procesamiento", ItemsSource = new[] { "DirectML (GPU, con respaldo CPU)", "CPU" }, SelectedIndex = _settings.Recognition.Inference.ExecutionProvider == Configuration.OnnxExecutionProvider.Cpu ? 1 : 0 };
        var language = new NumberBox { Header = "ID de idioma del modelo", Value = _settings.Recognition.Nemotron.LanguageId, Minimum = 0, SpinButtonPlacementMode = NumberBoxSpinButtonPlacementMode.Compact };
        var download = new CheckBox { Content = "Descargar los archivos que faltan al guardar" };
        var panel = new StackPanel { Spacing = 16 };
        panel.Children.Add(modelPath); panel.Children.Add(choose); panel.Children.Add(microphones); panel.Children.Add(provider); panel.Children.Add(language); panel.Children.Add(download);
        panel.Children.Add(new TextBlock { Text = "Puedes reutilizar la carpeta de modelos de la consola. La descarga solo es necesaria la primera vez; el dictado se procesa en tu equipo.", TextWrapping = TextWrapping.Wrap });
        var dialog = new ContentDialog { XamlRoot = Root.XamlRoot, Title = "Configuración", Content = panel, PrimaryButtonText = "Guardar", CloseButtonText = "Cancelar", DefaultButton = ContentDialogButton.Primary };
        choose.Click += async (_, _) =>
        {
            try
            {
                var picker = new FolderPicker(); picker.FileTypeFilter.Add("*");
                WinRT.Interop.InitializeWithWindow.Initialize(picker, WinRT.Interop.WindowNative.GetWindowHandle(this));
                var folder = await picker.PickSingleFolderAsync();
                if (folder is not null) modelPath.Text = folder.Path;
            }
            catch (Exception ex) { modelPath.Header = $"No se pudo abrir el selector: {ex.Message}"; }
        };
        dialog.PrimaryButtonClick += (_, eventArgs) =>
        {
            if (!Path.IsPathFullyQualified(modelPath.Text.Trim()) || !double.IsFinite(language.Value) || language.Value < 0 || language.Value > int.MaxValue || language.Value != Math.Truncate(language.Value))
            { eventArgs.Cancel = true; modelPath.Header = "Introduce una ruta absoluta y un ID de idioma entero válido"; }
        };
        if (await dialog.ShowAsync() != ContentDialogResult.Primary) return;
        _settings.ModelFolder = modelPath.Text.Trim();
        _settings.MicrophoneIndex = Math.Max(0, microphones.SelectedIndex);
        _settings.Recognition.Nemotron.LanguageId = (int)language.Value;
        _settings.Recognition.Inference.ExecutionProvider = provider.SelectedIndex == 1 ? Configuration.OnnxExecutionProvider.Cpu : Configuration.OnnxExecutionProvider.DirectMl;
        try
        {
            await _settingsStore.SaveAsync(_settings);
            if (download.IsChecked == true) await DownloadModelAsync();
        }
        catch (Exception ex) { await ShowErrorAsync(ex); }
    }

    private async Task DownloadModelAsync()
    {
        _downloadCancellation = new();
        SetBusy(true, "Descargando modelo…");
        DictateButton.IsEnabled = true;
        DictateLabel.Text = "Cancelar descarga";
        try
        {
            await new ModelAssetService().DownloadAsync(_settings, new Progress<string>(text => StateText.Text = text), _downloadCancellation.Token);
        }
        catch (OperationCanceledException) { StateText.Text = "Descarga cancelada"; }
        finally
        {
            _downloadCancellation.Dispose(); _downloadCancellation = null;
            DictateLabel.Text = "Iniciar dictado";
            SetBusy(false, "Listo para escribir");
        }
    }

    private async void Window_Closing(AppWindow sender, AppWindowClosingEventArgs args)
    {
        if (_allowClose) return;
        args.Cancel = true;
        if (_closing) return;
        if (_busy) { StateText.Text = "Espera a que termine la operación antes de cerrar."; _downloadCancellation?.Cancel(); return; }
        _closing = true;
        _saveTimer.Stop();
        try
        {
            if (_recording) await StopDictationAsync();
            if (!await SaveCurrentAsync()) { _closing = false; _saveTimer.Start(); SetBusy(false, "No se pudo guardar; revisa la carpeta de notas."); return; }
            _allowClose = true;
            Close();
        }
        catch (Exception ex) { _closing = false; _saveTimer.Start(); SetBusy(false, "No se pudo cerrar"); await ShowErrorAsync(ex); }
    }

    private Task<ContentDialogResult> ShowErrorAsync(Exception error) => new ContentDialog
    {
        XamlRoot = Root.XamlRoot, Title = "VoiceScribe", Content = error.Message, CloseButtonText = "Aceptar"
    }.ShowAsync().AsTask();
}
