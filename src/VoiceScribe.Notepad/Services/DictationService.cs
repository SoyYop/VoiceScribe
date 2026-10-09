using Microsoft.Extensions.Logging.Abstractions;
using NAudio.Wave;
using VoiceScribe.Notepad.Configuration;
using VoiceScribe.Notepad.Engine;
using VoiceScribe.Notepad.ModelAssets;

namespace VoiceScribe.Notepad.Services;

public sealed class DictationService
{
    private NemotronEngine? _engine;
    private WaveInEvent? _capture;
    private TaskCompletionSource? _recordingStopped;
    private bool _captureStarted;
    public event Action<string>? TextRecognized;
    public event Action<string>? Error;
    public static IReadOnlyList<string> Microphones() => Enumerable.Range(0, WaveIn.DeviceCount)
        .Select(index => WaveIn.GetCapabilities(index).ProductName).ToArray();

    public async Task StartAsync(AppSettings settings)
    {
        if (_engine is not null) throw new InvalidOperationException("El dictado ya está activo.");
        var missing = ModelAssetService.MissingFiles(settings.ModelFolder);
        if (missing.Count > 0) throw new InvalidOperationException("Selecciona la carpeta del modelo o descárgalo desde Configuración.");
        if (settings.MicrophoneIndex < 0 || settings.MicrophoneIndex >= WaveIn.DeviceCount)
            throw new InvalidOperationException("Selecciona un micrófono disponible desde Configuración.");
        var config = settings.Recognition;
        var model = NemotronModelDefinition.Load(Path.Combine(settings.ModelFolder, NemotronModelFiles.GenAiConfig));
        var errors = VoiceAppConfigValidator.Validate(config, model);
        if (errors.Count > 0) throw new InvalidOperationException(string.Join(Environment.NewLine, errors));
        try
        {
            _engine = await Task.Run(() => new NemotronEngine(NullLogger<NemotronEngine>.Instance,
                settings.ModelFolder, config.Audio, config.Nemotron, model,
                OnnxSessionFactoryResolver.CreateForNemotron(config.Inference, NullLogger.Instance), null));
            _engine.TextRecognized += ForwardText;
            _engine.ProcessingError += ForwardError;
            _recordingStopped = new(TaskCreationOptions.RunContinuationsAsynchronously);
            _capture = new WaveInEvent
            {
                DeviceNumber = settings.MicrophoneIndex,
                WaveFormat = new WaveFormat(config.Audio.SampleRate, config.Audio.BitsPerSample, config.Audio.Channels),
                BufferMilliseconds = config.Audio.BufferMilliseconds
            };
            _capture.DataAvailable += _engine.ProcessAudioChunk;
            _capture.RecordingStopped += OnRecordingStopped;
            _capture.StartRecording();
            _captureStarted = true;
        }
        catch { await StopAsync(); throw; }
    }

    private void ForwardText(string text) => TextRecognized?.Invoke(text);
    private void ForwardError(string error) => Error?.Invoke(error);
    private void OnRecordingStopped(object? sender, StoppedEventArgs args)
    {
        _recordingStopped?.TrySetResult();
        if (args.Exception is not null) Error?.Invoke(args.Exception.Message);
    }

    public async Task StopAsync()
    {
        if (_capture is not null)
        {
            _capture.StopRecording();
            if (_captureStarted && _recordingStopped is not null) await _recordingStopped.Task;
            if (_engine is not null) _capture.DataAvailable -= _engine.ProcessAudioChunk;
            _capture.RecordingStopped -= OnRecordingStopped;
            _capture.Dispose();
            _capture = null;
            _captureStarted = false;
        }
        if (_engine is not null)
        {
            await _engine.DisposeAsync();
            _engine.TextRecognized -= ForwardText;
            _engine.ProcessingError -= ForwardError;
            _engine = null;
        }
    }
}
