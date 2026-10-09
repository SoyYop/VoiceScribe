using Microsoft.Extensions.Logging.Abstractions;
using NAudio.Wave;
using VoiceScribe.Notepad.Configuration;
using VoiceScribe.Notepad.Engine;
using VoiceScribe.Notepad.ModelAssets;
using Xunit;

namespace VoiceScribe.Notepad.Tests;

public sealed class LocalModelFactAttribute : FactAttribute
{
    public LocalModelFactAttribute()
    {
        if (string.IsNullOrEmpty(Environment.GetEnvironmentVariable("VOICESCRIBE_TEST_MODEL")))
            Skip = "Set VOICESCRIBE_TEST_MODEL to a complete local ONNX model folder.";
    }
}

public sealed class EngineSmokeTests
{
    [LocalModelFact]
    public async Task RealModel_DrainsPartialAudioAndStopsIdempotently()
    {
        string folder = Environment.GetEnvironmentVariable("VOICESCRIBE_TEST_MODEL")!;
        var model = NemotronModelDefinition.Load(Path.Combine(folder, NemotronModelFiles.GenAiConfig));
        var options = new OnnxRuntimeOptions { ExecutionProvider = OnnxExecutionProvider.Cpu };
        var audio = new AudioCaptureOptions { FinalSilencePaddingChunks = 1 };
        var failures = new List<string>();
        await using var engine = new NemotronEngine(NullLogger<NemotronEngine>.Instance, folder, audio,
            new NemotronModelOptions(), model, OnnxSessionFactoryResolver.CreateForNemotron(options, NullLogger.Instance), null);
        engine.ProcessingError += failures.Add;
        // A nonempty partial PCM chunk must be padded and processed, even when it is silent.
        engine.ProcessAudioChunk(null, new WaveInEventArgs(new byte[3200], 3200));
        await engine.StopAsync();
        await engine.StopAsync();
        engine.ProcessAudioChunk(null, new WaveInEventArgs(new byte[3200], 3200));
        Assert.Empty(failures);
    }
}
