using VoiceScribe.Core.Configuration;
using VoiceScribe.Core.Engine;
using VoiceScribe.Core.ModelAssets;

namespace VoiceScribe.Core.Tests;

public sealed class VoiceAppConfigValidatorTests
{
    [Fact]
    public void Validate_AcceptsCompatibleConfiguration()
    {
        IReadOnlyList<string> errors =
            VoiceAppConfigValidator.Validate(CreateConfig(), CreateModel());

        Assert.Empty(errors);
    }

    [Fact]
    public void Validate_ReturnsAllRelevantCompatibilityErrors()
    {
        VoiceAppConfig config = CreateConfig();
        config.Audio.SampleRate = 8000;
        config.Audio.BufferMilliseconds = 100;
        config.Audio.QueueCapacity = 0;
        config.Nemotron.BlankId = 20000;
        config.Nemotron.MaxSymbolsPerStep = 0;
        config.Inference.DeviceId = -1;
        config.Inference.GpuMemoryLimitMiB = 0;

        IReadOnlyList<string> errors =
            VoiceAppConfigValidator.Validate(config, CreateModel());

        Assert.Contains(errors, error => error.Contains("QueueCapacity"));
        Assert.Contains(errors, error => error.Contains("SampleRate is 8000"));
        Assert.Contains(errors, error => error.Contains("samples per buffer"));
        Assert.Contains(errors, error => error.Contains("MaxSymbolsPerStep"));
        Assert.Contains(errors, error => error.Contains("BlankId"));
        Assert.Contains(errors, error => error.Contains("DeviceId"));
        Assert.Contains(errors, error => error.Contains("GpuMemoryLimitMiB"));
    }

    [Fact]
    public void Validate_HandlesDirectMlAccordingToRuntimeVariant()
    {
        VoiceAppConfig config = CreateConfig();
        config.Inference.ExecutionProvider = OnnxExecutionProvider.DirectMl;

        IReadOnlyList<string> errors =
            VoiceAppConfigValidator.Validate(config, CreateModel());

        if (OnnxRuntimeVariant.Supports(OnnxExecutionProvider.DirectMl))
            Assert.DoesNotContain(errors, error => error.Contains("runtime variant"));
        else
            Assert.Contains(errors, error => error.Contains("runtime variant"));
    }

    [Fact]
    public void Validate_HandlesSubmodelDirectMlAccordingToRuntimeVariant()
    {
        VoiceAppConfig config = CreateConfig();
        config.Inference.ExecutionProvider = OnnxExecutionProvider.Cpu;
        config.Inference.EncoderProvider = OnnxExecutionProvider.Cpu;
        config.Inference.DecoderProvider = OnnxExecutionProvider.Cpu;
        config.Inference.JoinerProvider = OnnxExecutionProvider.DirectMl;

        IReadOnlyList<string> errors =
            VoiceAppConfigValidator.Validate(config, CreateModel());

        if (OnnxRuntimeVariant.Supports(OnnxExecutionProvider.DirectMl))
        {
            Assert.DoesNotContain(
                errors,
                error => error.Contains("JoinerProvider") &&
                         error.Contains("runtime variant"));
        }
        else
        {
            Assert.Contains(
                errors,
                error => error.Contains("JoinerProvider") &&
                         error.Contains("runtime variant"));
        }
    }

    [Fact]
    public void Validate_RejectsCudaUntilCudaVariantExists()
    {
        VoiceAppConfig config = CreateConfig();
        config.Inference.ExecutionProvider = OnnxExecutionProvider.Cuda;

        IReadOnlyList<string> errors =
            VoiceAppConfigValidator.Validate(config, CreateModel());

        Assert.Contains(errors, error => error.Contains("runtime variant"));
    }

    [Fact]
    public void Validate_RejectsUnknownProviderValue()
    {
        VoiceAppConfig config = CreateConfig();
        config.Inference.ExecutionProvider = (OnnxExecutionProvider)99;

        IReadOnlyList<string> errors =
            VoiceAppConfigValidator.Validate(config, CreateModel());

        Assert.Contains(
            errors,
            error => error.Contains("ExecutionProvider is invalid"));
    }

    [Fact]
    public void Validate_RejectsUnknownSubmodelProviderValue()
    {
        VoiceAppConfig config = CreateConfig();
        config.Inference.JoinerProvider = (OnnxExecutionProvider)99;

        IReadOnlyList<string> errors =
            VoiceAppConfigValidator.Validate(config, CreateModel());

        Assert.Contains(
            errors,
            error => error.Contains("JoinerProvider is invalid"));
    }

    private static VoiceAppConfig CreateConfig() => new()
    {
        Audio = new AudioCaptureOptions
        {
            SampleRate = 16000,
            BitsPerSample = 16,
            Channels = 1,
            BufferMilliseconds = 560,
            SilenceThreshold = 0.003f,
            QueueCapacity = 8
        },
        Nemotron = new NemotronModelOptions(),
        Inference = new OnnxRuntimeOptions()
    };

    private static NemotronModelDefinition CreateModel() => new()
    {
        VocabularySize = 13088,
        BlankId = 13087,
        MaxSymbolsPerStep = 10,
        SampleRate = 16000,
        ChunkSamples = 8960,
        PreEncodeCacheSize = 9,
        EncoderHiddenSize = 1024,
        DecoderHiddenSize = 640,
        DecoderLayerCount = 2,
        Encoder = new EncoderContract
        {
            FileName = "encoder.onnx",
            AudioFeaturesInput = "audio_signal",
            InputLengthsInput = "length",
            CacheLastChannelInput = "cache_last_channel",
            CacheLastTimeInput = "cache_last_time",
            CacheLastChannelLengthInput = "cache_last_channel_len",
            LanguageIdInput = "lang_id",
            EncoderOutputsOutput = "outputs",
            CacheLastChannelOutput = "cache_last_channel_next",
            CacheLastTimeOutput = "cache_last_time_next",
            CacheLastChannelLengthOutput = "cache_last_channel_len_next"
        },
        Decoder = new DecoderContract
        {
            FileName = "decoder.onnx",
            TargetsInput = "targets",
            HiddenStateInput = "h_in",
            CellStateInput = "c_in",
            DecoderOutput = "decoder_output",
            HiddenStateOutput = "h_out",
            CellStateOutput = "c_out"
        },
        Joiner = new JoinerContract
        {
            FileName = "joint.onnx",
            EncoderOutputInput = "encoder_output",
            DecoderOutputInput = "decoder_output",
            LogitsOutput = "joint_output"
        }
    };
}
