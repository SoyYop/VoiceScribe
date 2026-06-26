using VoiceScribe.Core.Configuration;

namespace VoiceScribe.Core.Engine;

/// <summary>
/// Describes the execution providers compiled into this VoiceScribe build.
/// </summary>
public static class OnnxRuntimeVariant
{
#if VOICESCRIBE_ONNXRUNTIME_DIRECTML
    public const string Name = "DirectML";
#else
    public const string Name = "CPU";
#endif

    public static bool Supports(OnnxExecutionProvider provider) =>
        provider switch
        {
            OnnxExecutionProvider.Cpu => true,
#if VOICESCRIBE_ONNXRUNTIME_DIRECTML
            OnnxExecutionProvider.DirectMl => true,
#else
            OnnxExecutionProvider.DirectMl => false,
#endif
            OnnxExecutionProvider.Cuda => false,
            _ => false
        };
}
