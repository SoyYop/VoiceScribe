using VoiceScribe.Notepad.Configuration;

namespace VoiceScribe.Notepad.Engine;

/// <summary>
/// Describes the WindowsML runtime used by VoiceScribe.
/// </summary>
public static class OnnxRuntimeInfo
{
    public const string Name = "WindowsML";

    public static bool Supports(OnnxExecutionProvider provider) =>
        provider switch
        {
            OnnxExecutionProvider.Cpu => true,
            OnnxExecutionProvider.DirectMl => true,
            _ => false
        };
}
