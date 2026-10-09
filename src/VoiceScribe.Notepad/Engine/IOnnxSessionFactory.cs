using Microsoft.ML.OnnxRuntime;
using VoiceScribe.Notepad.Configuration;

namespace VoiceScribe.Notepad.Engine;

/// <summary>
/// Creates consistently configured ONNX Runtime sessions.
/// </summary>
public interface IOnnxSessionFactory
{
    OnnxExecutionProvider ExecutionProvider { get; }

    InferenceSession CreateSession(string modelPath);
}
