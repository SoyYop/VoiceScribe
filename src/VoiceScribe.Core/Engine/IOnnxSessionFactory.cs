using Microsoft.ML.OnnxRuntime;
using VoiceScribe.Core.Configuration;

namespace VoiceScribe.Core.Engine;

/// <summary>
/// Creates consistently configured ONNX Runtime sessions.
/// </summary>
public interface IOnnxSessionFactory
{
    OnnxExecutionProvider ExecutionProvider { get; }

    InferenceSession CreateSession(string modelPath);
}
