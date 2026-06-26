using Microsoft.Extensions.Logging;
using Microsoft.ML.OnnxRuntime;
using VoiceScribe.Core.Configuration;

namespace VoiceScribe.Core.Engine;

/// <summary>
/// Creates sessions backed by the standard ONNX Runtime CPU package.
/// </summary>
public sealed class CpuOnnxSessionFactory : IOnnxSessionFactory
{
    private readonly ILogger _logger;
    private readonly OnnxRuntimeOptions _options;

    public CpuOnnxSessionFactory(
        ILogger logger,
        OnnxRuntimeOptions options)
    {
        _logger = logger;
        _options = options;
    }

    public OnnxExecutionProvider ExecutionProvider =>
        OnnxExecutionProvider.Cpu;

    public InferenceSession CreateSession(string modelPath)
    {
        using var sessionOptions = new SessionOptions
        {
            GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
            ExecutionMode = ExecutionMode.ORT_SEQUENTIAL,
            EnableProfiling = _options.EnableProfiling,
            // Temporal
            LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_VERBOSE,
            LogVerbosityLevel = 5
            // 
        };

        _logger.LogInformation(
            "Creating CPU ONNX session for {ModelPath}.",
            modelPath);

        return new InferenceSession(modelPath, sessionOptions);
    }
}
