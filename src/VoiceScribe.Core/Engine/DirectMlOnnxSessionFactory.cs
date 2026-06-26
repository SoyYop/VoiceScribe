#if VOICESCRIBE_ONNXRUNTIME_DIRECTML
using Microsoft.Extensions.Logging;
using Microsoft.ML.OnnxRuntime;
using VoiceScribe.Core.Configuration;

namespace VoiceScribe.Core.Engine;

/// <summary>
/// Creates sessions backed by the ONNX Runtime DirectML execution provider.
/// </summary>
public sealed class DirectMlOnnxSessionFactory : IOnnxSessionFactory
{
    private readonly ILogger _logger;
    private readonly OnnxRuntimeOptions _options;

    public DirectMlOnnxSessionFactory(
        ILogger logger,
        OnnxRuntimeOptions options)
    {
        _logger = logger;
        _options = options;
    }

    public OnnxExecutionProvider ExecutionProvider =>
        OnnxExecutionProvider.DirectMl;

    public InferenceSession CreateSession(string modelPath)
    {
        try
        {
            return CreateDirectMlSession(modelPath);
        }
        catch (OnnxRuntimeException ex) when (_options.AllowCpuFallback)
        {
            _logger.LogWarning(
                ex,
                "DirectML could not initialize {ModelPath}. " +
                "Recreating this session with the CPU provider.",
                modelPath);

            return new CpuOnnxSessionFactory(_logger, _options)
                .CreateSession(modelPath);
        }
    }

    private InferenceSession CreateDirectMlSession(string modelPath)
    {
        using var sessionOptions = new SessionOptions
        {
            GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
            ExecutionMode = ExecutionMode.ORT_SEQUENTIAL,
            EnableMemoryPattern = false,
            EnableProfiling = _options.EnableProfiling
        };

        sessionOptions.AppendExecutionProvider_DML(_options.DeviceId);

        _logger.LogInformation(
            "Creating DirectML ONNX session for {ModelPath} on device {DeviceId}.",
            modelPath,
            _options.DeviceId);

        return new InferenceSession(modelPath, sessionOptions);
    }
}
#endif
