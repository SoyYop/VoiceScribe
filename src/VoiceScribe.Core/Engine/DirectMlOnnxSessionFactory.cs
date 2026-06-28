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
                "DirectML could not initialize {ModelPath}. " +
                "Recreating this session with the CPU provider. Reason: {Reason}",
                modelPath,
                GetFirstErrorLine(ex));

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

        OnnxSessionOptionsConfigurator.ApplyLogging(sessionOptions, _options);

        sessionOptions.AppendExecutionProvider_DML(_options.DeviceId);

        _logger.LogInformation(
            "Creating DirectML ONNX session via {Runtime} runtime for {ModelPath} on device {DeviceId}.",
            OnnxRuntimeInfo.Name,
            modelPath,
            _options.DeviceId);

        return new InferenceSession(modelPath, sessionOptions);
    }

    private static string GetFirstErrorLine(Exception exception)
    {
        string[] lines = exception.Message.Split(
            ['\r', '\n'],
            StringSplitOptions.RemoveEmptyEntries |
            StringSplitOptions.TrimEntries);

        return lines.Length > 0
            ? lines[0]
            : exception.GetType().Name;
    }
}
