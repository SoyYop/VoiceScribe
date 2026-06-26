using Microsoft.Extensions.Logging;
using VoiceScribe.Core.Configuration;

namespace VoiceScribe.Core.Engine;

/// <summary>
/// Resolves the session factory available in the current runtime variant.
/// </summary>
public static class OnnxSessionFactoryResolver
{
    public static IOnnxSessionFactory Create(
        OnnxRuntimeOptions options,
        ILogger logger)
    {
        ArgumentNullException.ThrowIfNull(options);
        ArgumentNullException.ThrowIfNull(logger);

        return options.ExecutionProvider switch
        {
            OnnxExecutionProvider.Cpu =>
                new CpuOnnxSessionFactory(logger, options),
#if VOICESCRIBE_ONNXRUNTIME_DIRECTML
            OnnxExecutionProvider.DirectMl =>
                new DirectMlOnnxSessionFactory(logger, options),
#else
            OnnxExecutionProvider.DirectMl =>
                throw CreateUnavailableException(OnnxExecutionProvider.DirectMl),
#endif
            OnnxExecutionProvider.Cuda =>
                throw CreateUnavailableException(OnnxExecutionProvider.Cuda),
            _ => throw new ArgumentOutOfRangeException(
                nameof(options),
                options.ExecutionProvider,
                "Unknown ONNX execution provider.")
        };
    }

    private static NotSupportedException CreateUnavailableException(
        OnnxExecutionProvider provider) =>
        new(
            $"Execution provider '{provider}' is not available in the " +
            $"{OnnxRuntimeVariant.Name} " +
            "runtime variant. Use the matching VoiceScribe runtime build.");
}
