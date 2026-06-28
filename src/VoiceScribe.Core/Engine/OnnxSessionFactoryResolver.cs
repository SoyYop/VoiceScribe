using Microsoft.Extensions.Logging;
using VoiceScribe.Core.Configuration;

namespace VoiceScribe.Core.Engine;

/// <summary>
/// Resolves session factories for execution providers available in WindowsML.
/// </summary>
public static class OnnxSessionFactoryResolver
{
    public static IOnnxSessionFactory Create(
        OnnxRuntimeOptions options,
        ILogger logger)
    {
        ArgumentNullException.ThrowIfNull(options);
        ArgumentNullException.ThrowIfNull(logger);

        return Create(options.ExecutionProvider, options, logger);
    }

    public static NemotronOnnxSessionFactories CreateForNemotron(
        OnnxRuntimeOptions options,
        ILogger logger)
    {
        ArgumentNullException.ThrowIfNull(options);
        ArgumentNullException.ThrowIfNull(logger);

        return new NemotronOnnxSessionFactories(
            Create(options.GetEncoderProvider(), options, logger),
            Create(options.GetDecoderProvider(), options, logger),
            Create(options.GetJoinerProvider(), options, logger));
    }

    public static IOnnxSessionFactory Create(
        OnnxExecutionProvider provider,
        OnnxRuntimeOptions options,
        ILogger logger)
    {
        ArgumentNullException.ThrowIfNull(options);
        ArgumentNullException.ThrowIfNull(logger);

        return provider switch
        {
            OnnxExecutionProvider.Cpu =>
                new CpuOnnxSessionFactory(logger, options),
            OnnxExecutionProvider.DirectMl =>
                new DirectMlOnnxSessionFactory(logger, options),
            _ => throw new ArgumentOutOfRangeException(
                nameof(provider),
                provider,
                "Unknown ONNX execution provider.")
        };
    }
}
