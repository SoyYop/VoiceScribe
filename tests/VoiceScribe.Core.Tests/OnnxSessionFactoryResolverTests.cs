using Microsoft.Extensions.Logging.Abstractions;
using VoiceScribe.Core.Configuration;
using VoiceScribe.Core.Engine;

namespace VoiceScribe.Core.Tests;

public sealed class OnnxSessionFactoryResolverTests
{
    [Fact]
    public void Create_ReturnsCpuFactoryForCpuProvider()
    {
        var options = new OnnxRuntimeOptions
        {
            ExecutionProvider = OnnxExecutionProvider.Cpu
        };

        IOnnxSessionFactory factory =
            OnnxSessionFactoryResolver.Create(
                options,
                NullLogger.Instance);

        Assert.IsType<CpuOnnxSessionFactory>(factory);
        Assert.Equal(OnnxExecutionProvider.Cpu, factory.ExecutionProvider);
    }

    [Fact]
    public void Create_ReturnsDirectMlFactoryForDirectMlProvider()
    {
        var options = new OnnxRuntimeOptions
        {
            ExecutionProvider = OnnxExecutionProvider.DirectMl
        };

        IOnnxSessionFactory factory =
            OnnxSessionFactoryResolver.Create(
                options,
                NullLogger.Instance);

        Assert.Equal(
            OnnxExecutionProvider.DirectMl,
            factory.ExecutionProvider);
        Assert.IsType<DirectMlOnnxSessionFactory>(factory);
    }

    [Fact]
    public void CreateForNemotron_UsesSubmodelProviderOverrides()
    {
        var options = new OnnxRuntimeOptions
        {
            ExecutionProvider = OnnxExecutionProvider.Cpu,
            EncoderProvider = OnnxExecutionProvider.Cpu,
            DecoderProvider = OnnxExecutionProvider.Cpu,
            JoinerProvider = OnnxExecutionProvider.DirectMl
        };

        NemotronOnnxSessionFactories factories =
            OnnxSessionFactoryResolver.CreateForNemotron(
                options,
                NullLogger.Instance);

        Assert.Equal(OnnxExecutionProvider.Cpu, factories.Encoder.ExecutionProvider);
        Assert.Equal(OnnxExecutionProvider.Cpu, factories.Decoder.ExecutionProvider);
        Assert.Equal(OnnxExecutionProvider.DirectMl, factories.Joiner.ExecutionProvider);
    }
}
