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
    public void Create_HandlesDirectMlAccordingToRuntimeVariant()
    {
        var options = new OnnxRuntimeOptions
        {
            ExecutionProvider = OnnxExecutionProvider.DirectMl
        };

        if (OnnxRuntimeVariant.Supports(OnnxExecutionProvider.DirectMl))
        {
            IOnnxSessionFactory factory =
                OnnxSessionFactoryResolver.Create(
                    options,
                    NullLogger.Instance);

            Assert.Equal(
                OnnxExecutionProvider.DirectMl,
                factory.ExecutionProvider);
            Assert.Equal("DirectMlOnnxSessionFactory", factory.GetType().Name);
        }
        else
        {
            NotSupportedException exception =
                Assert.Throws<NotSupportedException>(
                    () => OnnxSessionFactoryResolver.Create(
                        options,
                        NullLogger.Instance));

            Assert.Contains("runtime variant", exception.Message);
        }
    }

    [Fact]
    public void Create_RejectsCudaUntilCudaVariantExists()
    {
        var options = new OnnxRuntimeOptions
        {
            ExecutionProvider = OnnxExecutionProvider.Cuda
        };

        NotSupportedException exception =
            Assert.Throws<NotSupportedException>(
                () => OnnxSessionFactoryResolver.Create(
                options,
                NullLogger.Instance));

        Assert.Contains("runtime variant", exception.Message);
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

        if (OnnxRuntimeVariant.Supports(OnnxExecutionProvider.DirectMl))
        {
            NemotronOnnxSessionFactories factories =
                OnnxSessionFactoryResolver.CreateForNemotron(
                    options,
                    NullLogger.Instance);

            Assert.Equal(OnnxExecutionProvider.Cpu, factories.Encoder.ExecutionProvider);
            Assert.Equal(OnnxExecutionProvider.Cpu, factories.Decoder.ExecutionProvider);
            Assert.Equal(OnnxExecutionProvider.DirectMl, factories.Joiner.ExecutionProvider);
        }
        else
        {
            NotSupportedException exception =
                Assert.Throws<NotSupportedException>(
                    () => OnnxSessionFactoryResolver.CreateForNemotron(
                        options,
                        NullLogger.Instance));

            Assert.Contains("runtime variant", exception.Message);
        }
    }
}
