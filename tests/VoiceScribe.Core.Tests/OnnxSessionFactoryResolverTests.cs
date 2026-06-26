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

    [Theory]
    [InlineData(OnnxExecutionProvider.DirectMl)]
    [InlineData(OnnxExecutionProvider.Cuda)]
    public void Create_RejectsProviderUnavailableInCpuVariant(
        OnnxExecutionProvider provider)
    {
        var options = new OnnxRuntimeOptions
        {
            ExecutionProvider = provider
        };

        NotSupportedException exception = Assert.Throws<NotSupportedException>(
            () => OnnxSessionFactoryResolver.Create(
                options,
                NullLogger.Instance));

        Assert.Contains("CPU runtime variant", exception.Message);
    }
}
