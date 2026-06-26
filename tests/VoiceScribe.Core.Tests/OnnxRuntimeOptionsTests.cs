using System.Text.Json;
using VoiceScribe.Core.Configuration;

namespace VoiceScribe.Core.Tests;

public sealed class OnnxRuntimeOptionsTests
{
    [Theory]
    [InlineData("Cpu", OnnxExecutionProvider.Cpu)]
    [InlineData("directml", OnnxExecutionProvider.DirectMl)]
    [InlineData("CUDA", OnnxExecutionProvider.Cuda)]
    public void ExecutionProvider_DeserializesFromString(
        string jsonValue,
        OnnxExecutionProvider expected)
    {
        OnnxRuntimeOptions? options =
            JsonSerializer.Deserialize<OnnxRuntimeOptions>(
                $$"""{"ExecutionProvider":"{{jsonValue}}"}""");

        Assert.NotNull(options);
        Assert.Equal(expected, options.ExecutionProvider);
    }

    [Fact]
    public void SubmodelProviders_DeserializeFromStrings()
    {
        OnnxRuntimeOptions? options =
            JsonSerializer.Deserialize<OnnxRuntimeOptions>(
                """
                {
                  "ExecutionProvider": "Cpu",
                  "EncoderProvider": "Cpu",
                  "DecoderProvider": "Cpu",
                  "JoinerProvider": "DirectMl"
                }
                """);

        Assert.NotNull(options);
        Assert.Equal(OnnxExecutionProvider.Cpu, options.GetEncoderProvider());
        Assert.Equal(OnnxExecutionProvider.Cpu, options.GetDecoderProvider());
        Assert.Equal(OnnxExecutionProvider.DirectMl, options.GetJoinerProvider());
    }

    [Fact]
    public void SubmodelProviders_DefaultToGlobalExecutionProvider()
    {
        var options = new OnnxRuntimeOptions
        {
            ExecutionProvider = OnnxExecutionProvider.DirectMl
        };

        Assert.Equal(OnnxExecutionProvider.DirectMl, options.GetEncoderProvider());
        Assert.Equal(OnnxExecutionProvider.DirectMl, options.GetDecoderProvider());
        Assert.Equal(OnnxExecutionProvider.DirectMl, options.GetJoinerProvider());
    }
}
