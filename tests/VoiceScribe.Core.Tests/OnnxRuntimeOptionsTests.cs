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
}
