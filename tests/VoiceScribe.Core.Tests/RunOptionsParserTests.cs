using VoiceScribe.Console.CommandLine;

namespace VoiceScribe.Core.Tests;

public sealed class RunOptionsParserTests
{
    [Fact]
    public void Parse_NoArgs_ReturnsInteractiveDefaults()
    {
        RunOptions options = RunOptionsParser.Parse([]);

        Assert.False(options.Benchmark);
        Assert.Equal(20, options.BenchmarkChunks);
        Assert.Null(options.TranscriptPath);
    }

    [Fact]
    public void Parse_PathArg_TreatsArgAsTranscriptPath()
    {
        RunOptions options = RunOptionsParser.Parse(["transcript.txt"]);

        Assert.False(options.Benchmark);
        Assert.Equal(20, options.BenchmarkChunks);
        Assert.Equal("transcript.txt", options.TranscriptPath);
    }

    [Fact]
    public void Parse_BenchmarkWithoutCount_UsesDefaultChunkCount()
    {
        RunOptions options = RunOptionsParser.Parse(["--benchmark"]);

        Assert.True(options.Benchmark);
        Assert.Equal(20, options.BenchmarkChunks);
        Assert.Null(options.TranscriptPath);
    }

    [Fact]
    public void Parse_BenchmarkWithCount_UsesProvidedChunkCount()
    {
        RunOptions options = RunOptionsParser.Parse(["--benchmark", "64"]);

        Assert.True(options.Benchmark);
        Assert.Equal(64, options.BenchmarkChunks);
    }

    [Theory]
    [InlineData("0")]
    [InlineData("-1")]
    [InlineData("abc")]
    public void Parse_BenchmarkWithInvalidCount_Throws(string value)
    {
        ArgumentException exception =
            Assert.Throws<ArgumentException>(
                () => RunOptionsParser.Parse(["--benchmark", value]));

        Assert.Contains("positive chunk count", exception.Message);
    }
}
