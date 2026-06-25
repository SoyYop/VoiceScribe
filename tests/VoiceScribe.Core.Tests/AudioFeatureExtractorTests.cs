using VoiceScribe.Core.Audio;

namespace VoiceScribe.Core.Tests;

public sealed class AudioFeatureExtractorTests
{
    [Fact]
    public void Extract_DefaultModel_ReturnsExpectedShape()
    {
        var extractor = new AudioFeatureExtractor();
        var pcm = new float[8960];

        AudioFeatures features = extractor.Extract(pcm);

        Assert.Equal(65, features.Frames);
        Assert.Equal(128, features.MelBins);
        Assert.Equal(65 * 128, features.Data.Length);
    }

    [Fact]
    public void ResetStreamingState_RestoresInitialOutputForSameInput()
    {
        var extractor = new AudioFeatureExtractor();
        float[] pcm = Enumerable
            .Range(0, 8960)
            .Select(index => MathF.Sin(index * 0.01f) * 0.1f)
            .ToArray();

        AudioFeatures first = extractor.Extract(pcm);
        extractor.Extract(pcm);
        extractor.ResetStreamingState();
        AudioFeatures afterReset = extractor.Extract(pcm);

        Assert.Equal(first.Data, afterReset.Data);
    }
}
