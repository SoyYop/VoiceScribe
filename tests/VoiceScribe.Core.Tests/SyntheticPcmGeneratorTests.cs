using VoiceScribe.Console.Benchmark;
using VoiceScribe.Core.Configuration;

namespace VoiceScribe.Core.Tests;

public sealed class SyntheticPcmGeneratorTests
{
    [Theory]
    [InlineData(8, 1)]
    [InlineData(16, 1)]
    [InlineData(24, 1)]
    [InlineData(32, 1)]
    [InlineData(16, 2)]
    public void CreateSineWaveChunk_ReturnsExpectedPcmByteLength(
        int bitsPerSample,
        int channels)
    {
        AudioCaptureOptions options = CreateAudioOptions(bitsPerSample, channels);
        double phase = 0;

        byte[] pcm = SyntheticPcmGenerator.CreateSineWaveChunk(
            options,
            samples: 160,
            ref phase);

        Assert.Equal(160 * channels * (bitsPerSample / 8), pcm.Length);
        Assert.True(phase > 0);
    }

    [Fact]
    public void CreateSineWaveChunk_DuplicatesSampleAcrossChannels()
    {
        AudioCaptureOptions options = CreateAudioOptions(bitsPerSample: 16, channels: 2);
        double phase = Math.PI / 2;

        byte[] pcm = SyntheticPcmGenerator.CreateSineWaveChunk(
            options,
            samples: 1,
            ref phase);

        short left = BitConverter.ToInt16(pcm, 0);
        short right = BitConverter.ToInt16(pcm, 2);

        Assert.Equal(left, right);
        Assert.NotEqual(0, left);
    }

    [Fact]
    public void CreateSineWaveChunk_RejectsUnsupportedBitDepth()
    {
        AudioCaptureOptions options = CreateAudioOptions(bitsPerSample: 12, channels: 1);
        double phase = 0;

        Assert.Throws<NotSupportedException>(
            () => SyntheticPcmGenerator.CreateSineWaveChunk(
                options,
                samples: 1,
                ref phase));
    }

    private static AudioCaptureOptions CreateAudioOptions(
        int bitsPerSample,
        int channels) =>
        new()
        {
            SampleRate = 16000,
            BitsPerSample = bitsPerSample,
            Channels = channels,
            BufferMilliseconds = 10,
            SilenceThreshold = 0.003f,
            QueueCapacity = 8
        };
}
