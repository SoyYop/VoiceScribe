using VoiceScribe.Core.Configuration;

namespace VoiceScribe.Console.Benchmark;

internal static class SyntheticPcmGenerator
{
    internal const double DefaultFrequencyHz = 440;
    internal const double DefaultAmplitude = 0.25;

    internal static byte[] CreateSineWaveChunk(
        AudioCaptureOptions audioOptions,
        int samples,
        ref double phase,
        double frequencyHz = DefaultFrequencyHz,
        double amplitude = DefaultAmplitude)
    {
        ArgumentNullException.ThrowIfNull(audioOptions);

        int bytesPerSample = audioOptions.BitsPerSample / 8;
        int bytesPerFrame = bytesPerSample * audioOptions.Channels;
        byte[] buffer = new byte[checked(samples * bytesPerFrame)];
        double phaseStep = 2 * Math.PI * frequencyHz / audioOptions.SampleRate;

        for (int sampleIndex = 0; sampleIndex < samples; sampleIndex++)
        {
            float sample = (float)(Math.Sin(phase) * amplitude);
            phase += phaseStep;

            for (int channel = 0; channel < audioOptions.Channels; channel++)
            {
                int offset = sampleIndex * bytesPerFrame + channel * bytesPerSample;
                WritePcmSample(buffer, offset, audioOptions.BitsPerSample, sample);
            }
        }

        return buffer;
    }

    private static void WritePcmSample(
        byte[] buffer,
        int offset,
        int bitsPerSample,
        float sample)
    {
        switch (bitsPerSample)
        {
            case 8:
                buffer[offset] = (byte)Math.Clamp(
                    (int)Math.Round(sample * 127 + 128),
                    0,
                    255);
                break;
            case 16:
                BitConverter.TryWriteBytes(
                    buffer.AsSpan(offset, 2),
                    (short)Math.Clamp(
                        (int)Math.Round(sample * 32767),
                        short.MinValue,
                        short.MaxValue));
                break;
            case 24:
                int sample24 = Math.Clamp(
                    (int)Math.Round(sample * 8388607),
                    -8388608,
                    8388607);
                buffer[offset] = (byte)(sample24 & 0xFF);
                buffer[offset + 1] = (byte)((sample24 >> 8) & 0xFF);
                buffer[offset + 2] = (byte)((sample24 >> 16) & 0xFF);
                break;
            case 32:
                BitConverter.TryWriteBytes(
                    buffer.AsSpan(offset, 4),
                    Math.Clamp(
                        (int)Math.Round(sample * int.MaxValue),
                        int.MinValue,
                        int.MaxValue));
                break;
            default:
                throw new NotSupportedException(
                    $"PCM bit depth '{bitsPerSample}' is not supported.");
        }
    }
}
