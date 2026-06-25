namespace VoiceScribe.Core.Configuration
{
    /// <summary>
    /// Audio capture and chunk filtering settings used by the console and inference engine.
    /// </summary>
    public sealed class AudioCaptureOptions
    {
        public int SampleRate { get; set; } = 16000;

        public int BitsPerSample { get; set; } = 16;

        public int Channels { get; set; } = 1;

        public int BufferMilliseconds { get; set; } = 560;

        public float SilenceThreshold { get; set; } = 0.003f;

        public int ShutdownDrainMilliseconds { get; set; } = 300;

        public int SamplesPerBuffer =>
            checked(SampleRate * BufferMilliseconds / 1000);
    }
}
