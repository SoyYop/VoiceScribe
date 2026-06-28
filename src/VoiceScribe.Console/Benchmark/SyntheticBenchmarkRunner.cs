using NAudio.Wave;
using VoiceScribe.Core.Configuration;
using VoiceScribe.Core.Engine;
using VoiceScribe.Core.ModelAssets;

namespace VoiceScribe.Console.Benchmark;

internal static class SyntheticBenchmarkRunner
{
    internal static async Task RunAsync(
        NemotronEngine engine,
        NemotronModelDefinition modelDefinition,
        AudioCaptureOptions audioOptions,
        int chunks,
        CancellationToken cancellationToken)
    {
        ConsoleOutput.WriteLine(
            $"\n>>> Synthetic benchmark active. Processing {chunks} generated audio chunks. <<<\n",
            ConsoleColor.Cyan);

        double phase = 0;
        for (int i = 0; i < chunks; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();

            byte[] buffer =
                SyntheticPcmGenerator.CreateSineWaveChunk(
                    audioOptions,
                    modelDefinition.ChunkSamples,
                    ref phase);
            engine.ProcessAudioChunk(null, new WaveInEventArgs(buffer, buffer.Length));
        }

        await engine.StopAsync(cancellationToken);
    }
}
