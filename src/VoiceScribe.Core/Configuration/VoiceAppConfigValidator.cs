using VoiceScribe.Core.ModelAssets;

namespace VoiceScribe.Core.Configuration
{
    public static class VoiceAppConfigValidator
    {
        private static readonly int[] SupportedBitDepths = [8, 16, 24, 32];

        public static IReadOnlyList<string> Validate(
            VoiceAppConfig config,
            NemotronModelDefinition model)
        {
            var errors = new List<string>();
            AudioCaptureOptions audio = config.Audio;
            NemotronModelOptions nemotron = config.Nemotron;
            OnnxRuntimeOptions inference = config.Inference;

            if (audio.SampleRate <= 0)
                errors.Add("Audio.SampleRate must be greater than zero.");
            if (!SupportedBitDepths.Contains(audio.BitsPerSample))
                errors.Add("Audio.BitsPerSample must be 8, 16, 24 or 32.");
            if (audio.Channels <= 0)
                errors.Add("Audio.Channels must be greater than zero.");
            if (audio.BufferMilliseconds <= 0)
                errors.Add("Audio.BufferMilliseconds must be greater than zero.");
            if (audio.QueueCapacity <= 0)
                errors.Add("Audio.QueueCapacity must be greater than zero.");
            if (audio.SilenceThreshold is < 0 or > 1)
                errors.Add("Audio.SilenceThreshold must be between 0 and 1.");
            if ((long)audio.SampleRate * audio.BufferMilliseconds % 1000 != 0)
                errors.Add("Audio.SampleRate and BufferMilliseconds must produce a whole number of samples.");

            long samplesPerBuffer =
                (long)audio.SampleRate * audio.BufferMilliseconds / 1000;

            if (audio.SampleRate != model.SampleRate)
                errors.Add(
                    $"Audio.SampleRate is {audio.SampleRate}, but the model requires {model.SampleRate}.");

            if (samplesPerBuffer != model.ChunkSamples)
                errors.Add(
                    $"Audio produces {samplesPerBuffer} samples per buffer, but the model requires {model.ChunkSamples}.");

            if (nemotron.MaxSymbolsPerStep is <= 0)
                errors.Add("Nemotron.MaxSymbolsPerStep must be greater than zero when specified.");

            if (!Enum.IsDefined(inference.ExecutionProvider))
                errors.Add("Inference.ExecutionProvider is invalid.");
            else if (inference.ExecutionProvider != OnnxExecutionProvider.Cpu)
                errors.Add(
                    $"Inference.ExecutionProvider '{inference.ExecutionProvider}' is not available " +
                    "in the CPU runtime variant.");
            if (inference.DeviceId < 0)
                errors.Add("Inference.DeviceId must be zero or greater.");
            if (inference.GpuMemoryLimitMiB is <= 0)
                errors.Add(
                    "Inference.GpuMemoryLimitMiB must be greater than zero when specified.");

            if (model.MaxSymbolsPerStep <= 0)
                errors.Add("The model declares an invalid max_symbols_per_step value.");
            if (model.ChunkSamples <= 0 || model.SampleRate <= 0)
                errors.Add("The model declares invalid audio dimensions.");
            if (model.VocabularySize <= 0)
                errors.Add("The model declares an invalid vocabulary size.");

            long blankId = nemotron.BlankId ?? model.BlankId;
            if (blankId < 0 || blankId >= model.VocabularySize)
                errors.Add(
                    $"Nemotron.BlankId must be within [0, {model.VocabularySize - 1}].");

            return errors;
        }
    }
}
