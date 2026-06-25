using System.Text.Json;

namespace VoiceScribe.Core.ModelAssets
{
    /// <summary>
    /// Model contract declared by Nemotron's genai_config.json.
    /// </summary>
    public sealed class NemotronModelDefinition
    {
        public required int VocabularySize { get; init; }
        public required long BlankId { get; init; }
        public required int MaxSymbolsPerStep { get; init; }
        public required int SampleRate { get; init; }
        public required int ChunkSamples { get; init; }
        public required int PreEncodeCacheSize { get; init; }
        public required int EncoderHiddenSize { get; init; }
        public required int DecoderHiddenSize { get; init; }
        public required int DecoderLayerCount { get; init; }
        public required EncoderContract Encoder { get; init; }
        public required DecoderContract Decoder { get; init; }
        public required JoinerContract Joiner { get; init; }

        public static NemotronModelDefinition Load(string path)
        {
            using var document = JsonDocument.Parse(File.ReadAllText(path));
            JsonElement model = document.RootElement.GetProperty("model");
            JsonElement encoder = model.GetProperty("encoder");
            JsonElement decoder = model.GetProperty("decoder");
            JsonElement joiner = model.GetProperty("joiner");

            JsonElement encoderInputs = encoder.GetProperty("inputs");
            JsonElement encoderOutputs = encoder.GetProperty("outputs");
            JsonElement decoderInputs = decoder.GetProperty("inputs");
            JsonElement decoderOutputs = decoder.GetProperty("outputs");
            JsonElement joinerInputs = joiner.GetProperty("inputs");
            JsonElement joinerOutputs = joiner.GetProperty("outputs");

            return new NemotronModelDefinition
            {
                VocabularySize = model.GetProperty("vocab_size").GetInt32(),
                BlankId = model.GetProperty("blank_id").GetInt64(),
                MaxSymbolsPerStep = model.GetProperty("max_symbols_per_step").GetInt32(),
                SampleRate = model.GetProperty("sample_rate").GetInt32(),
                ChunkSamples = model.GetProperty("chunk_samples").GetInt32(),
                PreEncodeCacheSize = model.GetProperty("pre_encode_cache_size").GetInt32(),
                EncoderHiddenSize = encoder.GetProperty("hidden_size").GetInt32(),
                DecoderHiddenSize = decoder.GetProperty("hidden_size").GetInt32(),
                DecoderLayerCount = decoder.GetProperty("num_hidden_layers").GetInt32(),
                Encoder = new EncoderContract
                {
                    FileName = encoder.GetProperty("filename").GetString()!,
                    AudioFeaturesInput = encoderInputs.GetProperty("audio_features").GetString()!,
                    InputLengthsInput = encoderInputs.GetProperty("input_lengths").GetString()!,
                    CacheLastChannelInput = encoderInputs.GetProperty("cache_last_channel").GetString()!,
                    CacheLastTimeInput = encoderInputs.GetProperty("cache_last_time").GetString()!,
                    CacheLastChannelLengthInput = encoderInputs.GetProperty("cache_last_channel_len").GetString()!,
                    LanguageIdInput = encoderInputs.GetProperty("lang_id").GetString()!,
                    EncoderOutputsOutput = encoderOutputs.GetProperty("encoder_outputs").GetString()!,
                    CacheLastChannelOutput = encoderOutputs.GetProperty("cache_last_channel_next").GetString()!,
                    CacheLastTimeOutput = encoderOutputs.GetProperty("cache_last_time_next").GetString()!,
                    CacheLastChannelLengthOutput = encoderOutputs.GetProperty("cache_last_channel_len_next").GetString()!
                },
                Decoder = new DecoderContract
                {
                    FileName = decoder.GetProperty("filename").GetString()!,
                    TargetsInput = decoderInputs.GetProperty("targets").GetString()!,
                    HiddenStateInput = decoderInputs.GetProperty("lstm_hidden_state").GetString()!,
                    CellStateInput = decoderInputs.GetProperty("lstm_cell_state").GetString()!,
                    DecoderOutput = decoderOutputs.GetProperty("outputs").GetString()!,
                    HiddenStateOutput = decoderOutputs.GetProperty("lstm_hidden_state").GetString()!,
                    CellStateOutput = decoderOutputs.GetProperty("lstm_cell_state").GetString()!
                },
                Joiner = new JoinerContract
                {
                    FileName = joiner.GetProperty("filename").GetString()!,
                    EncoderOutputInput = joinerInputs.GetProperty("encoder_outputs").GetString()!,
                    DecoderOutputInput = joinerInputs.GetProperty("decoder_outputs").GetString()!,
                    LogitsOutput = joinerOutputs.GetProperty("logits").GetString()!
                }
            };
        }
    }

    public sealed class EncoderContract
    {
        public required string FileName { get; init; }
        public required string AudioFeaturesInput { get; init; }
        public required string InputLengthsInput { get; init; }
        public required string CacheLastChannelInput { get; init; }
        public required string CacheLastTimeInput { get; init; }
        public required string CacheLastChannelLengthInput { get; init; }
        public required string LanguageIdInput { get; init; }
        public required string EncoderOutputsOutput { get; init; }
        public required string CacheLastChannelOutput { get; init; }
        public required string CacheLastTimeOutput { get; init; }
        public required string CacheLastChannelLengthOutput { get; init; }
    }

    public sealed class DecoderContract
    {
        public required string FileName { get; init; }
        public required string TargetsInput { get; init; }
        public required string HiddenStateInput { get; init; }
        public required string CellStateInput { get; init; }
        public required string DecoderOutput { get; init; }
        public required string HiddenStateOutput { get; init; }
        public required string CellStateOutput { get; init; }
    }

    public sealed class JoinerContract
    {
        public required string FileName { get; init; }
        public required string EncoderOutputInput { get; init; }
        public required string DecoderOutputInput { get; init; }
        public required string LogitsOutput { get; init; }
    }
}
