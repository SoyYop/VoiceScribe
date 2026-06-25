using VoiceScribe.Core.ModelAssets;

namespace VoiceScribe.Core.Tests;

public sealed class NemotronModelDefinitionTests
{
    [Fact]
    public void Load_ReadsModelContractAndNodeNames()
    {
        string path = WriteTemporaryConfig();

        try
        {
            NemotronModelDefinition model = NemotronModelDefinition.Load(path);

            Assert.Equal(13088, model.VocabularySize);
            Assert.Equal(13087, model.BlankId);
            Assert.Equal(8960, model.ChunkSamples);
            Assert.Equal(1024, model.EncoderHiddenSize);
            Assert.Equal(640, model.DecoderHiddenSize);
            Assert.Equal("audio_signal", model.Encoder.AudioFeaturesInput);
            Assert.Equal("decoder_output", model.Decoder.DecoderOutput);
            Assert.Equal("joint_output", model.Joiner.LogitsOutput);
        }
        finally
        {
            File.Delete(path);
        }
    }

    private static string WriteTemporaryConfig()
    {
        string path = Path.Combine(
            Path.GetTempPath(),
            $"voicescribe-genai-{Guid.NewGuid():N}.json");

        File.WriteAllText(path, """
        {
          "model": {
            "vocab_size": 13088,
            "blank_id": 13087,
            "max_symbols_per_step": 10,
            "sample_rate": 16000,
            "chunk_samples": 8960,
            "pre_encode_cache_size": 9,
            "encoder": {
              "filename": "encoder.onnx",
              "hidden_size": 1024,
              "inputs": {
                "audio_features": "audio_signal",
                "input_lengths": "length",
                "cache_last_channel": "cache_last_channel",
                "cache_last_time": "cache_last_time",
                "cache_last_channel_len": "cache_last_channel_len",
                "lang_id": "lang_id"
              },
              "outputs": {
                "encoder_outputs": "outputs",
                "cache_last_channel_next": "cache_last_channel_next",
                "cache_last_time_next": "cache_last_time_next",
                "cache_last_channel_len_next": "cache_last_channel_len_next"
              }
            },
            "decoder": {
              "filename": "decoder.onnx",
              "hidden_size": 640,
              "num_hidden_layers": 2,
              "inputs": {
                "targets": "targets",
                "lstm_hidden_state": "h_in",
                "lstm_cell_state": "c_in"
              },
              "outputs": {
                "outputs": "decoder_output",
                "lstm_hidden_state": "h_out",
                "lstm_cell_state": "c_out"
              }
            },
            "joiner": {
              "filename": "joint.onnx",
              "inputs": {
                "encoder_outputs": "encoder_output",
                "decoder_outputs": "decoder_output"
              },
              "outputs": {
                "logits": "joint_output"
              }
            }
          }
        }
        """);

        return path;
    }
}
