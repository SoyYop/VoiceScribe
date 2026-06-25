namespace VoiceScribe.Core.ModelAssets
{
    /// <summary>
    /// Standard file names required by the supported Nemotron ONNX export.
    /// </summary>
    public static class NemotronModelFiles
    {
        public const string Encoder = "encoder.onnx";
        public const string EncoderData = "encoder.onnx.data";
        public const string Decoder = "decoder.onnx";
        public const string DecoderData = "decoder.onnx.data";
        public const string Joint = "joint.onnx";
        public const string JointData = "joint.onnx.data";
        public const string Tokenizer = "tokenizer.json";
        public const string GenAiConfig = "genai_config.json";
        public const string AudioProcessorConfig = "audio_processor_config.json";
        public const string ModelConfig = "model_config.json";

        public static List<string> CreateRequiredFileList() =>
        [
            Encoder,
            EncoderData,
            Decoder,
            DecoderData,
            Joint,
            JointData,
            Tokenizer,
            GenAiConfig,
            AudioProcessorConfig,
            ModelConfig
        ];
    }
}
