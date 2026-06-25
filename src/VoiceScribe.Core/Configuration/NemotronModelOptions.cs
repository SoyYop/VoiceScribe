namespace VoiceScribe.Core.Configuration
{
    /// <summary>
    /// Runtime settings and explicit overrides for the supported Nemotron RNN-T export.
    /// Tensor dimensions are derived from ONNX metadata whenever possible.
    /// </summary>
    public sealed class NemotronModelOptions
    {
        /// <summary>
        /// Language identifier passed to the encoder.
        /// </summary>
        public long LanguageId { get; set; } = 101;

        /// <summary>
        /// Maximum number of non-blank symbols emitted for one acoustic frame.
        /// </summary>
        public int? MaxSymbolsPerStep { get; set; }

        /// <summary>
        /// Optional blank token override. When null, the engine uses the final
        /// class in the joint output tensor.
        /// </summary>
        public long? BlankId { get; set; }
    }
}
