using VoiceScribe.Core.Configuration;

namespace VoiceScribe.Core.Engine;

/// <summary>
/// ONNX Runtime session factories selected for each Nemotron submodel.
/// </summary>
public sealed class NemotronOnnxSessionFactories
{
    public NemotronOnnxSessionFactories(
        IOnnxSessionFactory encoder,
        IOnnxSessionFactory decoder,
        IOnnxSessionFactory joiner)
    {
        Encoder = encoder;
        Decoder = decoder;
        Joiner = joiner;
    }

    public IOnnxSessionFactory Encoder { get; }

    public IOnnxSessionFactory Decoder { get; }

    public IOnnxSessionFactory Joiner { get; }

    public string Describe() =>
        $"encoder={Encoder.ExecutionProvider}, " +
        $"decoder={Decoder.ExecutionProvider}, " +
        $"joiner={Joiner.ExecutionProvider}";
}
