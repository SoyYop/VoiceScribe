using System.Text.Json.Serialization;

namespace VoiceScribe.Core.Configuration;

[JsonConverter(typeof(JsonStringEnumConverter<OnnxExecutionProvider>))]
public enum OnnxExecutionProvider
{
    Cpu,
    DirectMl,
    Cuda
}
