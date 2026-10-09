using System.Text.Json.Serialization;

namespace VoiceScribe.Notepad.Configuration;

[JsonConverter(typeof(JsonStringEnumConverter<OnnxExecutionProvider>))]
public enum OnnxExecutionProvider
{
    Cpu,
    DirectMl
}
