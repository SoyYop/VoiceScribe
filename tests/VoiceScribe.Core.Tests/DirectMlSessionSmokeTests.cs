using Microsoft.Extensions.Logging.Abstractions;
using VoiceScribe.Core.Configuration;
using VoiceScribe.Core.Engine;

namespace VoiceScribe.Core.Tests;

public sealed class DirectMlSessionSmokeTests
{
    [Theory]
    [InlineData("encoder.onnx")]
    [InlineData("decoder.onnx")]
    [InlineData("joint.onnx")]
    public void CreateSession_WhenDirectMlProviderAndModelIsAvailable(
        string modelFileName)
    {
        string? modelFolder = FindModelFolder();
        if (modelFolder is null)
            return;

        var options = new OnnxRuntimeOptions
        {
            ExecutionProvider = OnnxExecutionProvider.DirectMl,
            DeviceId = 0
        };
        IOnnxSessionFactory factory =
            OnnxSessionFactoryResolver.Create(options, NullLogger.Instance);

        using var session =
            factory.CreateSession(Path.Combine(modelFolder, modelFileName));

        Assert.NotEmpty(session.InputMetadata);
    }

    private static string? FindModelFolder()
    {
        DirectoryInfo? directory =
            new(AppContext.BaseDirectory);

        while (directory is not null)
        {
            string candidate = Path.Combine(
                directory.FullName,
                "artifacts",
                "models",
                "nemotron-3.5-asr");

            if (File.Exists(Path.Combine(candidate, "encoder.onnx")) &&
                File.Exists(Path.Combine(candidate, "decoder.onnx")) &&
                File.Exists(Path.Combine(candidate, "joint.onnx")))
            {
                return candidate;
            }

            directory = directory.Parent;
        }

        return null;
    }
}
