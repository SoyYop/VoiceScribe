using Microsoft.ML.OnnxRuntime;
using VoiceScribe.Core.Configuration;

namespace VoiceScribe.Core.Engine;

internal static class OnnxSessionOptionsConfigurator
{
    internal static void ApplyLogging(
        SessionOptions sessionOptions,
        OnnxRuntimeOptions options)
    {
        if (TryResolveLogSeverityLevel(
            options.LogSeverityLevel,
            out OrtLoggingLevel severityLevel))
        {
            sessionOptions.LogSeverityLevel = severityLevel;
        }

        if (options.LogVerbosityLevel.HasValue)
            sessionOptions.LogVerbosityLevel = options.LogVerbosityLevel.Value;
    }

    internal static bool IsValidLogSeverityLevel(string? value) =>
        string.IsNullOrWhiteSpace(value) ||
        TryResolveLogSeverityLevel(value, out _);

    private static bool TryResolveLogSeverityLevel(
        string? value,
        out OrtLoggingLevel severityLevel)
    {
        switch (value?.Trim().ToUpperInvariant())
        {
            case "VERBOSE":
                severityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_VERBOSE;
                return true;
            case "INFO":
            case "INFORMATION":
                severityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_INFO;
                return true;
            case "WARNING":
            case "WARN":
                severityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_WARNING;
                return true;
            case "ERROR":
                severityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_ERROR;
                return true;
            case "FATAL":
                severityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_FATAL;
                return true;
            default:
                severityLevel = default;
                return false;
        }
    }
}
