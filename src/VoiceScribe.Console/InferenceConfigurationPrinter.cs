using VoiceScribe.Core.Configuration;
using VoiceScribe.Core.Engine;

namespace VoiceScribe.Console;

internal static class InferenceConfigurationPrinter
{
    internal static void Print(
        OnnxRuntimeOptions options,
        NemotronOnnxSessionFactories sessionFactories)
    {
        ConsoleOutput.WriteLine("[Inference] Current configuration:", ConsoleColor.DarkGray);
        System.Console.WriteLine($"  Runtime            : {OnnxRuntimeInfo.Name}");
        System.Console.WriteLine($"  Default provider   : {options.ExecutionProvider}");
        System.Console.WriteLine($"  Encoder provider   : {sessionFactories.Encoder.ExecutionProvider}");
        System.Console.WriteLine($"  Decoder provider   : {sessionFactories.Decoder.ExecutionProvider}");
        System.Console.WriteLine($"  Joiner provider    : {sessionFactories.Joiner.ExecutionProvider}");
        System.Console.WriteLine($"  Device id          : {options.DeviceId}");
        System.Console.WriteLine($"  CPU fallback       : {options.AllowCpuFallback}");
        System.Console.WriteLine($"  Profiling          : {options.EnableProfiling}");
        System.Console.WriteLine($"  ORT log severity   : {options.LogSeverityLevel ?? "default"}");
        System.Console.WriteLine(
            $"  ORT log verbosity  : {FormatOptional(options.LogVerbosityLevel)}");
        System.Console.WriteLine(
            $"  GPU memory limit   : {FormatMemoryLimit(options.GpuMemoryLimitMiB)}");

        if (UsesGpuProvider(sessionFactories) && options.AllowCpuFallback)
        {
            System.Console.WriteLine(
                "  Note               : individual models may fall back to CPU if the GPU provider cannot initialize them.");
        }
    }

    private static bool UsesGpuProvider(
        NemotronOnnxSessionFactories sessionFactories) =>
        IsGpuProvider(sessionFactories.Encoder.ExecutionProvider) ||
        IsGpuProvider(sessionFactories.Decoder.ExecutionProvider) ||
        IsGpuProvider(sessionFactories.Joiner.ExecutionProvider);

    private static bool IsGpuProvider(OnnxExecutionProvider provider) =>
        provider is OnnxExecutionProvider.DirectMl;

    private static string FormatOptional(int? value) =>
        value.HasValue ? value.Value.ToString() : "default";

    private static string FormatMemoryLimit(int? value) =>
        value.HasValue ? $"{value} MiB" : "not set";
}
