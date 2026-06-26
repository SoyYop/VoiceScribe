namespace VoiceScribe.Core.Configuration;

/// <summary>
/// Operational settings for the ONNX Runtime execution provider.
/// </summary>
public sealed class OnnxRuntimeOptions
{
    /// <summary>
    /// Execution provider requested for all model sessions.
    /// </summary>
    public OnnxExecutionProvider ExecutionProvider { get; set; } =
        OnnxExecutionProvider.Cpu;

    /// <summary>
    /// Device selected by GPU execution providers.
    /// </summary>
    public int DeviceId { get; set; }

    /// <summary>
    /// Optional GPU memory limit. It is ignored by the CPU provider.
    /// </summary>
    public int? GpuMemoryLimitMiB { get; set; }

    /// <summary>
    /// Enables ONNX Runtime profiling when supported by the selected provider.
    /// </summary>
    public bool EnableProfiling { get; set; }
}
