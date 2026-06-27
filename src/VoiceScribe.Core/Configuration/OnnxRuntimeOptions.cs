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
    /// Optional execution provider override for encoder.onnx.
    /// When unset, ExecutionProvider is used.
    /// </summary>
    public OnnxExecutionProvider? EncoderProvider { get; set; }

    /// <summary>
    /// Optional execution provider override for decoder.onnx.
    /// When unset, ExecutionProvider is used.
    /// </summary>
    public OnnxExecutionProvider? DecoderProvider { get; set; }

    /// <summary>
    /// Optional execution provider override for joint.onnx.
    /// When unset, ExecutionProvider is used.
    /// </summary>
    public OnnxExecutionProvider? JoinerProvider { get; set; }

    /// <summary>
    /// Device selected by GPU execution providers.
    /// </summary>
    public int DeviceId { get; set; }

    /// <summary>
    /// Optional GPU memory limit. It is ignored by the CPU provider.
    /// </summary>
    public int? GpuMemoryLimitMiB { get; set; }

    /// <summary>
    /// Allows a GPU session that cannot initialize to be recreated on CPU.
    /// The fallback is logged with the original provider error.
    /// </summary>
    public bool AllowCpuFallback { get; set; } = true;

    /// <summary>
    /// Enables ONNX Runtime profiling when supported by the selected provider.
    /// </summary>
    public bool EnableProfiling { get; set; }

    /// <summary>
    /// Optional ONNX Runtime log severity level: Verbose, Info, Warning, Error or Fatal.
    /// When unset, ONNX Runtime uses its default severity.
    /// </summary>
    public string? LogSeverityLevel { get; set; }

    /// <summary>
    /// Optional ONNX Runtime verbosity level. It is mainly useful with Verbose logging.
    /// </summary>
    public int? LogVerbosityLevel { get; set; }

    public OnnxExecutionProvider GetEncoderProvider() =>
        EncoderProvider ?? ExecutionProvider;

    public OnnxExecutionProvider GetDecoderProvider() =>
        DecoderProvider ?? ExecutionProvider;

    public OnnxExecutionProvider GetJoinerProvider() =>
        JoinerProvider ?? ExecutionProvider;
}
