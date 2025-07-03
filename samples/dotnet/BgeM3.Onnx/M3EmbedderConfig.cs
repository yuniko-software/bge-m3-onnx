using Microsoft.ML.OnnxRuntime;

namespace BgeM3.Onnx;

/// <summary>
/// Configuration for M3Embedder initialization
/// </summary>
public record M3EmbedderConfig
{
    /// <summary>
    /// Primary execution provider to use
    /// </summary>
    public ExecutionProvider ExecutionProvider { get; init; } = ExecutionProvider.CPU;

    /// <summary>
    /// Fallback execution providers in order of preference
    /// </summary>
    public ExecutionProvider[] FallbackProviders { get; init; } = [ExecutionProvider.CPU];

    /// <summary>
    /// CUDA device ID (only used when ExecutionProvider is CUDA)
    /// </summary>
    public int CudaDeviceId { get; init; } = 0;

    /// <summary>
    /// Enable memory pattern optimization
    /// </summary>
    public bool EnableMemoryPattern { get; init; } = true;

    /// <summary>
    /// Enable CPU memory arena
    /// </summary>
    public bool EnableCpuMemArena { get; init; } = true;

    /// <summary>
    /// Log severity level (0=Verbose, 1=Info, 2=Warning, 3=Error, 4=Fatal)
    /// </summary>
    public OrtLoggingLevel LogSeverityLevel { get; init; } = OrtLoggingLevel.ORT_LOGGING_LEVEL_WARNING;
}
