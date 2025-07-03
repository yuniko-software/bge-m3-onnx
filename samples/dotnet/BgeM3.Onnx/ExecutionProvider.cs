namespace BgeM3.Onnx;

/// <summary>
/// Supported execution providers for ONNX inference
/// </summary>
public enum ExecutionProvider
{
    /// <summary>
    /// CPU execution provider (default, always available)
    /// </summary>
    CPU,

    /// <summary>
    /// CUDA execution provider (requires CUDA-enabled GPU and CUDA runtime)
    /// </summary>
    CUDA
}
