using Microsoft.ML.OnnxRuntime;

namespace BgeM3.Onnx;

/// <summary>
/// Factory class for creating M3Embedder instances with common configurations
/// </summary>
public static class M3EmbedderFactory
{
    /// <summary>
    /// Creates an M3Embedder optimized for CPU inference
    /// </summary>
    /// <param name="tokenizerPath">Path to the ONNX tokenizer model</param>
    /// <param name="modelPath">Path to the ONNX embedding model</param>
    /// <returns>M3Embedder configured for CPU</returns>
    public static M3Embedder CreateCpuOptimized(string tokenizerPath, string modelPath)
    {
        var config = new M3EmbedderConfig
        {
            ExecutionProvider = ExecutionProvider.CPU,
            EnableMemoryPattern = true,
            EnableCpuMemArena = true,
            LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_WARNING
        };

        return new M3Embedder(tokenizerPath, modelPath, config);
    }

    /// <summary>
    /// Creates an M3Embedder optimized for CUDA inference
    /// </summary>
    /// <param name="tokenizerPath">Path to the ONNX tokenizer model</param>
    /// <param name="modelPath">Path to the ONNX embedding model</param>
    /// <param name="deviceId">CUDA device ID (default: 0)</param>
    /// <returns>M3Embedder configured for CUDA</returns>
    public static M3Embedder CreateCudaOptimized(string tokenizerPath, string modelPath, int deviceId = 0)
    {
        var config = new M3EmbedderConfig
        {
            ExecutionProvider = ExecutionProvider.CUDA,
            FallbackProviders = [ExecutionProvider.CPU],
            CudaDeviceId = deviceId,
            EnableMemoryPattern = true,
            EnableCpuMemArena = false, // Disable CPU memory arena when using GPU
            LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_WARNING
        };

        return new M3Embedder(tokenizerPath, modelPath, config);
    }

    /// <summary>
    /// Creates an M3Embedder with custom configuration
    /// </summary>
    /// <param name="tokenizerPath">Path to the ONNX tokenizer model</param>
    /// <param name="modelPath">Path to the ONNX embedding model</param>
    /// <param name="primaryProvider">Primary execution provider</param>
    /// <param name="fallbackProviders">Fallback providers in order of preference</param>
    /// <param name="cudaDeviceId">CUDA device ID (used only if CUDA is specified)</param>
    /// <returns>M3Embedder with custom configuration</returns>
    public static M3Embedder CreateCustom(
        string tokenizerPath,
        string modelPath,
        ExecutionProvider primaryProvider,
        ExecutionProvider[]? fallbackProviders = null,
        int cudaDeviceId = 0)
    {
        var config = new M3EmbedderConfig
        {
            ExecutionProvider = primaryProvider,
            FallbackProviders = fallbackProviders ?? [ExecutionProvider.CPU],
            CudaDeviceId = cudaDeviceId,
            EnableMemoryPattern = true,
            EnableCpuMemArena = primaryProvider == ExecutionProvider.CPU,
            LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_WARNING
        };

        return new M3Embedder(tokenizerPath, modelPath, config);
    }
}