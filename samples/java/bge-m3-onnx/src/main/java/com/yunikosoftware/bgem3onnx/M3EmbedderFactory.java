package com.yunikosoftware.bgem3onnx;

import ai.onnxruntime.OrtException;

/**
 * Factory class for creating M3Embedder instances with common configurations
 */
public class M3EmbedderFactory {
    
    /**
     * Creates an M3Embedder optimized for CPU inference
     * 
     * @param tokenizerPath Path to the ONNX tokenizer model
     * @param modelPath Path to the ONNX embedding model
     * @return M3Embedder configured for CPU
     * @throws OrtException If there's an error initializing the ONNX sessions
     */
    public static M3Embedder createCpuOptimized(String tokenizerPath, String modelPath) throws OrtException {
        M3EmbedderConfig config = new M3EmbedderConfig.Builder()
                .executionProvider(ExecutionProvider.CPU)
                .enableMemoryPattern(true)
                .enableCpuMemArena(true)
                .logSeverityLevel(2)
                .build();

        return new M3Embedder(tokenizerPath, modelPath, config);
    }

    /**
     * Creates an M3Embedder optimized for CUDA inference
     * 
     * @param tokenizerPath Path to the ONNX tokenizer model
     * @param modelPath Path to the ONNX embedding model
     * @param deviceId CUDA device ID (default: 0)
     * @return M3Embedder configured for CUDA
     * @throws OrtException If there's an error initializing the ONNX sessions
     */
    public static M3Embedder createCudaOptimized(String tokenizerPath, String modelPath, int deviceId) throws OrtException {
        M3EmbedderConfig config = new M3EmbedderConfig.Builder()
                .executionProvider(ExecutionProvider.CUDA)
                .fallbackProviders(ExecutionProvider.CPU)
                .cudaDeviceId(deviceId)
                .enableMemoryPattern(true)
                .enableCpuMemArena(false) // Disable CPU memory arena when using GPU
                .logSeverityLevel(2)
                .build();

        return new M3Embedder(tokenizerPath, modelPath, config);
    }

    /**
     * Creates an M3Embedder optimized for CUDA inference with default device ID (0)
     * 
     * @param tokenizerPath Path to the ONNX tokenizer model
     * @param modelPath Path to the ONNX embedding model
     * @return M3Embedder configured for CUDA
     * @throws OrtException If there's an error initializing the ONNX sessions
     */
    public static M3Embedder createCudaOptimized(String tokenizerPath, String modelPath) throws OrtException {
        return createCudaOptimized(tokenizerPath, modelPath, 0);
    }

    /**
     * Creates an M3Embedder with custom configuration
     * 
     * @param tokenizerPath Path to the ONNX tokenizer model
     * @param modelPath Path to the ONNX embedding model
     * @param primaryProvider Primary execution provider
     * @param fallbackProviders Fallback providers in order of preference
     * @param cudaDeviceId CUDA device ID (used only if CUDA is specified)
     * @return M3Embedder with custom configuration
     * @throws OrtException If there's an error initializing the ONNX sessions
     */
    public static M3Embedder createCustom(String tokenizerPath, String modelPath, 
                                         ExecutionProvider primaryProvider, 
                                         ExecutionProvider[] fallbackProviders, 
                                         int cudaDeviceId) throws OrtException {
        M3EmbedderConfig config = new M3EmbedderConfig.Builder()
                .executionProvider(primaryProvider)
                .fallbackProviders(fallbackProviders != null ? fallbackProviders : new ExecutionProvider[]{ExecutionProvider.CPU})
                .cudaDeviceId(cudaDeviceId)
                .enableMemoryPattern(true)
                .enableCpuMemArena(primaryProvider == ExecutionProvider.CPU)
                .logSeverityLevel(2)
                .build();

        return new M3Embedder(tokenizerPath, modelPath, config);
    }
}