package com.yunikosoftware.bgem3onnx;

/**
 * Supported execution providers for ONNX inference
 */
public enum ExecutionProvider {
    /**
     * CPU execution provider (default, always available)
     */
    CPU,

    /**
     * CUDA execution provider (requires CUDA-enabled GPU and CUDA runtime)
     */
    CUDA
}