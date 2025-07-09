package com.yunikosoftware.bgem3onnx;

/**
 * Configuration for M3Embedder initialization
 */
public class M3EmbedderConfig {
    private final ExecutionProvider executionProvider;
    private final ExecutionProvider[] fallbackProviders;
    private final int cudaDeviceId;
    private final boolean enableMemoryPattern;
    private final boolean enableCpuMemArena;
    private final int logSeverityLevel;

    public M3EmbedderConfig() {
        this(ExecutionProvider.CPU, new ExecutionProvider[]{ExecutionProvider.CPU}, 0, true, true, 2);
    }

    public M3EmbedderConfig(ExecutionProvider executionProvider, ExecutionProvider[] fallbackProviders, 
                           int cudaDeviceId, boolean enableMemoryPattern, boolean enableCpuMemArena, 
                           int logSeverityLevel) {
        this.executionProvider = executionProvider;
        this.fallbackProviders = fallbackProviders;
        this.cudaDeviceId = cudaDeviceId;
        this.enableMemoryPattern = enableMemoryPattern;
        this.enableCpuMemArena = enableCpuMemArena;
        this.logSeverityLevel = logSeverityLevel;
    }

    public ExecutionProvider getExecutionProvider() {
        return executionProvider;
    }

    public ExecutionProvider[] getFallbackProviders() {
        return fallbackProviders;
    }

    public int getCudaDeviceId() {
        return cudaDeviceId;
    }

    public boolean isEnableMemoryPattern() {
        return enableMemoryPattern;
    }

    public boolean isEnableCpuMemArena() {
        return enableCpuMemArena;
    }

    public int getLogSeverityLevel() {
        return logSeverityLevel;
    }

    public static class Builder {
        private ExecutionProvider executionProvider = ExecutionProvider.CPU;
        private ExecutionProvider[] fallbackProviders = new ExecutionProvider[]{ExecutionProvider.CPU};
        private int cudaDeviceId = 0;
        private boolean enableMemoryPattern = true;
        private boolean enableCpuMemArena = true;
        private int logSeverityLevel = 2;

        public Builder executionProvider(ExecutionProvider executionProvider) {
            this.executionProvider = executionProvider;
            return this;
        }

        public Builder fallbackProviders(ExecutionProvider... fallbackProviders) {
            this.fallbackProviders = fallbackProviders;
            return this;
        }

        public Builder cudaDeviceId(int cudaDeviceId) {
            this.cudaDeviceId = cudaDeviceId;
            return this;
        }

        public Builder enableMemoryPattern(boolean enableMemoryPattern) {
            this.enableMemoryPattern = enableMemoryPattern;
            return this;
        }

        public Builder enableCpuMemArena(boolean enableCpuMemArena) {
            this.enableCpuMemArena = enableCpuMemArena;
            return this;
        }

        public Builder logSeverityLevel(int logSeverityLevel) {
            this.logSeverityLevel = logSeverityLevel;
            return this;
        }

        public M3EmbedderConfig build() {
            return new M3EmbedderConfig(executionProvider, fallbackProviders, cudaDeviceId, 
                                      enableMemoryPattern, enableCpuMemArena, logSeverityLevel);
        }
    }
}