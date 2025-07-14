package com.yunikosoftware.bgem3onnx.performance;

import com.fasterxml.jackson.annotation.JsonProperty;

/**
 * Benchmark result data structure
 */
public class BenchmarkResult {
    @JsonProperty("scenario")
    private String scenario;
    
    @JsonProperty("total_time_seconds")
    private double totalTimeSeconds;
    
    @JsonProperty("initialization_time_seconds")
    private double initializationTimeSeconds;
    
    @JsonProperty("average_latency_ms")
    private double averageLatencyMs;
    
    @JsonProperty("median_latency_ms")
    private double medianLatencyMs;
    
    @JsonProperty("min_latency_ms")
    private double minLatencyMs;
    
    @JsonProperty("max_latency_ms")
    private double maxLatencyMs;
    
    @JsonProperty("throughput_texts_per_second")
    private double throughputTextsPerSecond;
    
    @JsonProperty("successful_embeddings")
    private int successfulEmbeddings;
    
    @JsonProperty("failed_embeddings")
    private int failedEmbeddings;
    
    @JsonProperty("per_text_latencies_ms")
    private double[] perTextLatenciesMs;
    
    @JsonProperty("execution_provider")
    private String executionProvider;
    
    @JsonProperty("error")
    private String error;

    // Default constructor
    public BenchmarkResult() {}

    public BenchmarkResult(String scenario, double totalTimeSeconds, double initializationTimeSeconds,
                          double averageLatencyMs, double medianLatencyMs, double minLatencyMs,
                          double maxLatencyMs, double throughputTextsPerSecond, int successfulEmbeddings,
                          int failedEmbeddings, double[] perTextLatenciesMs, String executionProvider) {
        this.scenario = scenario;
        this.totalTimeSeconds = totalTimeSeconds;
        this.initializationTimeSeconds = initializationTimeSeconds;
        this.averageLatencyMs = averageLatencyMs;
        this.medianLatencyMs = medianLatencyMs;
        this.minLatencyMs = minLatencyMs;
        this.maxLatencyMs = maxLatencyMs;
        this.throughputTextsPerSecond = throughputTextsPerSecond;
        this.successfulEmbeddings = successfulEmbeddings;
        this.failedEmbeddings = failedEmbeddings;
        this.perTextLatenciesMs = perTextLatenciesMs;
        this.executionProvider = executionProvider;
    }

    public BenchmarkResult(String scenario, String error, String executionProvider) {
        this.scenario = scenario;
        this.error = error;
        this.executionProvider = executionProvider;
        this.totalTimeSeconds = 0;
        this.initializationTimeSeconds = 0;
        this.averageLatencyMs = 0;
        this.medianLatencyMs = 0;
        this.minLatencyMs = 0;
        this.maxLatencyMs = 0;
        this.throughputTextsPerSecond = 0;
        this.successfulEmbeddings = 0;
        this.failedEmbeddings = 0;
        this.perTextLatenciesMs = new double[0];
    }

    // Getters and setters
    public String getScenario() { return scenario; }
    public void setScenario(String scenario) { this.scenario = scenario; }

    public double getTotalTimeSeconds() { return totalTimeSeconds; }
    public void setTotalTimeSeconds(double totalTimeSeconds) { this.totalTimeSeconds = totalTimeSeconds; }

    public double getInitializationTimeSeconds() { return initializationTimeSeconds; }
    public void setInitializationTimeSeconds(double initializationTimeSeconds) { this.initializationTimeSeconds = initializationTimeSeconds; }

    public double getAverageLatencyMs() { return averageLatencyMs; }
    public void setAverageLatencyMs(double averageLatencyMs) { this.averageLatencyMs = averageLatencyMs; }

    public double getMedianLatencyMs() { return medianLatencyMs; }
    public void setMedianLatencyMs(double medianLatencyMs) { this.medianLatencyMs = medianLatencyMs; }

    public double getMinLatencyMs() { return minLatencyMs; }
    public void setMinLatencyMs(double minLatencyMs) { this.minLatencyMs = minLatencyMs; }

    public double getMaxLatencyMs() { return maxLatencyMs; }
    public void setMaxLatencyMs(double maxLatencyMs) { this.maxLatencyMs = maxLatencyMs; }

    public double getThroughputTextsPerSecond() { return throughputTextsPerSecond; }
    public void setThroughputTextsPerSecond(double throughputTextsPerSecond) { this.throughputTextsPerSecond = throughputTextsPerSecond; }

    public int getSuccessfulEmbeddings() { return successfulEmbeddings; }
    public void setSuccessfulEmbeddings(int successfulEmbeddings) { this.successfulEmbeddings = successfulEmbeddings; }

    public int getFailedEmbeddings() { return failedEmbeddings; }
    public void setFailedEmbeddings(int failedEmbeddings) { this.failedEmbeddings = failedEmbeddings; }

    public double[] getPerTextLatenciesMs() { return perTextLatenciesMs; }
    public void setPerTextLatenciesMs(double[] perTextLatenciesMs) { this.perTextLatenciesMs = perTextLatenciesMs; }

    public String getExecutionProvider() { return executionProvider; }
    public void setExecutionProvider(String executionProvider) { this.executionProvider = executionProvider; }

    public String getError() { return error; }
    public void setError(String error) { this.error = error; }

    public boolean hasError() { return error != null && !error.isEmpty(); }
}