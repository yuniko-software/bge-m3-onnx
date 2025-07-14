package com.yunikosoftware.bgem3onnx.performance;

import com.yunikosoftware.bgem3onnx.ExecutionProvider;
import com.yunikosoftware.bgem3onnx.M3Embedder;
import com.yunikosoftware.bgem3onnx.M3EmbedderFactory;

import java.util.Arrays;
import java.util.List;

/**
 * Runs performance benchmarks for BGE-M3 ONNX embedder
 */
public class BenchmarkRunner {
    private final String tokenizerPath;
    private final String modelPath;

    public BenchmarkRunner(String tokenizerPath, String modelPath) {
        this.tokenizerPath = tokenizerPath;
        this.modelPath = modelPath;
    }

    /**
     * Core benchmark logic shared across different execution providers
     */
    private BenchmarkResult runBenchmarkCore(List<TestText> texts, String scenarioName, 
                                           M3Embedder embedder, double initTime, 
                                           ExecutionProvider executionProvider) {
        System.out.println("Benchmarking " + scenarioName + "...");

        // Warm up
        try {
            embedder.generateEmbeddings("warm up text");
        } catch (Exception e) {
            System.err.println("Warmup failed: " + e.getMessage());
        }

        // Run benchmark
        long totalStartTime = System.currentTimeMillis();
        double[] latencies = new double[texts.size()];
        int successCount = 0;
        int failureCount = 0;

        for (int i = 0; i < texts.size(); i++) {
            TestText textItem = texts.get(i);
            long textStartTime = System.currentTimeMillis();

            try {
                embedder.generateEmbeddings(textItem.getText());
                successCount++;
            } catch (Exception ex) {
                System.err.println("Error processing text " + i + ": " + ex.getMessage());
                failureCount++;
            }

            double latencyMs = System.currentTimeMillis() - textStartTime;
            latencies[i] = latencyMs;

            if ((i + 1) % 100 == 0) {
                System.out.println("Processed " + (i + 1) + "/" + texts.size() + " texts");
            }
        }

        double totalTimeSeconds = (System.currentTimeMillis() - totalStartTime) / 1000.0;

        return new BenchmarkResult(
            scenarioName,
            totalTimeSeconds,
            initTime,
            Arrays.stream(latencies).average().orElse(0),
            getMedian(latencies),
            Arrays.stream(latencies).min().orElse(0),
            Arrays.stream(latencies).max().orElse(0),
            texts.size() / totalTimeSeconds,
            successCount,
            failureCount,
            latencies,
            executionProvider.toString()
        );
    }

    /**
     * Benchmark CPU execution provider
     */
    public BenchmarkResult benchmarkCpu(List<TestText> texts) {
        try {
            // Initialize model
            long initStartTime = System.currentTimeMillis();
            M3Embedder embedder = M3EmbedderFactory.createCpuOptimized(tokenizerPath, modelPath);
            double initTime = (System.currentTimeMillis() - initStartTime) / 1000.0;

            try {
                return runBenchmarkCore(texts, "onnx_cpu", embedder, initTime, ExecutionProvider.CPU);
            } finally {
                embedder.close();
            }
        } catch (Exception ex) {
            return new BenchmarkResult("onnx_cpu", ex.getMessage(), ExecutionProvider.CPU.toString());
        }
    }

    /**
     * Benchmark CUDA execution provider
     */
    public BenchmarkResult benchmarkCuda(List<TestText> texts) {
        try {
            // Initialize model
            long initStartTime = System.currentTimeMillis();
            M3Embedder embedder = M3EmbedderFactory.createCudaOptimized(tokenizerPath, modelPath);
            double initTime = (System.currentTimeMillis() - initStartTime) / 1000.0;

            try {
                // Verify CUDA is being used
                if (embedder.getConfig().getExecutionProvider() != ExecutionProvider.CUDA) {
                    System.out.println("CUDA provider not available, skipping CUDA benchmark");
                    return new BenchmarkResult("onnx_cuda", "CUDA provider not available", 
                                             embedder.getConfig().getExecutionProvider().toString());
                }

                return runBenchmarkCore(texts, "onnx_cuda", embedder, initTime, ExecutionProvider.CUDA);
            } finally {
                embedder.close();
            }
        } catch (Exception ex) {
            System.err.println("CUDA benchmark failed: " + ex.getMessage());
            return new BenchmarkResult("onnx_cuda", ex.getMessage(), "UNKNOWN");
        }
    }

    /**
     * Calculate median of an array of doubles
     */
    private static double getMedian(double[] values) {
        double[] sorted = values.clone();
        Arrays.sort(sorted);
        int length = sorted.length;
        
        if (length % 2 == 0) {
            return (sorted[length / 2 - 1] + sorted[length / 2]) / 2.0;
        } else {
            return sorted[length / 2];
        }
    }
} 