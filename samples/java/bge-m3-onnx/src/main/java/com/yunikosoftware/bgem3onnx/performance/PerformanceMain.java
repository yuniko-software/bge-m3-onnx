package com.yunikosoftware.bgem3onnx.performance;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.PropertyNamingStrategies;

import java.io.File;
import java.io.FileNotFoundException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.time.Instant;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Main class for running BGE-M3 Java performance benchmarks
 */
public class PerformanceMain {

    public static void main(String[] args) {
        try {
            // Find repository root and paths
            Path repoRoot = findRepositoryRoot();
            Path performanceDataDir = repoRoot.resolve("samples").resolve("performance_data");
            Path onnxDir = repoRoot.resolve("onnx");

            Path tokenizerPath = onnxDir.resolve("bge_m3_tokenizer.onnx");
            Path modelPath = onnxDir.resolve("bge_m3_model.onnx");

            System.out.println("=" + "=".repeat(59));
            System.out.println("BGE-M3 Java Performance Benchmark");
            System.out.println("=" + "=".repeat(59));
            System.out.println("Performance data: " + performanceDataDir);
            System.out.println("Tokenizer: " + tokenizerPath);
            System.out.println("Model: " + modelPath);

            // Verify required files exist
            String[] requiredFiles = {tokenizerPath.toString(), modelPath.toString()};
            for (String path : requiredFiles) {
                if (!new File(path).exists()) {
                    System.err.println("ERROR: Required file not found: " + path);
                    System.exit(1);
                }
            }

            // Load test dataset
            List<TestText> texts;
            try {
                System.out.println("\nLoading test dataset...");
                texts = loadTestDataset(performanceDataDir.toString());
                System.out.println("Loaded " + texts.size() + " texts");
            } catch (Exception ex) {
                System.err.println("ERROR: Failed to load test dataset: " + ex.getMessage());
                System.exit(1);
                return;
            }

            // Initialize benchmark runner
            BenchmarkRunner runner = new BenchmarkRunner(tokenizerPath.toString(), modelPath.toString());

            // Prepare results
            PerformanceReport.TestInfo testInfo = new PerformanceReport.TestInfo(
                Instant.now().toString(),
                texts.size(),
                texts.stream().limit(5).map(TestText::getText).toArray(String[]::new)
            );

            Map<String, BenchmarkResult> scenarios = new HashMap<>();
            PerformanceReport results = new PerformanceReport(testInfo, scenarios);

            System.out.println("\nRunning benchmarks on " + texts.size() + " texts...");

            // Benchmark 1: ONNX CPU
            try {
                System.out.println("\n" + "-".repeat(40));
                BenchmarkResult cpuResult = runner.benchmarkCpu(texts);
                scenarios.put("onnx_cpu", cpuResult);

                if (!cpuResult.hasError()) {
                    System.out.printf("ONNX CPU: %.1fms avg, %.1f texts/sec%n", 
                                    cpuResult.getAverageLatencyMs(), 
                                    cpuResult.getThroughputTextsPerSecond());
                } else {
                    System.out.println("ONNX CPU: ERROR - " + cpuResult.getError());
                }
            } catch (Exception ex) {
                System.err.println("ONNX CPU benchmark failed: " + ex.getMessage());
                scenarios.put("onnx_cpu", new BenchmarkResult("onnx_cpu", ex.getMessage(), "CPU"));
            }

            // Benchmark 2: ONNX CUDA
            try {
                System.out.println("\n" + "-".repeat(40));
                BenchmarkResult cudaResult = runner.benchmarkCuda(texts);
                scenarios.put("onnx_cuda", cudaResult);

                if (!cudaResult.hasError()) {
                    System.out.printf("ONNX CUDA: %.1fms avg, %.1f texts/sec%n", 
                                    cudaResult.getAverageLatencyMs(), 
                                    cudaResult.getThroughputTextsPerSecond());
                } else {
                    System.out.println("ONNX CUDA: ERROR - " + cudaResult.getError());
                }
            } catch (Exception ex) {
                System.err.println("ONNX CUDA benchmark failed: " + ex.getMessage());
                scenarios.put("onnx_cuda", new BenchmarkResult("onnx_cuda", ex.getMessage(), "CUDA"));
            }

            // Save results
            File outputFile = onnxDir.resolve("performance_java.json").toFile();
            outputFile.getParentFile().mkdirs();

            ObjectMapper objectMapper = new ObjectMapper();
            objectMapper.setPropertyNamingStrategy(PropertyNamingStrategies.SNAKE_CASE);
            objectMapper.writerWithDefaultPrettyPrinter().writeValue(outputFile, results);

            // Print summary
            System.out.println("\n" + "=".repeat(60));
            System.out.println("Benchmark Summary");
            System.out.println("=".repeat(60));

            if (!scenarios.isEmpty()) {
                System.out.printf("%-20s %-18s %-18s %s%n", "Scenario", "Avg Latency (ms)", "Throughput (t/s)", "Status");
                System.out.println("-".repeat(70));

                for (Map.Entry<String, BenchmarkResult> entry : scenarios.entrySet()) {
                    String scenarioName = entry.getKey();
                    BenchmarkResult scenarioData = entry.getValue();

                    if (scenarioData.hasError()) {
                        System.out.printf("%-20s %-18s %-18s %s%n", scenarioName, "ERROR", "ERROR", scenarioData.getError());
                    } else {
                        System.out.printf("%-20s %-18.1f %-18.1f %s%n", 
                                        scenarioName, 
                                        scenarioData.getAverageLatencyMs(), 
                                        scenarioData.getThroughputTextsPerSecond(), 
                                        "Success");
                    }
                }
            }

            System.out.println("\nResults saved to: " + outputFile.getAbsolutePath());
            System.out.println("Java performance benchmark completed!");

        } catch (Exception e) {
            System.err.println("Error: " + e.getMessage());
            e.printStackTrace();
            System.exit(1);
        }
    }

    /**
     * Load test dataset from JSON file
     */
    private static List<TestText> loadTestDataset(String dataDir) throws Exception {
        File datasetFile = new File(dataDir, "test_texts.json");

        if (!datasetFile.exists()) {
            throw new FileNotFoundException("Test dataset not found at " + datasetFile.getAbsolutePath() + 
                                          ". Please run the dataset generator first.");
        }

        ObjectMapper objectMapper = new ObjectMapper();
        return objectMapper.readValue(datasetFile, new TypeReference<List<TestText>>() {});
    }

    /**
     * Find repository root by looking for 'onnx' directory
     */
    private static Path findRepositoryRoot() throws FileNotFoundException {
        Path currentDir = Paths.get("").toAbsolutePath();

        for (int i = 0; i < 10; i++) {
            Path onnxDir = currentDir.resolve("onnx");

            if (onnxDir.toFile().exists() && onnxDir.toFile().isDirectory()) {
                return currentDir;
            }

            Path parent = currentDir.getParent();
            if (parent == null) {
                break;
            }
            currentDir = parent;
        }

        throw new FileNotFoundException("Could not locate repository root with 'onnx' directory");
    }
}