using Microsoft.ML.OnnxRuntime;
using System.Diagnostics;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace BgeM3.Onnx.Performance;

public record TestText(
    [property: JsonPropertyName("id")] int Id,
    [property: JsonPropertyName("text")] string Text,
    [property: JsonPropertyName("length_category")] string LengthCategory,
    [property: JsonPropertyName("language")] string Language,
    [property: JsonPropertyName("domain")] string Domain,
    [property: JsonPropertyName("word_count")] int WordCount,
    [property: JsonPropertyName("char_count")] int CharCount,
    [property: JsonPropertyName("source")] string Source
);

public record BenchmarkResult(
    [property: JsonPropertyName("scenario")] string Scenario,
    [property: JsonPropertyName("total_time_seconds")] double TotalTimeSeconds,
    [property: JsonPropertyName("initialization_time_seconds")] double InitializationTimeSeconds,
    [property: JsonPropertyName("average_latency_ms")] double AverageLatencyMs,
    [property: JsonPropertyName("median_latency_ms")] double MedianLatencyMs,
    [property: JsonPropertyName("min_latency_ms")] double MinLatencyMs,
    [property: JsonPropertyName("max_latency_ms")] double MaxLatencyMs,
    [property: JsonPropertyName("throughput_texts_per_second")] double ThroughputTextsPerSecond,
    [property: JsonPropertyName("successful_embeddings")] int SuccessfulEmbeddings,
    [property: JsonPropertyName("failed_embeddings")] int FailedEmbeddings,
    [property: JsonPropertyName("per_text_latencies_ms")] double[] PerTextLatenciesMs,
    [property: JsonPropertyName("execution_provider")] string ExecutionProvider,
    [property: JsonPropertyName("error")] string? Error = null
);

public record PerformanceReport(
    [property: JsonPropertyName("test_info")] TestInfo TestInfo,
    [property: JsonPropertyName("scenarios")] Dictionary<string, BenchmarkResult> Scenarios
);

public record TestInfo(
    [property: JsonPropertyName("timestamp")] string Timestamp,
    [property: JsonPropertyName("test_dataset_size")] int TestDatasetSize,
    [property: JsonPropertyName("sample_texts")] string[] SampleTexts
);

public class BenchmarkRunner(string tokenizerPath, string modelPath)
{
    private readonly string _tokenizerPath = tokenizerPath;
    private readonly string _modelPath = modelPath;

    private static BenchmarkResult RunBenchmarkCore(List<TestText> texts, M3Embedder embedder,
        string scenarioName, double initTime, ExecutionProvider executionProvider)
    {
        Console.WriteLine($"Benchmarking {scenarioName}...");

        // Warm up
        embedder.GenerateEmbeddings("warm up text");

        // Run benchmark
        var totalStopwatch = Stopwatch.StartNew();
        var latencies = new List<double>();
        var successCount = 0;
        var failureCount = 0;

        foreach (var (text, index) in texts.Select((t, i) => (t, i)))
        {
            var textStopwatch = Stopwatch.StartNew();

            try
            {
                var result = embedder.GenerateEmbeddings(text.Text);
                successCount++;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error processing text {index}: {ex.Message}");
                failureCount++;
            }

            var latencyMs = textStopwatch.Elapsed.TotalMilliseconds;
            latencies.Add(latencyMs);

            if ((index + 1) % 100 == 0)
                Console.WriteLine($"Processed {index + 1}/{texts.Count} texts");
        }

        var totalTime = totalStopwatch.Elapsed.TotalSeconds;

        return new BenchmarkResult(
            Scenario: scenarioName,
            TotalTimeSeconds: totalTime,
            InitializationTimeSeconds: initTime,
            AverageLatencyMs: latencies.Average(),
            MedianLatencyMs: GetMedian(latencies),
            MinLatencyMs: latencies.Min(),
            MaxLatencyMs: latencies.Max(),
            ThroughputTextsPerSecond: texts.Count / totalTime,
            SuccessfulEmbeddings: successCount,
            FailedEmbeddings: failureCount,
            PerTextLatenciesMs: [.. latencies],
            ExecutionProvider: executionProvider.ToString()
        );
    }

    public BenchmarkResult BenchmarkCpu(List<TestText> texts)
    {
        try
        {
            // Initialize model
            var initStopwatch = Stopwatch.StartNew();
            using var embedder = M3EmbedderFactory.CreateCpuOptimized(_tokenizerPath, _modelPath);
            var initTime = initStopwatch.Elapsed.TotalSeconds;

            return RunBenchmarkCore(texts, embedder, "onnx_cpu", initTime, ExecutionProvider.CPU);
        }
        catch (Exception ex)
        {
            return new BenchmarkResult(
                Scenario: "onnx_cpu",
                TotalTimeSeconds: 0,
                InitializationTimeSeconds: 0,
                AverageLatencyMs: 0,
                MedianLatencyMs: 0,
                MinLatencyMs: 0,
                MaxLatencyMs: 0,
                ThroughputTextsPerSecond: 0,
                SuccessfulEmbeddings: 0,
                FailedEmbeddings: texts.Count,
                PerTextLatenciesMs: [],
                ExecutionProvider: ExecutionProvider.CPU.ToString(),
                Error: ex.Message
            );
        }
    }

    public BenchmarkResult? BenchmarkCuda(List<TestText> texts)
    {
        try
        {
            // Initialize model
            var initStopwatch = Stopwatch.StartNew();
            using var embedder = M3EmbedderFactory.CreateCudaOptimized(_tokenizerPath, _modelPath);
            var initTime = initStopwatch.Elapsed.TotalSeconds;

            // Verify CUDA is being used
            if (embedder.Config.ExecutionProvider != ExecutionProvider.CUDA)
            {
                Console.WriteLine("CUDA provider not available, skipping CUDA benchmark");
                return new BenchmarkResult(
                    Scenario: "onnx_cuda",
                    TotalTimeSeconds: 0,
                    InitializationTimeSeconds: 0,
                    AverageLatencyMs: 0,
                    MedianLatencyMs: 0,
                    MinLatencyMs: 0,
                    MaxLatencyMs: 0,
                    ThroughputTextsPerSecond: 0,
                    SuccessfulEmbeddings: 0,
                    FailedEmbeddings: 0,
                    PerTextLatenciesMs: [],
                    ExecutionProvider: embedder.Config.ExecutionProvider.ToString(),
                    Error: "CUDA provider not available"
                );
            }

            return RunBenchmarkCore(texts, embedder, "onnx_cuda", initTime, ExecutionProvider.CUDA);
        }
        catch (OnnxRuntimeException ex)
        {
            Console.WriteLine($"CUDA benchmark failed: {ex.Message}");
            return new BenchmarkResult(
                Scenario: "onnx_cuda",
                TotalTimeSeconds: 0,
                InitializationTimeSeconds: 0,
                AverageLatencyMs: 0,
                MedianLatencyMs: 0,
                MinLatencyMs: 0,
                MaxLatencyMs: 0,
                ThroughputTextsPerSecond: 0,
                SuccessfulEmbeddings: 0,
                FailedEmbeddings: texts.Count,
                PerTextLatenciesMs: [],
                ExecutionProvider: "UNKNOWN",
                Error: ex.Message
            );
        }
        catch (Exception ex)
        {
            Console.WriteLine($"CUDA benchmark failed with unexpected error: {ex.Message}");
            return null;
        }
    }

    private static double GetMedian(List<double> values)
    {
        var sorted = values.OrderBy(x => x).ToList();
        var count = sorted.Count;
        return count % 2 == 0
            ? (sorted[count / 2 - 1] + sorted[count / 2]) / 2.0
            : sorted[count / 2];
    }
}

public class Program
{
    public static async Task<int> Main()
    {
        var performanceDataDir = RepositoryUtils.GetPerformanceDataDirectory();
        var tokenizerPath = RepositoryUtils.GetTokenizerPath();
        var modelPath = RepositoryUtils.GetModelPath();

        Console.WriteLine("=" + new string('=', 59));
        Console.WriteLine("BGE-M3 C# Performance Benchmark");
        Console.WriteLine("=" + new string('=', 59));
        Console.WriteLine($"Performance data: {performanceDataDir}");
        Console.WriteLine($"Tokenizer: {tokenizerPath}");
        Console.WriteLine($"Model: {modelPath}");

        // Verify required files exist
        var requiredFiles = new[] { tokenizerPath, modelPath };
        foreach (var path in requiredFiles)
        {
            if (!File.Exists(path))
            {
                Console.WriteLine($"ERROR: Required file not found: {path}");
                return 1;
            }
        }

        // Load test dataset
        List<TestText> texts;
        try
        {
            Console.WriteLine("\nLoading test dataset...");
            texts = await LoadTestDataset(performanceDataDir);
            Console.WriteLine($"Loaded {texts.Count} texts");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"ERROR: Failed to load test dataset: {ex.Message}");
            return 1;
        }

        // Initialize benchmark runner
        var runner = new BenchmarkRunner(tokenizerPath, modelPath);

        // Prepare results
        var results = new PerformanceReport(
            TestInfo: new TestInfo(
                Timestamp: DateTime.UtcNow.ToString("O"),
                TestDatasetSize: texts.Count,
                SampleTexts: [.. texts.Take(5).Select(t => t.Text)]
            ),
            Scenarios: []
        );

        Console.WriteLine($"\nRunning benchmarks on {texts.Count} texts...");

        // Benchmark 1: ONNX CPU
        try
        {
            Console.WriteLine("\n" + new string('-', 40));
            var cpuResult = runner.BenchmarkCpu(texts);
            results.Scenarios["onnx_cpu"] = cpuResult;

            if (cpuResult.Error == null)
            {
                Console.WriteLine($"ONNX CPU: {cpuResult.AverageLatencyMs:F1}ms avg, " +
                                $"{cpuResult.ThroughputTextsPerSecond:F1} texts/sec");
            }
            else
            {
                Console.WriteLine($"ONNX CPU: ERROR - {cpuResult.Error}");
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"ONNX CPU benchmark failed: {ex.Message}");
            results.Scenarios["onnx_cpu"] = new BenchmarkResult(
                Scenario: "onnx_cpu",
                TotalTimeSeconds: 0,
                InitializationTimeSeconds: 0,
                AverageLatencyMs: 0,
                MedianLatencyMs: 0,
                MinLatencyMs: 0,
                MaxLatencyMs: 0,
                ThroughputTextsPerSecond: 0,
                SuccessfulEmbeddings: 0,
                FailedEmbeddings: texts.Count,
                PerTextLatenciesMs: [],
                ExecutionProvider: "CPU",
                Error: ex.Message
            );
        }

        // Benchmark 2: ONNX CUDA
        try
        {
            Console.WriteLine("\n" + new string('-', 40));
            var cudaResult = runner.BenchmarkCuda(texts);

            if (cudaResult != null)
            {
                results.Scenarios["onnx_cuda"] = cudaResult;

                if (cudaResult.Error == null)
                {
                    Console.WriteLine($"ONNX CUDA: {cudaResult.AverageLatencyMs:F1}ms avg, " +
                                    $"{cudaResult.ThroughputTextsPerSecond:F1} texts/sec");
                }
                else
                {
                    Console.WriteLine($"ONNX CUDA: ERROR - {cudaResult.Error}");
                }
            }
            else
            {
                Console.WriteLine("ONNX CUDA: Failed to initialize");
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"ONNX CUDA benchmark failed: {ex.Message}");
            results.Scenarios["onnx_cuda"] = new BenchmarkResult(
                Scenario: "onnx_cuda",
                TotalTimeSeconds: 0,
                InitializationTimeSeconds: 0,
                AverageLatencyMs: 0,
                MedianLatencyMs: 0,
                MinLatencyMs: 0,
                MaxLatencyMs: 0,
                ThroughputTextsPerSecond: 0,
                SuccessfulEmbeddings: 0,
                FailedEmbeddings: texts.Count,
                PerTextLatenciesMs: [],
                ExecutionProvider: "CUDA",
                Error: ex.Message
            );
        }

        // Save results
        var outputPath = Path.Combine(RepositoryUtils.GetOnnxDirectory(), "performance_dotnet.json");

        var jsonOptions = new JsonSerializerOptions
        {
            WriteIndented = true,
            PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower
        };

        var jsonString = JsonSerializer.Serialize(results, jsonOptions);
        await File.WriteAllTextAsync(outputPath, jsonString);

        // Print summary
        Console.WriteLine("\n" + new string('=', 60));
        Console.WriteLine("Benchmark Summary");
        Console.WriteLine(new string('=', 60));

        if (results.Scenarios.Count != 0)
        {
            Console.WriteLine($"{"Scenario",-20} {"Avg Latency (ms)",-18} {"Throughput (t/s)",-18} {"Status"}");
            Console.WriteLine(new string('-', 70));

            foreach (var (scenarioName, scenarioData) in results.Scenarios)
            {
                if (!string.IsNullOrEmpty(scenarioData.Error))
                {
                    Console.WriteLine($"{scenarioName,-20} {"ERROR",-18} {"ERROR",-18} {scenarioData.Error}");
                }
                else
                {
                    Console.WriteLine($"{scenarioName,-20} {scenarioData.AverageLatencyMs,-18:F1} " +
                                    $"{scenarioData.ThroughputTextsPerSecond,-18:F1} {"Success"}");
                }
            }
        }

        Console.WriteLine($"\nResults saved to: {outputPath}");
        Console.WriteLine("C# performance benchmark completed!");

        return 0;
    }

    private static async Task<List<TestText>> LoadTestDataset(string dataDir)
    {
        var datasetPath = Path.Combine(dataDir, "test_texts.json");

        if (!File.Exists(datasetPath))
        {
            throw new FileNotFoundException($"Test dataset not found at {datasetPath}. Please run the dataset generator first.");
        }

        var jsonString = await File.ReadAllTextAsync(datasetPath);

        var dataset = JsonSerializer.Deserialize<List<TestText>>(jsonString) ?? throw new InvalidOperationException("Failed to deserialize test dataset");

        return dataset;
    }
}