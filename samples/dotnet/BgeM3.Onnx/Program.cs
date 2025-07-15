using BgeM3.Onnx;
using System.Globalization;

var tokenizerPath = RepositoryUtils.GetTokenizerPath();
var modelPath = RepositoryUtils.GetModelPath();

// Sample text to test with
string text = "A test text! Texto de prueba! Текст для теста! 測試文字! Testtext! Testez le texte! Сынақ мәтіні! Тестни текст! परीक्षण पाठ! Kiểm tra văn bản!";

Console.WriteLine("===== BGE-M3 ONNX Multi-Provider Test =====");
Console.WriteLine($"Tokenizer: {Path.GetFileName(tokenizerPath)}");
Console.WriteLine($"Model: {Path.GetFileName(modelPath)}");

// Test CPU provider
Console.WriteLine("\n===== CPU PROVIDER =====");
TestProvider("CPU", () => M3EmbedderFactory.CreateCpuOptimized(tokenizerPath, modelPath), text);

// Test CUDA provider
Console.WriteLine("\n===== CUDA PROVIDER =====");
try
{
    TestProvider("CUDA", () => M3EmbedderFactory.CreateCudaOptimized(tokenizerPath, modelPath), text);
}
catch (Exception ex)
{
    Console.WriteLine($"CUDA not available: {ex.Message}");
}

Console.WriteLine("\n===== TEST COMPLETE =====");

static void TestProvider(string providerName, Func<M3Embedder> embedderFactory, string testText)
{
    using var embedder = embedderFactory();

    Console.WriteLine($"Provider: {embedder.Config.ExecutionProvider}");

    // Generate embeddings
    var stopwatch = System.Diagnostics.Stopwatch.StartNew();
    var embeddings = embedder.GenerateEmbeddings(testText);
    stopwatch.Stop();

    Console.WriteLine($"Inference time: {stopwatch.ElapsedMilliseconds}ms");

    // Print dense embedding information
    Console.WriteLine("\n=== DENSE EMBEDDING ===");
    var denseEmbedding = embeddings.DenseEmbedding;
    Console.WriteLine($"Length: {denseEmbedding.Length}");
    Console.WriteLine($"First 10 values: [{string.Join(", ", denseEmbedding.Take(10).Select(v => v.ToString("F6", CultureInfo.InvariantCulture)))}]");

    // Print sparse weights information
    Console.WriteLine("\n=== SPARSE WEIGHTS ===");
    var sparseWeights = embeddings.SparseWeights;
    Console.WriteLine($"Non-zero tokens: {sparseWeights.Count}");

    // Top tokens
    var topWeights = sparseWeights
        .OrderByDescending(kv => kv.Value)
        .Take(5)
        .ToList();

    Console.WriteLine("Top 5 tokens:");
    foreach (var (tokenId, weight) in topWeights)
    {
        Console.WriteLine($"  {tokenId}: {weight:F6}");
    }

    // Print ColBERT vectors information
    Console.WriteLine("\n=== COLBERT VECTORS ===");
    var colbertVectors = embeddings.ColBertVectors;
    Console.WriteLine($"Token count: {colbertVectors.Length}");
    if (colbertVectors.Length > 0)
    {
        Console.WriteLine($"Vector dimension: {colbertVectors[0].Length}");
        Console.WriteLine("First vector (first 10 values):");
        Console.WriteLine($"[{string.Join(", ", colbertVectors[0].Take(10).Select(v => v.ToString("F6", CultureInfo.InvariantCulture)))}]");
    }

    Console.WriteLine($"\n✓ {providerName} completed successfully!");
}