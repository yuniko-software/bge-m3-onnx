using BgeM3.Onnx;
using System.Globalization;

// Define paths relative to the project
var repoDir = Path.GetFullPath(Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "..", "..", "..", "..", "..", "..", ".."));
var onnxDir = Path.Combine(repoDir, "onnx");

var tokenizerPath = Path.Combine(onnxDir, "bge_m3_tokenizer.onnx");
var modelPath = Path.Combine(onnxDir, "bge_m3_model.onnx");

// Sample text to test with
string text = "A test text! Texto de prueba! Текст для теста! 測試文字! Testtext! Testez le texte! Сынақ мәтіні! Тестни текст! परीक्षण पाठ! Kiểm tra văn bản!";

Console.WriteLine("===== BGE-M3 ONNX Embedding Research =====");
Console.WriteLine($"Using tokenizer: {tokenizerPath}");
Console.WriteLine($"Using model: {modelPath}");

// Create the embedding generator with all 3 vector types enabled
using var embeddingGenerator = new M3Embedder(tokenizerPath, modelPath);

var embeddings = embeddingGenerator.GenerateEmbeddings(text);

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
Console.WriteLine($"Vector dimension: {colbertVectors[0].Length}");

// Print first vector
Console.WriteLine("First vector (first 10 values):");
Console.WriteLine($"[{string.Join(", ", colbertVectors[0].Take(10).Select(v => v.ToString("F6", CultureInfo.InvariantCulture)))}]");

Console.WriteLine("\n===== SUCCESS =====");
Console.WriteLine("All embedding types generated successfully!");
