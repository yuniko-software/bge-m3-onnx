using Microsoft.ML.OnnxRuntime;
using System.Globalization;
using System.Text.Json;

namespace BgeM3.Onnx.Tests;

/// <summary>
/// Reference embedding data structure from Python-generated JSON
/// </summary>
public record BgeM3ReferenceEmbedding(
    float[] DenseVecs,
    Dictionary<int, float> LexicalWeights,
    float[][] ColbertVecs
);

public sealed class BgeM3EmbeddingComparisonTests : IDisposable
{
    private readonly M3Embedder _cpuEmbedder;
    private readonly M3Embedder? _cudaEmbedder;
    private readonly Dictionary<string, BgeM3ReferenceEmbedding> _referenceEmbeddings;
    private readonly bool _cudaAvailable;

    public BgeM3EmbeddingComparisonTests()
    {
        var repoDir = Path.GetFullPath(Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "..", "..", "..", "..", "..", ".."));
        var onnxDir = Path.Combine(repoDir, "onnx");

        var tokenizerPath = Path.Combine(onnxDir, "bge_m3_tokenizer.onnx");
        var modelPath = Path.Combine(onnxDir, "bge_m3_model.onnx");
        var referenceFile = Path.Combine(onnxDir, "bge_m3_reference_embeddings.json");

        if (!File.Exists(tokenizerPath))
        {
            throw new FileNotFoundException($"Tokenizer file not found at {tokenizerPath}");
        }
        if (!File.Exists(modelPath))
        {
            throw new FileNotFoundException($"Model file not found at {modelPath}");
        }
        if (!File.Exists(referenceFile))
        {
            throw new FileNotFoundException($"Reference embeddings file not found at {referenceFile}. Please run the Python script to generate reference embeddings first.");
        }

        // Initialize CPU embedder (always available)
        _cpuEmbedder = M3EmbedderFactory.CreateCpuOptimized(tokenizerPath, modelPath);

        // Try to initialize CUDA embedder
        try
        {
            _cudaEmbedder = M3EmbedderFactory.CreateCudaOptimized(tokenizerPath, modelPath);
            _cudaAvailable = true;
        }
        catch (OnnxRuntimeException)
        {
            // CUDA not available, embedder will be null
            _cudaEmbedder = null;
            _cudaAvailable = false;
        }

        // Load reference embeddings
        var jsonContent = File.ReadAllText(referenceFile);
        var rawEmbeddings = JsonSerializer.Deserialize<Dictionary<string, JsonElement>>(jsonContent)!;

        _referenceEmbeddings = [];

        foreach (var kvp in rawEmbeddings)
        {
            var element = kvp.Value;

            var denseVecs = element.GetProperty("dense_vecs").EnumerateArray()
                .Select(x => (float)x.GetDouble()).ToArray();

            var lexicalWeights = new Dictionary<int, float>();
            foreach (var prop in element.GetProperty("lexical_weights").EnumerateObject())
            {
                lexicalWeights[int.Parse(prop.Name, CultureInfo.InvariantCulture)] = (float)prop.Value.GetDouble();
            }

            var colbertVecs = element.GetProperty("colbert_vecs").EnumerateArray()
                .Select(arr => arr.EnumerateArray().Select(x => (float)x.GetDouble()).ToArray())
                .ToArray();

            _referenceEmbeddings[kvp.Key] = new BgeM3ReferenceEmbedding(
                DenseVecs: denseVecs,
                LexicalWeights: lexicalWeights,
                ColbertVecs: colbertVecs
            );
        }
    }

    [Fact]
    public void CpuEmbeddings_ShouldMatchPythonEmbeddings()
    {
        ValidateEmbeddingsAgainstReference(_cpuEmbedder, "CPU");
    }

    [SkippableFact]
    public void CudaEmbeddings_ShouldMatchPythonEmbeddings()
    {
        // Skip test if CUDA is not available
        Skip.If(!_cudaAvailable, "CUDA provider is not available on this system");

        Assert.NotNull(_cudaEmbedder);
        ValidateEmbeddingsAgainstReference(_cudaEmbedder, "CUDA");
    }

    /// <summary>
    /// Helper method to validate embeddings against Python reference embeddings
    /// </summary>
    /// <param name="embedder">The embedder instance to test</param>
    /// <param name="providerName">Name of the execution provider for error messages</param>
    private void ValidateEmbeddingsAgainstReference(M3Embedder embedder, string providerName)
    {
        var failedComparisons = new List<string>();

        foreach (var entry in _referenceEmbeddings)
        {
            var text = entry.Key;
            var referenceEmbedding = entry.Value;

            try
            {
                var result = embedder.GenerateEmbeddings(text);

                // Verify we're using the expected provider
                var expectedProvider = providerName == "CPU" ? ExecutionProvider.CPU : ExecutionProvider.CUDA;
                Assert.Equal(expectedProvider, embedder.Config.ExecutionProvider);

                var denseSimilarity = CalculateCosineSimilarity(result.DenseEmbedding, referenceEmbedding.DenseVecs);
                if (denseSimilarity <= 0.9999)
                {
                    failedComparisons.Add($"{providerName} Dense similarity {denseSimilarity:F10} for '{text}'");
                }

                if (!AreSparseWeightsEqual(result.SparseWeights, referenceEmbedding.LexicalWeights))
                {
                    failedComparisons.Add($"{providerName} Sparse weights mismatch for '{text}'");
                }

                if (!AreColBertVectorsEqual(result.ColBertVectors, referenceEmbedding.ColbertVecs))
                {
                    failedComparisons.Add($"{providerName} ColBERT vectors mismatch for '{text}'");
                }
            }
            catch (Exception ex)
            {
                failedComparisons.Add($"{providerName} Exception for '{text}': {ex.Message}");
            }
        }

        if (failedComparisons.Count != 0)
        {
            var errorMessage = $"{providerName} embedding comparison failures:\n{string.Join("\n", failedComparisons)}";
            Assert.Fail(errorMessage);
        }
    }

    private static double CalculateCosineSimilarity(float[] vectorA, float[] vectorB)
    {
        if (vectorA.Length != vectorB.Length)
        {
            throw new ArgumentException("Vectors must be of the same length");
        }

        double dotProduct = 0;
        double normA = 0;
        double normB = 0;

        for (int i = 0; i < vectorA.Length; i++)
        {
            dotProduct += vectorA[i] * vectorB[i];
            normA += vectorA[i] * vectorA[i];
            normB += vectorB[i] * vectorB[i];
        }

        return dotProduct / (Math.Sqrt(normA) * Math.Sqrt(normB));
    }

    private static bool AreSparseWeightsEqual(Dictionary<int, float> csharpWeights, Dictionary<int, float> pythonWeights)
    {
        if (csharpWeights.Count != pythonWeights.Count)
        {
            return false;
        }

        foreach (var kvp in pythonWeights)
        {
            if (!csharpWeights.TryGetValue(kvp.Key, out float value))
            {
                return false;
            }

            var difference = Math.Abs(kvp.Value - value);
            if (difference >= 1e-3f)
            {
                return false;
            }
        }

        return true;
    }

    private static bool AreColBertVectorsEqual(float[][] csharpVectors, float[][] pythonVectors)
    {
        if (csharpVectors.Length != pythonVectors.Length)
        {
            return false;
        }

        for (int i = 0; i < pythonVectors.Length; i++)
        {
            if (csharpVectors[i].Length != pythonVectors[i].Length)
            {
                return false;
            }

            var similarity = CalculateCosineSimilarity(csharpVectors[i], pythonVectors[i]);
            if (similarity <= 0.9999)
            {
                return false;
            }
        }

        return true;
    }

    public void Dispose()
    {
        _cpuEmbedder?.Dispose();
        _cudaEmbedder?.Dispose();
    }
}