using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace BgeM3.Onnx;

/// <summary>
/// Provides functionality to generate embeddings using ONNX bge-m3 model with multi-provider support
/// </summary>
public class M3Embedder : IDisposable
{
    private readonly InferenceSession _tokenizerSession;
    private readonly InferenceSession _modelSession;
    private readonly HashSet<int> _specialTokenIds = [0, 1, 2, 3]; // [PAD], [UNK], [CLS], [SEP]
    private readonly M3EmbedderConfig _config;
    private bool _disposed;

    /// <summary>
    /// Gets the configuration used by this embedder
    /// </summary>
    public M3EmbedderConfig Config => _config;

    /// <summary>
    /// Initializes a new instance of the M3Embedder class with default CPU provider
    /// </summary>
    /// <param name="tokenizerPath">Path to the ONNX tokenizer model</param>
    /// <param name="modelPath">Path to the ONNX embedding model</param>
    public M3Embedder(string tokenizerPath, string modelPath) : this(tokenizerPath, modelPath, new M3EmbedderConfig()) { }

    /// <summary>
    /// Initializes a new instance of the M3Embedder class with specified configuration
    /// </summary>
    /// <param name="tokenizerPath">Path to the ONNX tokenizer model</param>
    /// <param name="modelPath">Path to the ONNX embedding model</param>
    /// <param name="config">Configuration for execution providers and other options</param>
    public M3Embedder(string tokenizerPath, string modelPath, M3EmbedderConfig config)
    {
        _config = config;

        // Initialize tokenizer session with ONNX Extensions (CPU-only)
        var tokenizerOptions = CreateSessionOptions(forTokenizer: true);
        tokenizerOptions.RegisterOrtExtensions();
        _tokenizerSession = new InferenceSession(tokenizerPath, tokenizerOptions);

        // Initialize model session with specified execution provider
        var modelOptions = CreateSessionOptions(forTokenizer: false);
        _modelSession = new InferenceSession(modelPath, modelOptions);
    }

    /// <summary>
    /// Creates session options with appropriate execution providers
    /// </summary>
    private SessionOptions CreateSessionOptions(bool forTokenizer)
    {
        var sessionOptions = new SessionOptions
        {
            EnableMemoryPattern = _config.EnableMemoryPattern,
            EnableCpuMemArena = _config.EnableCpuMemArena,
            LogSeverityLevel = _config.LogSeverityLevel
        };

        // For tokenizer, we use CPU since it involves ONNX Extensions
        // and string processing which may not be optimized for GPU
        if (forTokenizer)
        {
            return sessionOptions;
        }

        // For the main model, apply the requested execution providers
        var providers = GetProviderList();

        foreach (var provider in providers)
        {
            switch (provider)
            {
                case ExecutionProvider.CUDA:
                    sessionOptions.AppendExecutionProvider_CUDA(_config.CudaDeviceId);
                    break;

                case ExecutionProvider.CPU:
                    // CPU is always available and added by default
                    break;

                default:
                    throw new ArgumentException($"Unsupported execution provider: {provider}");
            }
        }

        return sessionOptions;
    }

    /// <summary>
    /// Gets the list of execution providers to try in order
    /// </summary>
    private List<ExecutionProvider> GetProviderList()
    {
        var providers = new List<ExecutionProvider> { _config.ExecutionProvider };

        // Add fallback providers if they're different from the primary
        foreach (var fallback in _config.FallbackProviders)
        {
            if (fallback != _config.ExecutionProvider && !providers.Contains(fallback))
            {
                providers.Add(fallback);
            }
        }

        // Always ensure CPU is available as the final fallback
        if (!providers.Contains(ExecutionProvider.CPU))
        {
            providers.Add(ExecutionProvider.CPU);
        }

        return providers;
    }

    /// <summary>
    /// Generates all embeddings (dense, sparse, ColBERT) for the input text
    /// </summary>
    /// <param name="text">The input text</param>
    /// <returns>The full embedding output containing all vector types</returns>
    public M3EmbeddingOutput GenerateEmbeddings(string text)
    {
        // Create input tensor for tokenizer
        var stringTensor = new DenseTensor<string>([1]);
        stringTensor[0] = text;

        // Create input for tokenizer
        var tokenizerInputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("inputs", stringTensor)
        };

        // Run tokenizer
        using var tokenizerResults = _tokenizerSession.Run(tokenizerInputs);
        var tokenizerResultsList = tokenizerResults.ToList();

        // Extract tokens and token_indices (order: tokens, instance_indices, token_indices)
        var tokens = tokenizerResultsList[0].AsTensor<int>().ToArray();
        var tokenIndices = tokenizerResultsList[2].AsTensor<int>().ToArray();

        // Convert to input_ids by sorting tokens based on token_indices
        var tokenPairs = tokens.Zip(tokenIndices, (t, i) => (token: t, index: i))
            .OrderBy(p => p.index)
            .Select(p => p.token)
            .ToArray();

        // Create input_ids tensor with shape [1, tokenPairs.Length]
        var inputIdsTensor = new DenseTensor<long>([1, tokenPairs.Length]);
        for (int i = 0; i < tokenPairs.Length; i++)
        {
            inputIdsTensor[0, i] = tokenPairs[i];
        }

        // Create attention_mask as all 1s with same shape as input_ids
        var attentionMaskTensor = new DenseTensor<long>([1, tokenPairs.Length]);
        for (int i = 0; i < tokenPairs.Length; i++)
        {
            attentionMaskTensor[0, i] = 1;
        }

        // Run the model with the prepared inputs
        var modelInputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("input_ids", inputIdsTensor),
            NamedOnnxValue.CreateFromTensor("attention_mask", attentionMaskTensor)
        };

        using var modelResults = _modelSession.Run(modelInputs);
        var modelResultsList = modelResults.ToList();

        // Process outputs
        var output = new M3EmbeddingOutput(
            DenseEmbedding: [.. modelResultsList[0].AsTensor<float>()],
            SparseWeights: ExtractSparseWeights(modelResultsList[1], tokenPairs, [.. attentionMaskTensor]),
            ColBertVectors: ExtractColBertVectors(modelResultsList[2], [.. attentionMaskTensor]),
            TokenIds: tokenPairs
        );

        return output;
    }

    /// <summary>
    /// Extract sparse weights from model output
    /// </summary>
    private Dictionary<int, float> ExtractSparseWeights(NamedOnnxValue sparseOutput, int[] tokenIds, long[] attentionMask)
    {
        var sparseWeights = new Dictionary<int, float>();
        var tensor = sparseOutput.AsTensor<float>();
        var shape = tensor.Dimensions.ToArray();

        var seqLen = Math.Min(tokenIds.Length, shape[1]);

        for (int i = 0; i < seqLen; i++)
        {
            if (attentionMask[i] == 1 && !_specialTokenIds.Contains(tokenIds[i]))
            {
                var tokenId = tokenIds[i];

                // Use maximum value along the hidden dimension as the token weight
                float maxWeight = 0;
                for (int j = 0; j < shape[2]; j++)
                {
                    maxWeight = Math.Max(maxWeight, tensor[0, i, j]);
                }

                if (maxWeight > 0)
                {
                    sparseWeights[tokenId] = Math.Max(
                        sparseWeights.GetValueOrDefault(tokenId, 0),
                        maxWeight);
                }
            }
        }

        return sparseWeights;
    }

    /// <summary>
    /// Extract ColBERT vectors from model output
    /// </summary>
    private static float[][] ExtractColBertVectors(NamedOnnxValue colbertOutput, long[] attentionMask)
    {
        var colbertVectors = new List<float[]>();
        var tensor = colbertOutput.AsTensor<float>();
        var shape = tensor.Dimensions.ToArray();

        var seqLen = shape[1];
        var hiddenSize = shape[2];

        for (int i = 0; i < seqLen && i < attentionMask.Length; i++)
        {
            if (attentionMask[i] == 1)
            {
                var vector = new float[hiddenSize];
                for (int j = 0; j < hiddenSize; j++)
                {
                    vector[j] = tensor[0, i, j];
                }
                colbertVectors.Add(vector);
            }
        }

        return [.. colbertVectors];
    }

    /// <summary>
    /// Disposes the resources used by the M3Embedder
    /// </summary>
    public void Dispose()
    {
        Dispose(true);
        GC.SuppressFinalize(this);
    }

    protected virtual void Dispose(bool disposing)
    {
        if (!_disposed)
        {
            if (disposing)
            {
                _tokenizerSession?.Dispose();
                _modelSession?.Dispose();
            }

            _disposed = true;
        }
    }

    ~M3Embedder()
    {
        Dispose(false);
    }
}