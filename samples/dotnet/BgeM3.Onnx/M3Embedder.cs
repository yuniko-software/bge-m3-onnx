using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace BgeM3.Onnx;

/// <summary>
/// Provides functionality to generate embeddings using ONNX bge-m3 model
/// </summary>
public class M3Embedder : IDisposable
{
    private readonly InferenceSession _tokenizerSession;
    private readonly InferenceSession _modelSession;
    private readonly HashSet<int> _specialTokenIds = [0, 1, 2, 3]; // [PAD], [UNK], [CLS], [SEP]
    private bool _disposed;

    /// <summary>
    /// Initializes a new instance of the M3Embedder class
    /// </summary>
    /// <param name="tokenizerPath">Path to the ONNX tokenizer model</param>
    /// <param name="modelPath">Path to the ONNX embedding model</param>
    public M3Embedder(string tokenizerPath, string modelPath)
    {
        // Initialize tokenizer session with ONNX Extensions
        using var tokenizerOptions = new SessionOptions();
        tokenizerOptions.RegisterOrtExtensions();
        _tokenizerSession = new InferenceSession(tokenizerPath, tokenizerOptions);

        // Initialize model session
        _modelSession = new InferenceSession(modelPath);
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
    private float[][] ExtractColBertVectors(NamedOnnxValue colbertOutput, long[] attentionMask)
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