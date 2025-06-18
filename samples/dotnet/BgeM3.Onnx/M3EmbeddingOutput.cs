namespace BgeM3.Onnx;

/// <summary>
/// Output container for BGE-M3 embeddings including dense, sparse, and ColBERT vectors
/// </summary>
/// <param name="DenseEmbedding">Dense embedding vector (sentence-level representation)</param>
/// <param name="SparseWeights">Sparse embedding weights (token-level weights for lexical matching)</param>
/// <param name="ColBertVectors">ColBERT vectors (multi-vector representation, one per token)</param>
/// <param name="TokenIds">Original token IDs from the tokenizer</param>
public record M3EmbeddingOutput(float[] DenseEmbedding, Dictionary<int, float> SparseWeights, float[][] ColBertVectors, int[] TokenIds);