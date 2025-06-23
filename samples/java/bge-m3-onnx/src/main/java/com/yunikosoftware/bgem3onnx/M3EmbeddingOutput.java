package com.yunikosoftware.bgem3onnx;

import java.util.Map;

/**
 * Output container for BGE-M3 embeddings including dense, sparse, and ColBERT
 * vectors
 */
public class M3EmbeddingOutput {
    private final float[] denseEmbedding;
    private final Map<Integer, Float> sparseWeights;
    private final float[][] colBertVectors;
    private final int[] tokenIds;

    /**
     * Creates a new M3EmbeddingOutput instance
     * 
     * @param denseEmbedding Dense embedding vector (sentence-level representation)
     * @param sparseWeights  Sparse embedding weights (token-level weights for
     *                       lexical matching)
     * @param colBertVectors ColBERT vectors (multi-vector representation, one per
     *                       token)
     * @param tokenIds       Original token IDs from the tokenizer
     */
    public M3EmbeddingOutput(float[] denseEmbedding, Map<Integer, Float> sparseWeights,
            float[][] colBertVectors, int[] tokenIds) {
        this.denseEmbedding = denseEmbedding;
        this.sparseWeights = sparseWeights;
        this.colBertVectors = colBertVectors;
        this.tokenIds = tokenIds;
    }

    /**
     * Gets the dense embedding vector (sentence-level representation)
     * 
     * @return Dense embedding as float array
     */
    public float[] getDenseEmbedding() {
        return denseEmbedding;
    }

    /**
     * Gets the sparse embedding weights (token-level weights for lexical matching)
     * 
     * @return Map of token ID to weight
     */
    public Map<Integer, Float> getSparseWeights() {
        return sparseWeights;
    }

    /**
     * Gets the ColBERT vectors (multi-vector representation, one per token)
     * 
     * @return Array of ColBERT vectors
     */
    public float[][] getColBertVectors() {
        return colBertVectors;
    }

    /**
     * Gets the original token IDs from the tokenizer
     * 
     * @return Array of token IDs
     */
    public int[] getTokenIds() {
        return tokenIds;
    }
}