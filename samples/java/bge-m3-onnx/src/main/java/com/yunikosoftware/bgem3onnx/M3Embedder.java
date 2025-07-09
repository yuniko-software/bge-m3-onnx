package com.yunikosoftware.bgem3onnx;

import ai.onnxruntime.*;
import ai.onnxruntime.extensions.OrtxPackage;
import java.util.*;

/**
 * Provides functionality to generate embeddings using ONNX BGE-M3 model with multi-provider support
 */
public class M3Embedder implements AutoCloseable {
    private final OrtSession tokenizerSession;
    private final OrtSession modelSession;
    private final Set<Integer> specialTokenIds = Set.of(0, 1, 2, 3); // [PAD], [UNK], [CLS], [SEP]
    private final M3EmbedderConfig config;

    /**
     * Initializes a new instance of the M3Embedder class with default CPU provider
     * 
     * @param tokenizerPath Path to the ONNX tokenizer model
     * @param modelPath     Path to the ONNX BGE-M3 model
     * @throws OrtException If there's an error initializing the ONNX sessions
     */
    public M3Embedder(String tokenizerPath, String modelPath) throws OrtException {
        this(tokenizerPath, modelPath, new M3EmbedderConfig());
    }

    /**
     * Initializes a new instance of the M3Embedder class with specified configuration
     * 
     * @param tokenizerPath Path to the ONNX tokenizer model
     * @param modelPath     Path to the ONNX BGE-M3 model
     * @param config        Configuration for execution providers and other options
     * @throws OrtException If there's an error initializing the ONNX sessions
     */
    public M3Embedder(String tokenizerPath, String modelPath, M3EmbedderConfig config) throws OrtException {
        this.config = config;
        OrtEnvironment environment = OrtEnvironment.getEnvironment();

        // Initialize tokenizer session with ONNX Extensions (CPU-only)
        OrtSession.SessionOptions tokenizerOptions = createSessionOptions(true);
        tokenizerOptions.registerCustomOpLibrary(OrtxPackage.getLibraryPath());
        this.tokenizerSession = environment.createSession(tokenizerPath, tokenizerOptions);

        // Initialize model session with specified execution provider
        OrtSession.SessionOptions modelOptions = createSessionOptions(false);
        this.modelSession = environment.createSession(modelPath, modelOptions);
    }

    /**
     * Gets the configuration used by this embedder
     * 
     * @return The configuration
     */
    public M3EmbedderConfig getConfig() {
        return config;
    }

    /**
     * Creates session options with appropriate execution providers
     */
    private OrtSession.SessionOptions createSessionOptions(boolean forTokenizer) throws OrtException {
        OrtSession.SessionOptions sessionOptions = new OrtSession.SessionOptions();
        
        sessionOptions.setMemoryPatternOptimization(config.isEnableMemoryPattern());
        sessionOptions.setCPUArenaAllocator(config.isEnableCpuMemArena());
        sessionOptions.setSessionLogLevel(OrtLoggingLevel.values()[config.getLogSeverityLevel()]);

        // For tokenizer, we use CPU since it involves ONNX Extensions
        // and string processing which may not be optimized for GPU
        if (forTokenizer) {
            return sessionOptions;
        }

        // For the main model, apply the requested execution providers
        List<ExecutionProvider> providers = getProviderList();

        for (ExecutionProvider provider : providers) {
            switch (provider) {
                case CUDA:
                    Map<String, String> cudaOptions = new HashMap<>();
                    cudaOptions.put("device_id", String.valueOf(config.getCudaDeviceId()));
                    sessionOptions.addCUDA(config.getCudaDeviceId());
                    break;

                case CPU:
                    // CPU is always available and added by default
                    break;

                default:
                    throw new IllegalArgumentException("Unsupported execution provider: " + provider);
            }
        }

        return sessionOptions;
    }

    /**
     * Gets the list of execution providers to try in order
     */
    private List<ExecutionProvider> getProviderList() {
        List<ExecutionProvider> providers = new ArrayList<>();
        providers.add(config.getExecutionProvider());

        // Add fallback providers if they're different from the primary
        for (ExecutionProvider fallback : config.getFallbackProviders()) {
            if (fallback != config.getExecutionProvider() && !providers.contains(fallback)) {
                providers.add(fallback);
            }
        }

        // Always ensure CPU is available as the final fallback
        if (!providers.contains(ExecutionProvider.CPU)) {
            providers.add(ExecutionProvider.CPU);
        }

        return providers;
    }

    /**
     * Generates all embeddings (dense, sparse, ColBERT) for the input text
     * 
     * @param text The input text
     * @return The full embedding output containing all vector types
     * @throws OrtException If there's an error during inference
     */
    public M3EmbeddingOutput generateEmbeddings(String text) throws OrtException {
        OrtEnvironment env = OrtEnvironment.getEnvironment();

        // Create input tensor for tokenizer
        Map<String, OnnxTensor> tokenizerInputs = new HashMap<>();
        String[] inputArray = new String[] { text };

        try (OnnxTensor inputTensor = OnnxTensor.createTensor(env, inputArray)) {
            tokenizerInputs.put("inputs", inputTensor);

            // Run tokenizer
            try (OrtSession.Result tokenizerResults = tokenizerSession.run(tokenizerInputs)) {
                // Extract tokens and token_indices (order: tokens, instance_indices,
                // token_indices)
                int[] tokens = (int[]) tokenizerResults.get(0).getValue();
                int[] tokenIndices = (int[]) tokenizerResults.get(2).getValue();

                // Convert to input_ids by sorting tokens based on token_indices
                List<TokenIndexPair> tokenPairs = new ArrayList<>();
                for (int i = 0; i < tokens.length; i++) {
                    if (i < tokenIndices.length) {
                        tokenPairs.add(new TokenIndexPair(tokens[i], tokenIndices[i]));
                    }
                }

                // Sort by index and extract ordered tokens
                tokenPairs.sort(Comparator.comparing(pair -> pair.index));
                int[] orderedTokens = tokenPairs.stream()
                        .mapToInt(pair -> pair.token)
                        .toArray();

                // Create input_ids tensor with shape [1, orderedTokens.length]
                long[][] inputIds = new long[1][orderedTokens.length];
                for (int i = 0; i < orderedTokens.length; i++) {
                    inputIds[0][i] = orderedTokens[i];
                }

                // Create attention_mask as all 1s with same shape as input_ids
                long[][] attentionMask = new long[1][orderedTokens.length];
                Arrays.fill(attentionMask[0], 1);

                // Run the model with the prepared inputs
                Map<String, OnnxTensor> modelInputs = new HashMap<>();
                try (OnnxTensor inputIdsTensor = OnnxTensor.createTensor(env, inputIds);
                        OnnxTensor attentionMaskTensor = OnnxTensor.createTensor(env, attentionMask)) {

                    modelInputs.put("input_ids", inputIdsTensor);
                    modelInputs.put("attention_mask", attentionMaskTensor);

                    try (OrtSession.Result modelResults = modelSession.run(modelInputs)) {
                        // Process outputs
                        // Model outputs: dense_embeddings, sparse_weights, colbert_vectors
                        float[][] denseEmbeddings = (float[][]) modelResults.get(0).getValue();
                        float[][][] sparseWeights = (float[][][]) modelResults.get(1).getValue();
                        float[][][] colbertVectors = (float[][][]) modelResults.get(2).getValue();

                        return new M3EmbeddingOutput(
                                denseEmbeddings[0], // First dimension is batch
                                extractSparseWeights(sparseWeights, orderedTokens, attentionMask[0]),
                                extractColBertVectors(colbertVectors, attentionMask[0]),
                                orderedTokens);
                    }
                }
            }
        }
    }

    /**
     * Extract sparse weights from model output
     */
    private Map<Integer, Float> extractSparseWeights(float[][][] sparseOutput, int[] tokenIds, long[] attentionMask) {
        Map<Integer, Float> sparseWeights = new HashMap<>();

        int seqLen = Math.min(tokenIds.length, sparseOutput[0].length);

        for (int i = 0; i < seqLen; i++) {
            if (attentionMask[i] == 1 && !specialTokenIds.contains(tokenIds[i])) {
                int tokenId = tokenIds[i];

                // Use maximum value along the hidden dimension as the token weight
                float maxWeight = 0;
                for (int j = 0; j < sparseOutput[0][i].length; j++) {
                    maxWeight = Math.max(maxWeight, sparseOutput[0][i][j]);
                }

                if (maxWeight > 0) {
                    sparseWeights.put(tokenId, Math.max(
                            sparseWeights.getOrDefault(tokenId, 0.0f),
                            maxWeight));
                }
            }
        }

        return sparseWeights;
    }

    /**
     * Extract ColBERT vectors from model output
     */
    private float[][] extractColBertVectors(float[][][] colbertOutput, long[] attentionMask) {
        List<float[]> colbertVectors = new ArrayList<>();

        int seqLen = colbertOutput[0].length;
        int hiddenSize = colbertOutput[0][0].length;

        for (int i = 0; i < seqLen && i < attentionMask.length; i++) {
            if (attentionMask[i] == 1) {
                float[] vector = new float[hiddenSize];
                System.arraycopy(colbertOutput[0][i], 0, vector, 0, hiddenSize);
                colbertVectors.add(vector);
            }
        }

        return colbertVectors.toArray(new float[0][]);
    }

    /**
     * Helper class for pairing tokens with their indices
     */
    private static class TokenIndexPair {
        public final int token;
        public final int index;

        public TokenIndexPair(int token, int index) {
            this.token = token;
            this.index = index;
        }
    }

    @Override
    public void close() throws Exception {
        if (tokenizerSession != null) {
            tokenizerSession.close();
        }
        if (modelSession != null) {
            modelSession.close();
        }
    }
}