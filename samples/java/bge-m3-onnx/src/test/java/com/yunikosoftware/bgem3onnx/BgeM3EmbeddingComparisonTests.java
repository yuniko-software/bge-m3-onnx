package com.yunikosoftware.bgem3onnx;

import static org.junit.jupiter.api.Assertions.fail;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import java.io.FileNotFoundException;
import java.io.FileReader;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.AfterEach;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;

import ai.onnxruntime.OrtException;

/**
 * Reference embedding data structure from Python-generated JSON
 */
class BgeM3ReferenceEmbedding {
    public float[] dense_vecs;
    public Map<String, Double> lexical_weights;
    public float[][] colbert_vecs;
}

public class BgeM3EmbeddingComparisonTests {
    private static Map<String, BgeM3ReferenceEmbedding> referenceEmbeddings;
    private static String tokenizerPath;
    private static String modelPath;
    private static boolean cudaAvailable;

    private M3Embedder cpuEmbedder;
    private M3Embedder cudaEmbedder;

    @BeforeAll
    public static void setupClass() throws Exception {
        Path repoDir = findRepositoryRoot();
        Path onnxDir = repoDir.resolve("onnx");

        Path tokenizerFile = onnxDir.resolve("bge_m3_tokenizer.onnx");
        Path modelFile = onnxDir.resolve("bge_m3_model.onnx");
        Path referenceFile = onnxDir.resolve("bge_m3_reference_embeddings.json");

        if (!tokenizerFile.toFile().exists())
            throw new FileNotFoundException("Tokenizer file not found at " + tokenizerFile);
        if (!modelFile.toFile().exists())
            throw new FileNotFoundException("Model file not found at " + modelFile);
        if (!referenceFile.toFile().exists())
            throw new FileNotFoundException("Reference embeddings file not found at " + referenceFile +
                    ". Please run the Python script to generate reference embeddings first.");

        tokenizerPath = tokenizerFile.toString();
        modelPath = modelFile.toString();

        // Test CUDA availability
        try {
            M3Embedder testCudaEmbedder = M3EmbedderFactory.createCudaOptimized(tokenizerPath, modelPath);
            testCudaEmbedder.close();
            cudaAvailable = true;
        } catch (OrtException e) {
            cudaAvailable = false;
        }

        ObjectMapper mapper = new ObjectMapper();
        referenceEmbeddings = mapper.readValue(
                new FileReader(referenceFile.toFile()),
                new TypeReference<Map<String, BgeM3ReferenceEmbedding>>() {
                });
    }

    @Test
    public void cpuEmbeddings_ShouldMatchPythonEmbeddings() throws Exception {
        cpuEmbedder = M3EmbedderFactory.createCpuOptimized(tokenizerPath, modelPath);
        validateEmbeddingsAgainstReference(cpuEmbedder, "CPU");
    }

    @Test
    public void cudaEmbeddings_ShouldMatchPythonEmbeddings() throws Exception {
        // Skip test if CUDA is not available
        assumeTrue(cudaAvailable, "CUDA provider is not available on this system");

        cudaEmbedder = M3EmbedderFactory.createCudaOptimized(tokenizerPath, modelPath);
        validateEmbeddingsAgainstReference(cudaEmbedder, "CUDA");
    }

    /**
     * Helper method to validate embeddings against Python reference embeddings
     * 
     * @param embedder     The embedder instance to test
     * @param providerName Name of the execution provider for error messages
     */
    private void validateEmbeddingsAgainstReference(M3Embedder embedder, String providerName) throws Exception {
        List<String> failedComparisons = new ArrayList<>();

        for (Map.Entry<String, BgeM3ReferenceEmbedding> entry : referenceEmbeddings.entrySet()) {
            String text = entry.getKey();
            BgeM3ReferenceEmbedding referenceEmbedding = entry.getValue();

            try {
                M3EmbeddingOutput result = embedder.generateEmbeddings(text);

                // Verify we're using the expected provider
                ExecutionProvider expectedProvider = providerName.equals("CPU") ? ExecutionProvider.CPU
                        : ExecutionProvider.CUDA;
                if (embedder.getConfig().getExecutionProvider() != expectedProvider) {
                    failedComparisons.add(providerName + " Expected provider " + expectedProvider + " but got "
                            + embedder.getConfig().getExecutionProvider());
                }

                double denseSimilarity = calculateCosineSimilarity(result.getDenseEmbedding(),
                        referenceEmbedding.dense_vecs);
                if (denseSimilarity <= 0.9999) {
                    failedComparisons.add(
                            String.format("%s Dense similarity %.10f for '%s'", providerName, denseSimilarity, text));
                }

                if (!areSparseWeightsEqual(result.getSparseWeights(), referenceEmbedding.lexical_weights)) {
                    failedComparisons.add(String.format("%s Sparse weights mismatch for '%s'", providerName, text));
                }

                if (!areColBertVectorsEqual(result.getColBertVectors(), referenceEmbedding.colbert_vecs)) {
                    failedComparisons.add(String.format("%s ColBERT vectors mismatch for '%s'", providerName, text));
                }
            } catch (Exception ex) {
                failedComparisons.add(String.format("%s Exception for '%s': %s", providerName, text, ex.getMessage()));
            }
        }

        if (!failedComparisons.isEmpty()) {
            String errorMessage = providerName + " embedding comparison failures:\n"
                    + String.join("\n", failedComparisons);
            fail(errorMessage);
        }
    }

    private static double calculateCosineSimilarity(float[] vectorA, float[] vectorB) {
        if (vectorA.length != vectorB.length) {
            throw new IllegalArgumentException("Vectors must be of the same length");
        }

        double dotProduct = 0;
        double normA = 0;
        double normB = 0;

        for (int i = 0; i < vectorA.length; i++) {
            dotProduct += vectorA[i] * vectorB[i];
            normA += vectorA[i] * vectorA[i];
            normB += vectorB[i] * vectorB[i];
        }

        return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
    }

    private boolean areSparseWeightsEqual(Map<Integer, Float> javaWeights, Map<String, Double> pythonWeights) {
        if (javaWeights.size() != pythonWeights.size()) {
            return false;
        }

        for (Map.Entry<String, Double> entry : pythonWeights.entrySet()) {
            Integer tokenId = Integer.parseInt(entry.getKey());
            if (!javaWeights.containsKey(tokenId)) {
                return false;
            }

            double difference = Math.abs(entry.getValue() - javaWeights.get(tokenId));
            if (difference >= 1e-3) {
                return false;
            }
        }

        return true;
    }

    private boolean areColBertVectorsEqual(float[][] javaVectors, float[][] pythonVectors) {
        if (javaVectors.length != pythonVectors.length) {
            return false;
        }

        for (int i = 0; i < pythonVectors.length; i++) {
            if (javaVectors[i].length != pythonVectors[i].length) {
                return false;
            }

            double similarity = calculateCosineSimilarity(javaVectors[i], pythonVectors[i]);
            if (similarity <= 0.9999) {
                return false;
            }
        }

        return true;
    }

    @AfterEach
    public void cleanup() throws Exception {
        if (cpuEmbedder != null) {
            cpuEmbedder.close();
            cpuEmbedder = null;
        }
        if (cudaEmbedder != null) {
            cudaEmbedder.close();
            cudaEmbedder = null;
        }
    }

    private static Path findRepositoryRoot() throws FileNotFoundException {
        Path currentDir = Paths.get("").toAbsolutePath();

        // Search up the directory tree for the repository root
        for (int i = 0; i < 10; i++) {
            Path onnxDir = currentDir.resolve("onnx");

            if (onnxDir.toFile().exists() && onnxDir.toFile().isDirectory()) {
                return currentDir;
            }

            Path parent = currentDir.getParent();
            if (parent == null)
                break;
            currentDir = parent;
        }

        throw new FileNotFoundException("Could not locate repository root with 'onnx' directory");
    }
}