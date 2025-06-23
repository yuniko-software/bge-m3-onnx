package com.yunikosoftware.bgem3onnx;

import static org.junit.jupiter.api.Assertions.fail;

import java.io.FileNotFoundException;
import java.io.FileReader;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.AfterAll;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;

/**
 * Reference embedding data structure from Python-generated JSON
 */
class BgeM3ReferenceEmbedding {
    public float[] dense_vecs;
    public Map<String, Double> lexical_weights;
    public float[][] colbert_vecs;
}

public class BgeM3EmbeddingComparisonTests {
    private static M3Embedder embedder;
    private static Map<String, BgeM3ReferenceEmbedding> referenceEmbeddings;

    @BeforeAll
    public static void setup() throws Exception {
        Path repoDir = findRepositoryRoot();
        Path onnxDir = repoDir.resolve("onnx");

        Path tokenizerPath = onnxDir.resolve("bge_m3_tokenizer.onnx");
        Path modelPath = onnxDir.resolve("bge_m3_model.onnx");
        Path referenceFile = onnxDir.resolve("bge_m3_reference_embeddings.json");

        if (!tokenizerPath.toFile().exists())
            throw new FileNotFoundException("Tokenizer file not found at " + tokenizerPath);
        if (!modelPath.toFile().exists())
            throw new FileNotFoundException("Model file not found at " + modelPath);
        if (!referenceFile.toFile().exists())
            throw new FileNotFoundException("Reference embeddings file not found at " + referenceFile +
                    ". Please run the Python script to generate reference embeddings first.");

        embedder = new M3Embedder(tokenizerPath.toString(), modelPath.toString());

        ObjectMapper mapper = new ObjectMapper();
        referenceEmbeddings = mapper.readValue(
                new FileReader(referenceFile.toFile()),
                new TypeReference<Map<String, BgeM3ReferenceEmbedding>>() {
                });
    }

    @Test
    public void allEmbeddingTypes_ShouldMatchPythonEmbeddings() throws Exception {
        List<String> failedComparisons = new ArrayList<>();

        for (Map.Entry<String, BgeM3ReferenceEmbedding> entry : referenceEmbeddings.entrySet()) {
            String text = entry.getKey();
            BgeM3ReferenceEmbedding referenceEmbedding = entry.getValue();

            try {
                M3EmbeddingOutput result = embedder.generateEmbeddings(text);

                // Compare dense embeddings
                double denseSimilarity = calculateCosineSimilarity(result.getDenseEmbedding(),
                        referenceEmbedding.dense_vecs);
                if (denseSimilarity <= 0.9999) {
                    failedComparisons.add(String.format("Dense similarity %.10f for '%s'", denseSimilarity, text));
                }

                // Compare sparse weights
                if (!areSparseWeightsEqual(result.getSparseWeights(), referenceEmbedding.lexical_weights)) {
                    failedComparisons.add(String.format("Sparse weights mismatch for '%s'", text));
                }

                // Compare ColBERT vectors
                if (!areColBertVectorsEqual(result.getColBertVectors(), referenceEmbedding.colbert_vecs)) {
                    failedComparisons.add(String.format("ColBERT vectors mismatch for '%s'", text));
                }
            } catch (Exception ex) {
                failedComparisons.add(String.format("Exception for '%s': %s", text, ex.getMessage()));
            }
        }

        if (!failedComparisons.isEmpty()) {
            String errorMessage = "Embedding comparison failures:\n" + String.join("\n", failedComparisons);
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
            if (difference >= 1e-6) {
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

    @AfterAll
    public static void cleanup() throws Exception {
        if (embedder != null) {
            embedder.close();
        }
    }
}