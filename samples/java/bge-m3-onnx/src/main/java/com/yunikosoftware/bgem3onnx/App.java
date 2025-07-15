package com.yunikosoftware.bgem3onnx;

import java.text.DecimalFormat;
import java.util.Map;

/**
 * Sample application demonstrating BGE-M3 ONNX usage
 */
public class App {
    private static final DecimalFormat df = new DecimalFormat("#.######");

    public static void main(String[] args) {
        try {
            // Use utility class for paths
            var tokenizerPath = RepositoryUtils.getTokenizerPath();
            var modelPath = RepositoryUtils.getModelPath();

            // Sample text to test with
            String text = "A test text! Texto de prueba! Текст для теста! 測試文字! Testtext! Testez le texte! Сынақ мәтіні! Тестни текст! परीक्षण पाठ! Kiểm tra văn bản!";

            System.out.println("===== BGE-M3 ONNX Multi-Provider Demo =====");
            System.out.println("Tokenizer: " + tokenizerPath.getFileName());
            System.out.println("Model: " + modelPath.getFileName());

            // Test CPU provider
            System.out.println("\n===== CPU PROVIDER =====");
            testProvider("CPU", () -> {
                try {
                    return M3EmbedderFactory.createCpuOptimized(tokenizerPath.toString(), modelPath.toString());
                } catch (Exception e) {
                    throw new RuntimeException(e);
                }
            }, text);

            // Test CUDA provider
            System.out.println("\n===== CUDA PROVIDER =====");
            try {
                testProvider("CUDA", () -> {
                    try {
                        return M3EmbedderFactory.createCudaOptimized(tokenizerPath.toString(), modelPath.toString());
                    } catch (Exception e) {
                        throw new RuntimeException(e);
                    }
                }, text);
            } catch (Exception ex) {
                System.out.println("CUDA not available: " + ex.getMessage());
            }

            System.out.println("\n===== DEMO COMPLETE =====");

        } catch (Exception e) {
            System.err.println("Error: " + e.getMessage());
            e.printStackTrace();
            System.exit(1);
        }
    }

    private static void testProvider(String providerName, EmbedderSupplier embedderFactory, String testText) {
        try (M3Embedder embedder = embedderFactory.get()) {
            System.out.println("Provider: " + embedder.getConfig().getExecutionProvider());

            // Generate embeddings
            long startTime = System.currentTimeMillis();
            M3EmbeddingOutput embeddings = embedder.generateEmbeddings(testText);
            long elapsedTime = System.currentTimeMillis() - startTime;

            System.out.println("Inference time: " + elapsedTime + "ms");

            // Print dense embedding information
            System.out.println("\n=== DENSE EMBEDDING ===");
            float[] denseEmbedding = embeddings.getDenseEmbedding();
            System.out.println("Length: " + denseEmbedding.length);
            System.out.print("First 10 values: [");
            for (int i = 0; i < Math.min(10, denseEmbedding.length); i++) {
                if (i > 0)
                    System.out.print(", ");
                System.out.print(df.format(denseEmbedding[i]));
            }
            System.out.println("]");

            // Print sparse weights information
            System.out.println("\n=== SPARSE WEIGHTS ===");
            Map<Integer, Float> sparseWeights = embeddings.getSparseWeights();
            System.out.println("Non-zero tokens: " + sparseWeights.size());

            // Top tokens
            System.out.println("Top 5 tokens:");
            sparseWeights.entrySet().stream()
                    .sorted((a, b) -> Float.compare(b.getValue(), a.getValue()))
                    .limit(5)
                    .forEach(entry -> System.out
                            .println("  " + entry.getKey() + ": " + df.format(entry.getValue())));

            // Print ColBERT vectors information
            System.out.println("\n=== COLBERT VECTORS ===");
            float[][] colbertVectors = embeddings.getColBertVectors();
            System.out.println("Token count: " + colbertVectors.length);
            if (colbertVectors.length > 0) {
                System.out.println("Vector dimension: " + colbertVectors[0].length);

                // Print first vector
                System.out.print("First vector (first 10 values): [");
                for (int i = 0; i < Math.min(10, colbertVectors[0].length); i++) {
                    if (i > 0)
                        System.out.print(", ");
                    System.out.print(df.format(colbertVectors[0][i]));
                }
                System.out.println("]");
            }

            System.out.println("\n" + providerName + " completed successfully!");

        } catch (Exception e) {
            throw new RuntimeException("Failed to test " + providerName + " provider", e);
        }
    }

    @FunctionalInterface
    private interface EmbedderSupplier {
        M3Embedder get() throws Exception;
    }
}