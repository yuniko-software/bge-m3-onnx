package com.yunikosoftware.bgem3onnx;

import java.io.FileNotFoundException;
import java.nio.file.Path;
import java.nio.file.Paths;

/**
 * Utility class for repository path operations
 */
public class RepositoryUtils {
    /**
     * Gets the path to the onnx directory
     * 
     * @return The absolute path to the onnx directory
     * @throws FileNotFoundException If repository root cannot be found
     */
    public static Path getOnnxDirectory() throws FileNotFoundException {
        return findRepositoryRoot().resolve("onnx");
    }

    /**
     * Gets the path to the BGE-M3 tokenizer ONNX file
     * 
     * @return The absolute path to the tokenizer file
     * @throws FileNotFoundException If repository root cannot be found
     */
    public static Path getTokenizerPath() throws FileNotFoundException {
        return getOnnxDirectory().resolve("bge_m3_tokenizer.onnx");
    }

    /**
     * Gets the path to the BGE-M3 model ONNX file
     * 
     * @return The absolute path to the model file
     * @throws FileNotFoundException If repository root cannot be found
     */
    public static Path getModelPath() throws FileNotFoundException {
        return getOnnxDirectory().resolve("bge_m3_model.onnx");
    }

    /**
     * Gets the path to the performance data directory
     * 
     * @return The absolute path to the performance data directory
     * @throws FileNotFoundException If repository root cannot be found
     */
    public static Path getPerformanceDataDirectory() throws FileNotFoundException {
        return findRepositoryRoot().resolve("samples").resolve("performance_data");
    }

    /**
     * Finds the repository root by looking for the 'onnx' directory
     * 
     * @return The absolute path to the repository root
     * @throws FileNotFoundException If repository root cannot be found
     */
    private static Path findRepositoryRoot() throws FileNotFoundException {
        Path currentDir = Paths.get("").toAbsolutePath();

        for (int i = 0; i < 10; i++) {
            Path onnxDir = currentDir.resolve("onnx");

            if (onnxDir.toFile().exists() && onnxDir.toFile().isDirectory()) {
                return currentDir;
            }

            Path parent = currentDir.getParent();
            if (parent == null) {
                break;
            }
            currentDir = parent;
        }

        throw new FileNotFoundException("Could not locate repository root with 'onnx' directory");
    }
}