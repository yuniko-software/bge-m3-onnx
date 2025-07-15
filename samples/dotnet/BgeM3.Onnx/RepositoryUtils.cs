namespace BgeM3.Onnx;

/// <summary>
/// Utility class for repository path operations
/// </summary>
public static class RepositoryUtils
{
    /// <summary>
    /// Gets the path to the onnx directory
    /// </summary>
    /// <returns>The absolute path to the onnx directory</returns>
    public static string GetOnnxDirectory() => Path.Combine(FindRepositoryRoot(), "onnx");

    /// <summary>
    /// Gets the path to the BGE-M3 tokenizer ONNX file
    /// </summary>
    /// <returns>The absolute path to the tokenizer file</returns>
    public static string GetTokenizerPath() => Path.Combine(GetOnnxDirectory(), "bge_m3_tokenizer.onnx");

    /// <summary>
    /// Gets the path to the BGE-M3 model ONNX file
    /// </summary>
    /// <returns>The absolute path to the model file</returns>
    public static string GetModelPath() => Path.Combine(GetOnnxDirectory(), "bge_m3_model.onnx");

    /// <summary>
    /// Gets the path to the performance data directory
    /// </summary>
    /// <returns>The absolute path to the performance data directory</returns>
    public static string GetPerformanceDataDirectory() => Path.Combine(FindRepositoryRoot(), "samples", "performance_data");

    /// <summary>
    /// Finds the repository root by looking for the 'onnx' directory
    /// </summary>
    /// <returns>The absolute path to the repository root</returns>
    /// <exception cref="DirectoryNotFoundException">Thrown when repository root cannot be found</exception>
    private static string FindRepositoryRoot()
    {
        var currentDir = new DirectoryInfo(Directory.GetCurrentDirectory());

        for (int i = 0; i < 10; i++)
        {
            var onnxDir = Path.Combine(currentDir.FullName, "onnx");
            if (Directory.Exists(onnxDir))
            {
                return currentDir.FullName;
            }

            if (currentDir.Parent == null)
                break;

            currentDir = currentDir.Parent;
        }

        throw new DirectoryNotFoundException("Could not locate repository root with 'onnx' directory");
    }
}