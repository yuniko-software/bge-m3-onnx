# BGE-M3 ONNX

<img width="1589" height="1180" alt="image" src="https://github.com/user-attachments/assets/c30cf557-4b54-42be-adc6-1c84bb704337" />

![Build](https://github.com/yuniko-software/bge-m3-onnx/actions/workflows/ci-build.yml/badge.svg)

This repository demonstrates how to convert the complete [BGE-M3](https://github.com/FlagOpen/FlagEmbedding) model to [ONNX](https://github.com/microsoft/onnxruntime) format and use it in multiple programming languages with **full multi-vector functionality**.

## Key Features

- Generate all three BGE-M3 embedding types: dense, sparse, and ColBERT vectors
- Reduced latency with local embedding generation  
- Full control over the embedding pipeline with no external dependencies
- Works offline without internet connectivity requirements
- Cross-platform compatibility (C#, Java, Python)
- CUDA GPU acceleration support

## Repository Structure

- `bge-m3-to-onnx.ipynb` - Jupyter notebook demonstrating the BGE-M3 conversion process
- `/samples/dotnet` - C# implementation
- `/samples/java` - Java implementation
- `/samples/python` - Python implementation
- `generate_reference_embeddings.py` - Script to generate reference embeddings for cross-language testing
- `run_tests.sh` and `run_tests.ps1` - Test scripts for Linux/macOS and Windows

## Getting Started

1. Clone this repository:
   ```bash
   git clone https://github.com/yuniko-software/bge-m3-onnx.git
   cd bge-m3-onnx
   ```

2. Get the BGE-M3 ONNX models:
   - Option 1: Download from releases (recommended)
     - Check the repository releases and download `onnx.zip`
     - It already contains the bge-m3 embedding model and its tokenizer
   
   - Option 2: Generate yourself using the notebook
     - Open and run `bge-m3-to-onnx.ipynb` - this is the most important file in the repository
     - The notebook demonstrates how to convert BGE-M3 from FlagEmbedding to ONNX format
     - This will create `bge_m3_tokenizer.onnx`, `bge_m3_model.onnx`, and `bge_m3_model.onnx_data` in the `/onnx` folder
   
   > Note: This repository uses [`BAAI/bge-m3`](https://github.com/FlagOpen/FlagEmbedding) as the embedding model with its XLM-RoBERTa tokenizer.

3. Generate reference embeddings (optional):
   - Run `python generate_reference_embeddings.py` to create reference embeddings for testing

4. Run the samples:
   - Once you have the ONNX models in the `/onnx` folder, you can run any sample
   - Try the .NET sample in `/samples/dotnet` or the Java sample in `/samples/java`

5. Verify cross-language embeddings (optional):
   - To ensure that .NET and Java embeddings match the Python-generated embeddings, you can run:
   
   - On Linux/macOS:
     ```bash
     chmod +x run_tests.sh
     ./run_tests.sh
     ```
   
   - On Windows:
     ```powershell
     ./run_tests.ps1
     ```
   
   > Note: These scripts require Python, .NET, Java, and Maven to be installed.

## CUDA Support

This BGE-M3 ONNX model supports CUDA GPU acceleration for improved performance. To enable CUDA support:

### Python
Install the ONNX Runtime with CUDA support:

**Resource**: [ONNX Runtime CUDA Execution Provider Requirements](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements)

This model is compatible with:
- `pip install onnxruntime-gpu[cuda,cudnn]` - packages that include CUDA and cuDNN DLLs
- [PyTorch packages that include CUDA and cuDNN DLLs](https://pytorch.org/get-started/locally/)

### C# and Java
For C# and Java implementations, you need to install CUDA and cuDNN separately:

**CUDA Installation:**
- Linux: [CUDA Installation Guide for Linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux)
- Windows: [CUDA Installation Guide for Windows](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows)

**cuDNN Installation:**
- [cuDNN Backend Installation Guide](https://docs.nvidia.com/deeplearning/cudnn/installation/latest/backend.html)

## Python Example

```python
import onnxruntime as ort
import numpy as np
from onnxruntime_extensions import get_library_path

# Initialize BGE-M3 ONNX embedder
embedder = OnnxBGEM3Embedder("onnx/bge_m3_tokenizer.onnx", "onnx/bge_m3_model.onnx")

# Generate all three embedding types
result = embedder.encode("Hello world!")

print(f"Dense: {len(result['dense_vecs'])} dimensions")
print(f"Sparse: {len(result['lexical_weights'])} tokens")  
print(f"ColBERT: {len(result['colbert_vecs'])} vectors")

# See full implementation in generate_reference_embeddings.py
```

## C# Example

```csharp
using BgeM3.Onnx;

// Initialize embedder
using var embedder = new M3Embedder(tokenizerPath, modelPath);

// Generate all embedding types
var result = embedder.GenerateEmbeddings("Hello world!");

Console.WriteLine($"Dense: {result.DenseEmbedding.Length} dimensions");
Console.WriteLine($"Sparse: {result.SparseWeights.Count} tokens");
Console.WriteLine($"ColBERT: {result.ColBertVectors.Length} vectors");

// See full implementation in samples/dotnet
```

## Java Example

```java
import com.yunikosoftware.bgem3onnx.M3Embedder;

// Initialize embedder
try (M3Embedder embedder = new M3Embedder(tokenizerPath, modelPath)) {
    // Generate all embedding types
    M3EmbeddingOutput result = embedder.generateEmbeddings("Hello world!");
    
    System.out.println("Dense: " + result.getDenseEmbedding().length + " dimensions");
    System.out.println("Sparse: " + result.getSparseWeights().size() + " tokens");
    System.out.println("ColBERT: " + result.getColBertVectors().length + " vectors");
}

// See full implementation in samples/java
```

---

⭐ **If you find this project useful, please consider giving it a star on GitHub!** ⭐ 

Your support helps make this project more visible to other developers who might benefit from BGE-M3's complete multi-vector functionality.
