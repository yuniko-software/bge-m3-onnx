#!/usr/bin/env python3
"""
BGE-M3 Performance Testing Script for Python
Tests Transformers BGE-M3, Python ONNX CPU, and Python ONNX CUDA implementations
"""

import json
import os
import sys
import time
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime

# Add the samples/python directory to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

import torch
from FlagEmbedding.inference.embedder.encoder_only.m3 import M3Embedder as FlagM3Embedder
from bge_m3_embedder import create_cpu_embedder, create_cuda_embedder


class BenchmarkRunner:
    """Run performance benchmarks for different BGE-M3 implementations"""
    
    def __init__(self, tokenizer_path: str, model_path: str):
        self.tokenizer_path = tokenizer_path
        self.model_path = model_path
    
    def _run_benchmark_core(self, 
                           texts: List[str], 
                           scenario_name: str,
                           encode_func: Callable,
                           warmup_text: str = "warm up text") -> Dict[str, Any]:
        """
        Core benchmark logic shared across all implementations
        
        Args:
            texts: List of texts to process
            scenario_name: Name of the scenario for results
            encode_func: Function to call for encoding (takes text as parameter)
            warmup_text: Text to use for warmup
        """
        print(f"Benchmarking {scenario_name}...")
        
        # Warm up
        encode_func(warmup_text)
        
        # Run benchmark
        start_time = time.time()
        latencies = []
        results = []
        
        for i, text in enumerate(texts):
            text_start = time.time()
            
            try:
                result = encode_func(text)
                
                # Convert to consistent format
                processed_result = {
                    "dense_size": len(result.get("dense_vecs", [])),
                    "sparse_tokens": len(result.get("lexical_weights", {})),
                    "colbert_vectors": len(result.get("colbert_vecs", []))
                }
                results.append(processed_result)
                
            except Exception as e:
                print(f"Error processing text {i}: {e}")
                results.append(None)
            
            text_latency = time.time() - text_start
            latencies.append(text_latency)
            
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{len(texts)} texts")
        
        total_time = time.time() - start_time
        
        return {
            "scenario": scenario_name,
            "total_time_seconds": total_time,
            "average_latency_ms": (sum(latencies) / len(latencies)) * 1000,
            "median_latency_ms": sorted(latencies)[len(latencies) // 2] * 1000,
            "min_latency_ms": min(latencies) * 1000,
            "max_latency_ms": max(latencies) * 1000,
            "throughput_texts_per_second": len(texts) / total_time,
            "successful_embeddings": sum(1 for r in results if r is not None),
            "failed_embeddings": sum(1 for r in results if r is None),
            "per_text_latencies_ms": [l * 1000 for l in latencies]
        }
        
    def benchmark_transformers(self, texts: List[str]) -> Dict[str, Any]:
        """Benchmark original Transformers BGE-M3 implementation"""
        # Initialize model
        init_start = time.time()
        embedder = FlagM3Embedder(
            model_name_or_path="BAAI/bge-m3",
            use_fp16=False,
            normalize_embeddings=True
        )
        init_time = time.time() - init_start
        
        # Define encode function for transformers
        def encode_func(text):
            return embedder.encode(
                text,
                return_dense=True,
                return_sparse=True,
                return_colbert_vecs=True
            )
        
        # Run benchmark
        result = self._run_benchmark_core(texts, "transformers_cpu", encode_func)
        result["initialization_time_seconds"] = init_time
        
        return result
    
    def benchmark_onnx_cpu(self, texts: List[str]) -> Dict[str, Any]:
        """Benchmark ONNX CPU implementation"""
        # Initialize model
        init_start = time.time()
        embedder = create_cpu_embedder(self.tokenizer_path, self.model_path)
        init_time = time.time() - init_start
        
        # Define encode function for ONNX
        def encode_func(text):
            return embedder.encode(text)
        
        # Run benchmark
        result = self._run_benchmark_core(texts, "onnx_cpu", encode_func)
        result["initialization_time_seconds"] = init_time
        
        # Clean up
        embedder.close()
        
        return result
    
    def benchmark_onnx_cuda(self, texts: List[str]) -> Optional[Dict[str, Any]]:
        """Benchmark ONNX CUDA implementation"""
        try:
            # Initialize model
            init_start = time.time()
            embedder = create_cuda_embedder(self.tokenizer_path, self.model_path)
            init_time = time.time() - init_start
            
            # Check if CUDA is actually being used
            provider_info = embedder.get_provider_info()
            if "CUDA" not in provider_info.get("model_provider", ""):
                print("CUDA provider not available, skipping CUDA benchmark")
                embedder.close()
                return None
            
            # Clear GPU memory before benchmark
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Define encode function for ONNX CUDA
            def encode_func(text):
                return embedder.encode(text)
            
            # Run benchmark
            result = self._run_benchmark_core(texts, "onnx_cuda", encode_func)
            result["initialization_time_seconds"] = init_time
            result["provider_info"] = provider_info
            
            # Clean up
            embedder.close()
            
            return result
            
        except Exception as e:
            print(f"CUDA benchmark failed: {e}")
            return None


def load_test_dataset(data_dir: str) -> List[Dict[str, Any]]:
    """Load the test dataset"""
    dataset_path = os.path.join(data_dir, "test_texts.json")
    
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Test dataset not found at {dataset_path}. Please run the dataset generator first.")
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    return dataset


def main():
    """Main function to run performance benchmarks"""
    # Find repository root and paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(os.path.dirname(script_dir))
    
    # Set paths
    performance_data_dir = os.path.join(repo_root, "samples", "performance_data")
    onnx_dir = os.path.join(repo_root, "onnx")
    
    tokenizer_path = os.path.join(onnx_dir, "bge_m3_tokenizer.onnx")
    model_path = os.path.join(onnx_dir, "bge_m3_model.onnx")
    
    print("=" * 60)
    print("BGE-M3 Python Performance Benchmark")
    print("=" * 60)
    print(f"Performance data: {performance_data_dir}")
    print(f"Tokenizer: {tokenizer_path}")
    print(f"Model: {model_path}")
    
    # Verify required files exist
    for path in [tokenizer_path, model_path]:
        if not os.path.exists(path):
            print(f"ERROR: Required file not found: {path}")
            sys.exit(1)
    
    # Load test dataset
    try:
        print("\nLoading test dataset...")
        dataset = load_test_dataset(performance_data_dir)
        print(f"Loaded {len(dataset)} texts")
        
        texts = [item["text"] for item in dataset]
        
    except Exception as e:
        print(f"ERROR: Failed to load test dataset: {e}")
        sys.exit(1)
    
    # Initialize benchmark runner
    runner = BenchmarkRunner(tokenizer_path, model_path)
    
    # Run benchmarks
    results = {
        "test_info": {
            "timestamp": datetime.now().isoformat(),
            "test_dataset_size": len(texts),
            "sample_texts": texts[:5]  # Store first 5 texts for reference
        },
        "scenarios": {}
    }
    
    print(f"\nRunning benchmarks on {len(texts)} texts...")
    
    # Benchmark 1: Transformers BGE-M3
    try:
        print(f"\n{'-' * 40}")
        result = runner.benchmark_transformers(texts)
        results["scenarios"]["transformers_cpu"] = result
        print(f"Transformers BGE-M3: {result['average_latency_ms']:.1f}ms avg, "
              f"{result['throughput_texts_per_second']:.1f} texts/sec")
    except Exception as e:
        print(f"Transformers benchmark failed: {e}")
        results["scenarios"]["transformers_cpu"] = {"error": str(e)}
    
    # Benchmark 2: ONNX CPU
    try:
        print(f"\n{'-' * 40}")
        result = runner.benchmark_onnx_cpu(texts)
        results["scenarios"]["onnx_cpu"] = result
        print(f"ONNX CPU: {result['average_latency_ms']:.1f}ms avg, "
              f"{result['throughput_texts_per_second']:.1f} texts/sec")
    except Exception as e:
        print(f"ONNX CPU benchmark failed: {e}")
        results["scenarios"]["onnx_cpu"] = {"error": str(e)}
    
    # Benchmark 3: ONNX CUDA
    # Check CUDA availability before attempting
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        try:
            print(f"\n{'-' * 40}")
            result = runner.benchmark_onnx_cuda(texts)
            if result:
                results["scenarios"]["onnx_cuda"] = result
                print(f"ONNX CUDA: {result['average_latency_ms']:.1f}ms avg, "
                      f"{result['throughput_texts_per_second']:.1f} texts/sec")
            else:
                results["scenarios"]["onnx_cuda"] = {"error": "CUDA provider not available"}
        except Exception as e:
            print(f"ONNX CUDA benchmark failed: {e}")
            results["scenarios"]["onnx_cuda"] = {"error": str(e)}
    else:
        print(f"\n{'-' * 40}")
        print("ONNX CUDA: Skipped (CUDA not available)")
        results["scenarios"]["onnx_cuda"] = {"error": "CUDA not available"}
    
    # Save results
    os.makedirs(onnx_dir, exist_ok=True)
    output_path = os.path.join(onnx_dir, "performance_python.json")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'=' * 60}")
    print("Benchmark Summary")  
    print(f"{'=' * 60}")
    
    # Print summary table
    scenarios = results["scenarios"]
    if scenarios:
        print(f"{'Scenario':<20} {'Avg Latency (ms)':<18} {'Throughput (t/s)':<18} {'Status'}")
        print("-" * 70)
        
        for scenario_name, scenario_data in scenarios.items():
            if "error" in scenario_data:
                print(f"{scenario_name:<20} {'ERROR':<18} {'ERROR':<18} {scenario_data['error']}")
            else:
                avg_lat = scenario_data.get('average_latency_ms', 0)
                throughput = scenario_data.get('throughput_texts_per_second', 0)
                print(f"{scenario_name:<20} {avg_lat:<18.1f} {throughput:<18.1f} {'Success'}")
    
    print(f"\nResults saved to: {output_path}")
    print("Python performance benchmark completed!")


if __name__ == "__main__":
    main()