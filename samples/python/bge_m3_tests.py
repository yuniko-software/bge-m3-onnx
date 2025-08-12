import pytest
import numpy as np
import os
import torch
from typing import Dict, List
from bge_m3_embedder import create_cpu_embedder, create_cuda_embedder
from transformers import AutoTokenizer
from FlagEmbedding.inference.embedder.encoder_only.m3 import M3Embedder as FlagM3EmbedderClass

class BGE_M3_Reference_Wrapper:
    """Reference wrapper using FlagEmbedding's M3Embedder for comparison"""
    def __init__(self, model_name_or_path="BAAI/bge-m3"):
        self.embedder = FlagM3EmbedderClass(
            model_name_or_path=model_name_or_path,
            use_fp16=False,
            normalize_embeddings=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    
    def encode(self, text: str) -> Dict:
        """Generate embeddings using FlagEmbedding's implementation"""
        # Use FlagEmbedding's encode method directly
        outputs = self.embedder.encode(
            text,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=True
        )
        
        # Convert to the same format as our ONNX implementation
        result = {
            "dense_vecs": outputs["dense_vecs"].tolist() if hasattr(outputs["dense_vecs"], 'tolist') else outputs["dense_vecs"],
            "lexical_weights": {str(k): float(v) for k, v in outputs["lexical_weights"].items()},
            "colbert_vecs": [vec.tolist() if hasattr(vec, 'tolist') else vec for vec in outputs["colbert_vecs"]]
        }
        
        return result


class TestBgeM3EmbeddingComparison:
    """Test class for comparing ONNX and FlagEmbedding BGE-M3 implementations"""
    
    @classmethod
    def setup_class(cls):
        """Set up test class with model paths and test texts"""
        # Find repository root
        script_dir = os.path.dirname(os.path.abspath(__file__))
        repo_root = os.path.dirname(os.path.dirname(script_dir))
        onnx_dir = os.path.join(repo_root, "onnx")
        
        cls.tokenizer_path = os.path.join(onnx_dir, "bge_m3_tokenizer.onnx")
        cls.model_path = os.path.join(onnx_dir, "bge_m3_model.onnx")
        
        # Verify required files exist
        if not os.path.exists(cls.tokenizer_path):
            pytest.skip(f"Tokenizer file not found at {cls.tokenizer_path}")
        if not os.path.exists(cls.model_path):
            pytest.skip(f"Model file not found at {cls.model_path}")
        
        # Test texts for comparison
        cls.test_texts = [
            "This is a simple test text.",
            "Hello world!",
            "A test text! Texto de prueba! Текст для теста! 測試文字!",
            "ONNX Runtime is a performance-focused engine for ONNX models.",
            "Text with numbers: 12345 and symbols: !@#$%^&*()"
        ]
        
        # Initialize reference model once for all tests
        cls.reference_embedder = BGE_M3_Reference_Wrapper("BAAI/bge-m3")
    
    def test_cpu_embeddings_match_flagembedding(self):
        """Test that CPU ONNX embeddings match FlagEmbedding implementation"""
        print("\n=== Testing CPU Provider ===")
        
        with create_cpu_embedder(self.tokenizer_path, self.model_path) as onnx_embedder:
            provider_info = onnx_embedder.get_provider_info()
            print(f"ONNX Model Provider: {provider_info['model_provider']}")
            
            self._compare_embeddings_with_reference(onnx_embedder, "CPU")
    
    def test_cuda_embeddings_match_flagembedding(self):
        """Test that CUDA ONNX embeddings match FlagEmbedding implementation"""
        print("\n=== Testing CUDA Provider ===")
        
        try:
            with create_cuda_embedder(self.tokenizer_path, self.model_path) as onnx_embedder:
                provider_info = onnx_embedder.get_provider_info()
                print(f"ONNX Model Provider: {provider_info['model_provider']}")
                
                # Skip if CUDA is not actually available
                if "CUDA" not in provider_info['model_provider']:
                    pytest.skip("CUDA provider not available on this system")
                
                self._compare_embeddings_with_reference(onnx_embedder, "CUDA")
        except Exception as e:
            pytest.skip(f"CUDA provider initialization failed: {e}")
    
    def _compare_embeddings_with_reference(self, onnx_embedder, provider_name: str):
        """
        Compare ONNX embeddings with reference FlagEmbedding implementation

        Args:
            onnx_embedder: ONNX embedder instance
            provider_name: Name of the execution provider for error messages
        """
        for text in self.test_texts:
            # Generate embeddings with both implementations
            reference_outputs = self.reference_embedder.encode(text)
            onnx_outputs = onnx_embedder.encode(text)
            
            # Compare dense embeddings
            ref_dense = np.array(reference_outputs["dense_vecs"])
            onnx_dense = np.array(onnx_outputs["dense_vecs"])
            
            dense_similarity = self._calculate_cosine_similarity(ref_dense, onnx_dense)
            assert dense_similarity > 0.9999, f"{provider_name} Dense similarity {dense_similarity:.10f} too low for '{text}'"
            
            # Compare sparse weights
            assert self._are_sparse_weights_equal(
                onnx_outputs["lexical_weights"], 
                reference_outputs["lexical_weights"]
            ), f"{provider_name} Sparse weights mismatch for '{text}'"
            
            # Compare ColBERT vectors
            assert self._are_colbert_vectors_equal(
                onnx_outputs["colbert_vecs"], 
                reference_outputs["colbert_vecs"]
            ), f"{provider_name} ColBERT vectors mismatch for '{text}'"
        
        print(f"{provider_name} embeddings match reference implementation for all {len(self.test_texts)} test texts")
    
    @staticmethod
    def _calculate_cosine_similarity(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        if vector_a.shape != vector_b.shape:
            raise ValueError("Vectors must have the same shape")
        
        dot_product = np.dot(vector_a, vector_b)
        norm_a = np.linalg.norm(vector_a)
        norm_b = np.linalg.norm(vector_b)
        
        return dot_product / (norm_a * norm_b)
    
    @staticmethod
    def _are_sparse_weights_equal(onnx_weights: Dict[str, float], ref_weights: Dict[str, float]) -> bool:
        """Compare sparse weights from ONNX and reference implementation"""
        if len(onnx_weights) != len(ref_weights):
            return False
        
        for token_id_str, ref_weight in ref_weights.items():
            if token_id_str not in onnx_weights:
                return False
            
            onnx_weight = onnx_weights[token_id_str]
            difference = abs(ref_weight - onnx_weight)
            if difference >= 1e-3:
                return False
        
        return True
    
    @staticmethod
    def _are_colbert_vectors_equal(onnx_vectors: List[List[float]], ref_vectors: List[List[float]]) -> bool:
        """Compare ColBERT vectors from ONNX and reference implementation"""
        if len(onnx_vectors) != len(ref_vectors):
            return False
        
        for i, (onnx_vec, ref_vec) in enumerate(zip(onnx_vectors, ref_vectors)):
            onnx_array = np.array(onnx_vec)
            ref_array = np.array(ref_vec)
            
            if onnx_array.shape != ref_array.shape:
                return False
            
            similarity = TestBgeM3EmbeddingComparison._calculate_cosine_similarity(onnx_array, ref_array)
            if similarity <= 0.9999:
                return False
        
        return True


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])