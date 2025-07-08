import onnxruntime as ort
import numpy as np
from onnxruntime_extensions import get_library_path
from typing import Dict, List, Union


class BgeM3Embedder:
    """BGE-M3 embedder using ONNX tokenizer and model with multi-provider support"""
    
    def __init__(
        self, 
        tokenizer_path: str, 
        model_path: str, 
        provider: str = "cpu",
        device_id: int = 0
    ):
        """
        Initialize the embedder with ONNX tokenizer and model
        
        Args:
            tokenizer_path: Path to the ONNX tokenizer model
            model_path: Path to the ONNX embedding model
            provider: Execution provider ("cpu" or "cuda")
            device_id: CUDA device ID (only used when provider is "cuda")
        """
        self.provider = provider.lower()
        self.device_id = device_id
        
        # Special token IDs for sparse weights filtering
        self.special_token_ids = {0, 1, 2, 3}  # [PAD], [UNK], [CLS], [SEP]
        
        # Initialize tokenizer session (always CPU with ONNX Extensions)
        tokenizer_options = ort.SessionOptions()
        tokenizer_options.register_custom_ops_library(get_library_path())
        
        self.tokenizer_session = ort.InferenceSession(
            tokenizer_path,
            sess_options=tokenizer_options,
            providers=['CPUExecutionProvider']
        )
        
        # Initialize model session with specified provider
        model_providers = self._get_model_providers()
        self.model_session = ort.InferenceSession(
            model_path,
            providers=model_providers
        )
    
    def _get_model_providers(self) -> List[Union[str, tuple]]:
        """Get the list of execution providers for the model session"""
        if self.provider == "cuda":
            # Try CUDA first, fallback to CPU
            cuda_provider = (
                'CUDAExecutionProvider', 
                {
                    'device_id': self.device_id,
                    'arena_extend_strategy': 'kNextPowerOfTwo',
                    'gpu_mem_limit': 2 * 1024 * 1024 * 1024,  # 2GB
                    'cudnn_conv_algo_search': 'EXHAUSTIVE',
                    'do_copy_in_default_stream': True,
                }
            )
            return [cuda_provider, 'CPUExecutionProvider']
        else:
            return ['CPUExecutionProvider']
    
    def _convert_tokenizer_outputs(self, tokens: np.ndarray, token_indices: np.ndarray) -> tuple:
        """Convert tokenizer outputs to model input format"""
        # Pair tokens with their indices and sort by position
        token_pairs = list(zip(token_indices, tokens))
        token_pairs.sort()  # Sort by position (token_indices)
        
        # Get ordered tokens
        ordered_tokens = [pair[1] for pair in token_pairs]
        
        # Create input_ids and attention_mask
        input_ids = np.array([ordered_tokens], dtype=np.int64)
        attention_mask = np.ones_like(input_ids, dtype=np.int64)
        
        return input_ids, attention_mask
    
    def encode(self, text: str) -> Dict:
        """
        Generate all three types of embeddings for the input text
        
        Args:
            text: Input text to encode
            
        Returns:
            Dictionary containing:
            - dense_vecs: Dense embedding vector
            - lexical_weights: Sparse weights dictionary
            - colbert_vecs: List of ColBERT vectors
        """
        # Tokenize the input
        tokenizer_outputs = self.tokenizer_session.run(None, {"inputs": np.array([text])})
        tokens, _, token_indices = tokenizer_outputs
        
        # Convert to model input format
        input_ids, attention_mask = self._convert_tokenizer_outputs(tokens, token_indices)
        
        # Generate embeddings
        model_outputs = self.model_session.run(None, {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        })
        
        # ONNX outputs: dense_embeddings, sparse_weights, colbert_vectors
        dense_embeddings, sparse_weights, colbert_vectors = model_outputs
        
        # Process dense embeddings
        dense_vecs = dense_embeddings[0].tolist()  # Convert to list for JSON serialization
        
        # Process sparse weights
        sparse_dict = {}
        for i, token_id in enumerate(input_ids[0]):
            if attention_mask[0, i] == 1 and token_id not in self.special_token_ids:
                # Use maximum value along the hidden dimension as the token weight
                weight = np.max(sparse_weights[0, i])  # [batch, seq_len, hidden_dim]
                if weight > 0:
                    token_id_int = int(token_id)
                    sparse_dict[str(token_id_int)] = max(
                        sparse_dict.get(str(token_id_int), 0), 
                        float(weight)
                    )
        
        # Process ColBERT vectors
        colbert_list = []
        for i in range(colbert_vectors.shape[1]):  # Iterate over sequence length
            if attention_mask[0, i] == 1:  # Only include non-padding tokens
                colbert_list.append(colbert_vectors[0, i].tolist())  # Convert to list for JSON
        
        return {
            "dense_vecs": dense_vecs,
            "lexical_weights": sparse_dict,
            "colbert_vecs": colbert_list
        }
    
    def get_provider_info(self) -> Dict[str, str]:
        """Get information about the execution providers being used"""
        return {
            "tokenizer_provider": self.tokenizer_session.get_providers()[0],
            "model_provider": self.model_session.get_providers()[0],
            "requested_provider": self.provider
        }
    
    def close(self):
        """Close the ONNX sessions"""
        if hasattr(self, 'tokenizer_session'):
            del self.tokenizer_session
        if hasattr(self, 'model_session'):
            del self.model_session
    
    def __enter__(self):
        return self
    
    def __exit__(self):
        self.close()


def create_cpu_embedder(tokenizer_path: str, model_path: str) -> BgeM3Embedder:
    """Create an embedder optimized for CPU inference"""
    return BgeM3Embedder(tokenizer_path, model_path, provider="cpu")


def create_cuda_embedder(tokenizer_path: str, model_path: str, device_id: int = 0) -> BgeM3Embedder:
    """Create an embedder optimized for CUDA inference"""
    return BgeM3Embedder(tokenizer_path, model_path, provider="cuda", device_id=device_id)