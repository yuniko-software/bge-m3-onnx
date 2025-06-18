import json
import onnxruntime as ort
import numpy as np
from onnxruntime_extensions import get_library_path
import os

def convert_tokenizer_outputs(tokens, token_indices):
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

class OnnxBGEM3Embedder:
    """BGE-M3 embedder using ONNX tokenizer and model"""
    
    def __init__(self, tokenizer_path, model_path):
        """Initialize the embedder with ONNX tokenizer and model"""
        # Initialize tokenizer session
        sess_options = ort.SessionOptions()
        sess_options.register_custom_ops_library(get_library_path())
        self.tokenizer_session = ort.InferenceSession(
            tokenizer_path,
            sess_options=sess_options,
            providers=['CPUExecutionProvider']
        )
        
        # Initialize model session
        self.model_session = ort.InferenceSession(
            model_path,
            providers=['CPUExecutionProvider']
        )
        
        # Special token IDs for sparse weights filtering
        self.special_token_ids = {0, 1, 2, 3}
    
    def encode(self, text):
        """Generate all three types of embeddings for the input text"""
        # Tokenize the input
        tokenizer_outputs = self.tokenizer_session.run(None, {"inputs": np.array([text])})
        tokens, _, token_indices = tokenizer_outputs
        
        # Convert to model input format
        input_ids, attention_mask = convert_tokenizer_outputs(tokens, token_indices)
        
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
                weight = sparse_weights[0, i]  # [batch, seq_len]
                if weight > 0:
                    token_id_int = int(token_id)
                    sparse_dict[str(token_id_int)] = max(sparse_dict.get(str(token_id_int), 0), float(weight.item()))
        
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

def main():
    """Generate reference embeddings for all three types using BGE-M3 ONNX models"""
    
    script_dir = os.getcwd()
    onnx_dir = os.path.join(script_dir, "onnx")
    
    tokenizer_path = os.path.join(onnx_dir, "bge_m3_tokenizer.onnx")
    model_path = os.path.join(onnx_dir, "bge_m3_model.onnx")
    output_path = os.path.join(onnx_dir, "bge_m3_reference_embeddings.json")
    
    print(f"Using tokenizer: {tokenizer_path}")
    print(f"Using model: {model_path}")
    print(f"Output will be saved to: {output_path}")

    # Verify files exist
    if not os.path.exists(tokenizer_path):
        print(f"ERROR: Tokenizer file not found at {tokenizer_path}")
        return
    
    if not os.path.exists(model_path):
        print(f"ERROR: Model file not found at {model_path}")
        return

    # Initialize the BGE-M3 embedder
    print("Initializing BGE-M3 ONNX embedder...")
    embedder = OnnxBGEM3Embedder(tokenizer_path, model_path)
    
    # Test texts
    test_texts = [
        "This is a simple test text.",
        "Hello world!",
        "A test text! Texto de prueba! Текст для теста! 測試文字! Testtext! Testez le texte! Сынақ мәтіні! Тестни текст! परीक्षण पाठ! Kiểm tra văn bản!",
        "",
        "This is a longer text that should generate a meaningful embedding vector. The embedding model should capture the semantic meaning of this text.",
        "ONNX Runtime is a performance-focused engine for ONNX models.",
        "Text with numbers: 12345 and symbols: !@#$%^&*()",
        "English, Español, Русский, 中文, العربية, हिन्दी"
    ]
    
    embeddings = {}
    
    for text in test_texts:
        result = embedder.encode(text)
        
        embeddings[text] = result
    
    # Save to JSON file
    print(f"\nSaving reference embeddings to {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(embeddings, f, ensure_ascii=False, indent=2)
    
    print(f"Saved {len(embeddings)} reference embeddings with all three types (dense, sparse, ColBERT)")
    print("\nReference embeddings generated successfully!")

if __name__ == "__main__":
    main()