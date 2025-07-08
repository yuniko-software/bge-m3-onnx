import json
import os
import argparse
from bge_m3_embedder import create_cpu_embedder, create_cuda_embedder


def main():
    """Generate reference embeddings for all three types using BGE-M3 ONNX models"""
    
    parser = argparse.ArgumentParser(description='Generate BGE-M3 reference embeddings')
    parser.add_argument('--provider', choices=['cpu', 'cuda'], default='cpu',
                        help='Execution provider to use (default: cpu)')
    parser.add_argument('--device-id', type=int, default=0,
                        help='CUDA device ID (only used with cuda provider, default: 0)')
    
    args = parser.parse_args()
    
    # Find the repository root (go up two levels from samples/python)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(os.path.dirname(script_dir))
    onnx_dir = os.path.join(repo_root, "onnx")
    
    tokenizer_path = os.path.join(onnx_dir, "bge_m3_tokenizer.onnx")
    model_path = os.path.join(onnx_dir, "bge_m3_model.onnx")
    output_path = os.path.join(onnx_dir, "bge_m3_reference_embeddings.json")
    
    print(f"Using execution provider: {args.provider.upper()}")
    if args.provider == 'cuda':
        print(f"CUDA device ID: {args.device_id}")
    
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

    # Initialize the BGE-M3 embedder with specified provider
    print("Initializing BGE-M3 ONNX embedder...")
    
    try:
        if args.provider == 'cuda':
            embedder = create_cuda_embedder(tokenizer_path, model_path, args.device_id)
        else:
            embedder = create_cpu_embedder(tokenizer_path, model_path)
        
        # Print provider information
        provider_info = embedder.get_provider_info()
        print(f"Tokenizer provider: {provider_info['tokenizer_provider']}")
        print(f"Model provider: {provider_info['model_provider']}")
        
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
        
        print(f"\nGenerating embeddings for {len(test_texts)} test texts...")
        
        for i, text in enumerate(test_texts, 1):
            print(f"Processing text {i}/{len(test_texts)}...")
            
            result = embedder.encode(text)
            embeddings[text] = result
        
        # Save to JSON file
        print(f"\nSaving reference embeddings to {output_path}")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(embeddings, f, ensure_ascii=False, indent=2)
        
        print(f"Saved {len(embeddings)} reference embeddings with all three types (dense, sparse, ColBERT)")
        print("\nReference embeddings generated successfully!")
    
    except Exception as e:
        print(f"ERROR: Failed to generate embeddings: {e}")
        if args.provider == 'cuda':
            print("If you're having CUDA issues, try using --provider cpu")
        raise
    
    finally:
        if 'embedder' in locals():
            embedder.close()


if __name__ == "__main__":
    main()