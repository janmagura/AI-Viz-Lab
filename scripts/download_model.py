#!/usr/bin/env python3
"""
AI-Viz-Lab: Model Downloader
Downloads AI models from Hugging Face for local use.
"""

import argparse
import os
import sys
from pathlib import Path

def print_header(text):
    """Print a formatted header."""
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60 + "\n")

def check_transformers():
    """Check if transformers library is available."""
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        return True
    except ImportError:
        print("❌ transformers library not installed!")
        print("   Run: pip install transformers")
        return False

def download_model(model_name: str, cache_dir: str = None):
    """Download a model from Hugging Face."""
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor
    
    if cache_dir:
        print(f"Using cache directory: {cache_dir}")
    
    print(f"📥 Downloading model: {model_name}")
    print("   This may take a while depending on your internet speed...")
    
    try:
        # Download tokenizer
        print("\n[1/3] Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            trust_remote_code=True
        )
        print("   ✅ Tokenizer downloaded")
        
        # Download processor (for vision-language models)
        print("\n[2/3] Downloading processor...")
        try:
            processor = AutoProcessor.from_pretrained(
                model_name,
                cache_dir=cache_dir,
                trust_remote_code=True
            )
            print("   ✅ Processor downloaded")
        except Exception as e:
            print(f"   ⚠️  No processor found (text-only model): {e}")
        
        # Download model
        print("\n[3/3] Downloading model weights...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            trust_remote_code=True,
            torch_dtype='auto',
            device_map='auto'
        )
        print("   ✅ Model downloaded")
        
        print(f"\n✅ Successfully downloaded: {model_name}")
        print(f"   Cache location: {cache_dir or 'default'}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error downloading model: {e}")
        print("\nTroubleshooting tips:")
        print("  - Check your internet connection")
        print("  - Verify the model name is correct")
        print("  - Try using --cache-dir to specify a different location")
        print("  - Some models require authentication: huggingface-cli login")
        return False

def list_recommended_models():
    """List recommended models for different hardware."""
    print("\n📋 Recommended Models:")
    print("\nFor CPU or low RAM (<8GB):")
    print("  - Qwen/Qwen2.5-0.5B-Instruct")
    print("  - Qwen/Qwen2.5-1.5B-Instruct")
    
    print("\nFor GPU with 4-6GB VRAM:")
    print("  - Qwen/Qwen2.5-3B-Instruct")
    print("  - microsoft/Phi-3-mini-4k-instruct")
    
    print("\nFor GPU with 8GB+ VRAM:")
    print("  - Qwen/Qwen2.5-VL-3B-Instruct (Vision-Language)")
    print("  - Qwen/Qwen2.5-7B-Instruct")
    
    print("\nFor GPU with 16GB+ VRAM:")
    print("  - Qwen/Qwen2.5-VL-7B-Instruct")
    print("  - meta-llama/Llama-3.1-8B-Instruct")

def main():
    """Main download routine."""
    parser = argparse.ArgumentParser(
        description="Download AI models for AI-Viz-Lab"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=None,
        help="Model name from Hugging Face (e.g., Qwen/Qwen2.5-1.5B-Instruct)"
    )
    parser.add_argument(
        "--cache-dir", "-c",
        type=str,
        default=None,
        help="Custom cache directory for models"
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List recommended models"
    )
    
    args = parser.parse_args()
    
    print_header("🎓 AI-Viz-Lab: Model Downloader")
    
    if args.list:
        list_recommended_models()
        return
    
    if not args.model:
        print("⚠️  No model specified!")
        print("\nUsage:")
        print("  python scripts/download_model.py --model Qwen/Qwen2.5-1.5B-Instruct")
        print("\nRecommended models:")
        list_recommended_models()
        return
    
    if not check_transformers():
        sys.exit(1)
    
    # Create models directory
    models_dir = Path(__file__).parent.parent / "models"
    models_dir.mkdir(exist_ok=True)
    
    success = download_model(args.model, args.cache_dir)
    
    if success:
        print("\n✅ Model ready to use!")
        print(f"   Update config.yaml to use: {args.model}")
    else:
        print("\n⚠️  Download failed.")
        print("   You can still run the application with a different model.")

if __name__ == "__main__":
    main()
