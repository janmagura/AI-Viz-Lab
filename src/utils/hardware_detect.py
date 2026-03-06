#!/usr/bin/env python3
"""
AI-Viz-Lab: Hardware Detection Utility
Auto-detects CPU/GPU capabilities and generates optimal configuration.
"""

import argparse
import sys
import os
from pathlib import Path

def print_header(text):
    """Print a formatted header."""
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60 + "\n")

def detect_cpu():
    """Detect CPU information."""
    import platform
    import multiprocessing
    
    cpu_info = {
        'processor': platform.processor() or 'Unknown',
        'cores': multiprocessing.cpu_count(),
        'system': platform.system(),
        'machine': platform.machine()
    }
    
    print(f"\n🖥️  CPU Information:")
    print(f"   Processor: {cpu_info['processor']}")
    print(f"   Cores: {cpu_info['cores']}")
    print(f"   System: {cpu_info['system']} {cpu_info['machine']}")
    
    return cpu_info

def detect_gpu():
    """Detect GPU information."""
    gpu_info = {
        'available': False,
        'type': None,
        'name': None,
        'memory_gb': 0
    }
    
    # Try to detect NVIDIA GPU
    try:
        import torch
        if torch.cuda.is_available():
            gpu_info['available'] = True
            gpu_info['type'] = 'cuda'
            gpu_info['name'] = torch.cuda.get_device_name(0)
            gpu_info['memory_gb'] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            
            print(f"\n🎮 GPU Detected (NVIDIA CUDA):")
            print(f"   Name: {gpu_info['name']}")
            print(f"   Memory: {gpu_info['memory_gb']:.2f} GB")
            print(f"   CUDA Version: {torch.version.cuda}")
            return gpu_info
    except ImportError:
        pass
    except Exception as e:
        print(f"\n⚠️  GPU detection error: {e}")
    
    # Try to detect AMD GPU (ROCm)
    try:
        import torch
        if hasattr(torch, 'backends') and hasattr(torch.backends, 'mps'):
            # macOS Metal Performance Shaders
            if torch.backends.mps.is_available():
                gpu_info['available'] = True
                gpu_info['type'] = 'mps'
                gpu_info['name'] = 'Apple Silicon'
                
                print(f"\n🎮 GPU Detected (Apple MPS):")
                print(f"   Name: {gpu_info['name']}")
                return gpu_info
    except:
        pass
    
    print("\n💻 No dedicated GPU detected - will use CPU")
    return gpu_info

def check_memory():
    """Check system memory."""
    try:
        import psutil
        mem = psutil.virtual_memory()
        total_gb = mem.total / (1024**3)
        available_gb = mem.available / (1024**3)
        
        print(f"\n💾 Memory:")
        print(f"   Total: {total_gb:.2f} GB")
        print(f"   Available: {available_gb:.2f} GB")
        
        return {'total_gb': total_gb, 'available_gb': available_gb}
    except ImportError:
        print("\n⚠️  psutil not installed - skipping memory detection")
        return {'total_gb': 0, 'available_gb': 0}

def recommend_config(cpu_info, gpu_info, memory_info):
    """Recommend configuration based on hardware."""
    config = {
        'device': 'cpu',
        'threads': max(1, cpu_info['cores'] // 2),
        'max_memory_gb': max(2, int(memory_info.get('available_gb', 8) * 0.5)),
        'model_path': 'Qwen/Qwen2.5-1.5B-Instruct',
        'precision': 'float32',
        'max_new_tokens': 64
    }
    
    if gpu_info['available']:
        if gpu_info['type'] == 'cuda':
            config['device'] = 'cuda'
            config['cuda_device'] = 0
            
            # Recommend model based on VRAM
            vram = gpu_info['memory_gb']
            if vram >= 16:
                config['model_path'] = 'Qwen/Qwen2.5-VL-7B-Instruct'
                config['precision'] = 'float16'
                config['max_new_tokens'] = 512
                config['gpu_memory_limit_gb'] = int(vram * 0.8)
            elif vram >= 8:
                config['model_path'] = 'Qwen/Qwen2.5-VL-3B-Instruct'
                config['precision'] = 'float16'
                config['max_new_tokens'] = 256
                config['gpu_memory_limit_gb'] = int(vram * 0.8)
            elif vram >= 4:
                config['model_path'] = 'Qwen/Qwen2.5-3B-Instruct'
                config['precision'] = 'float16'
                config['max_new_tokens'] = 128
                config['gpu_memory_limit_gb'] = int(vram * 0.8)
            else:
                config['model_path'] = 'Qwen/Qwen2.5-1.5B-Instruct'
                config['precision'] = 'float32'
                config['max_new_tokens'] = 64
        
        elif gpu_info['type'] == 'mps':
            config['device'] = 'mps'
            config['model_path'] = 'Qwen/Qwen2.5-3B-Instruct'
            config['precision'] = 'float16'
            config['max_new_tokens'] = 256
    
    return config

def generate_config_file(config: dict, output_path: str):
    """Generate YAML configuration file."""
    yaml_content = f"""# AI-Viz-Lab Configuration
# Auto-generated by hardware_detect.py
# Generated for: {config['device'].upper()}

hardware:
  device: "{config['device']}"
  threads: {config['threads']}
  max_memory_gb: {config['max_memory_gb']}
"""
    
    if config['device'] == 'cuda':
        yaml_content += f"""  cuda_device: {config.get('cuda_device', 0)}
  gpu_memory_limit_gb: {config.get('gpu_memory_limit_gb', 6)}
"""
    
    yaml_content += f"""
model:
  path: "{config['model_path']}"
  precision: "{config['precision']}"
  max_new_tokens: {config['max_new_tokens']}

performance:
  enable_profiling: true
  log_to_file: false

ui:
  theme: "dark"
  animations: true
"""
    
    with open(output_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"\n✅ Configuration saved to: {output_path}")

def main():
    """Main detection routine."""
    parser = argparse.ArgumentParser(
        description="Detect hardware and generate configuration"
    )
    parser.add_argument(
        "--generate-config", "-g",
        action="store_true",
        help="Generate config.yaml based on detected hardware"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="config.yaml",
        help="Output path for generated config (default: config.yaml)"
    )
    
    args = parser.parse_args()
    
    print_header("🎓 AI-Viz-Lab: Hardware Detector")
    
    # Detect hardware
    cpu_info = detect_cpu()
    gpu_info = detect_gpu()
    memory_info = check_memory()
    
    # Generate recommendations
    config = recommend_config(cpu_info, gpu_info, memory_info)
    
    print(f"\n📋 Recommended Configuration:")
    print(f"   Device: {config['device']}")
    print(f"   Model: {config['model_path']}")
    print(f"   Precision: {config['precision']}")
    print(f"   Max Tokens: {config['max_new_tokens']}")
    
    if args.generate_config:
        output_path = args.output
        generate_config_file(config, output_path)
        
        print("\n✅ Next steps:")
        print(f"   1. Review and edit: {output_path}")
        print("   2. Download model: python scripts/download_model.py --model " + config['model_path'])
        print("   3. Launch app: python src/main.py")
    else:
        print("\n💡 To auto-generate config.yaml, run:")
        print("   python src/utils/hardware_detect.py --generate-config")

if __name__ == "__main__":
    main()
