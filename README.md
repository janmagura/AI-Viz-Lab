# AI-Viz-Lab
🎓 AI-Viz-Lab: Interactive AI Education Platform A complete, GitHub-ready educational application that teaches how AI works through interactive visualizations, live demonstrations, and hands-on experiments — all running locally on your PC!
### Interactive Visual Education for Understanding How AI Works

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Platform: Windows/Linux/macOS](https://img.shields.io/badge/Platform-Windows%2FLinux%2FmacOS-lightgrey.svg)](#installation)
[![Models: Qwen, Llama, Mistral](https://img.shields.io/badge/Models-Qwen%7CLlama%7CMistral-orange.svg)](#supported-models)

> 🧠 **Learn AI by seeing it work** — No cloud, no fees, just your local PC!

![Demo Screenshot](docs/images/demo_screenshot.png)

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🎯 **7 Interactive Lessons** | Tokens → Embeddings → Attention → Vision → Quantization |
| 🎨 **Live Visualizations** | 2D/3D plots, heatmaps, animated flows |
| ⚙️ **Hardware Adaptive** | Auto-detects CPU/GPU, suggests optimal settings |
| 🔧 **Fully Configurable** | Adjust model, precision, max tokens via YAML or UI |
| 💾 **100% Local** | No internet required after installation |
| 📊 **Performance Monitor** | Real-time tokens/sec, memory usage, latency |
| 📤 **Export Results** | Save visualizations, chat logs, experiment data |
| 🌍 **Multilingual Support** | Lessons available in English, Chinese, Spanish |

## 📚 Lesson Overview
| Lesson	| Topic	| Key | Concepts	| Interactive Demo
🔢 **01: Tokens**	How text becomes numbers	Tokenization, BPE, vocab size	✏️ Type text → see tokens/IDs
🧭 **02: Embeddings**	Words as vectors	Cosine similarity, meaning space	🌐 2D/3D plot of word clusters
🎯 **03: Attention**	How models "focus"	Q×Kᵀ×V, multi-head, layers	🔥 Animated attention heatmaps
🌍 **04: Multilingual**	One space, many languages	Cross-lingual alignment	🗣️ Compare English/Chinese vectors
🖼️ **05: Vision**	Images meet text	Patch encoding, joint attention	🖼️ Upload image → see token flow
⚡ **06: Quantization**	Speed vs precision	int8/int4, GGUF, tradeoffs	📊 Benchmark different precisions
🧪 **07: Sandbox**	Free experimentation	Mix & match concepts	🎛️ Build your own AI pipeline

## 🚀 Quick Start
### 1️⃣ Install Dependencies
```bash
# Clone the repository
git clone https://github.com/janmagura/AI-Viz-Lab.git
cd AI-Viz-Lab

# Run the auto-installer (recommended)
python scripts/install_deps.py

# Or manually:
pip install -r requirements.txt

### 2️⃣ Configure for Your Hardware
```bash
# Auto-detect and generate config
python src/utils/hardware_detect.py --generate-config

# Or edit manually:
notepad config.yaml

### 3️⃣ Download a Model (Optional)
```bash
# Download Qwen2.5-VL-3B (default)
python scripts/download_model.py --model Qwen/Qwen2.5-VL-3B-Instruct

# Or use a smaller model for testing:
python scripts/download_model.py --model Qwen/Qwen2.5-1.5B-Instruct

### 4️⃣ Launch the Application
```bash
python src/main.py

