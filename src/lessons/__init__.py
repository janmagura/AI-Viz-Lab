"""
AI-Viz-Lab: Lessons Package
Interactive lesson modules for AI education.
"""

from .lesson01_tokens import TokenVisualizer, demo_tokenize
from .lesson02_embeddings import EmbeddingVisualizer, demo_embeddings
from .lesson03_attention import AttentionVisualizer, demo_attention
from .lesson04_multilingual import MultilingualDemo
from .lesson05_vision import VisionDemo
from .lesson06_quantization import QuantizationDemo
from .lesson07_sandbox import SandboxDemo

__all__ = [
    'TokenVisualizer', 'demo_tokenize',
    'EmbeddingVisualizer', 'demo_embeddings',
    'AttentionVisualizer', 'demo_attention',
    'MultilingualDemo',
    'VisionDemo',
    'QuantizationDemo',
    'SandboxDemo'
]
