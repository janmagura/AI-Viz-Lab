"""
AI-Viz-Lab: Lesson 03 - Attention
Interactive visualization of attention mechanisms.
"""

import numpy as np
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt


class AttentionVisualizer:
    """Visualizes attention mechanism with Q×Kᵀ×V computation."""
    
    def __init__(self, num_heads: int = 4, hidden_dim: int = 64):
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
    
    def create_sample_attention(self, words: List[str]) -> Dict:
        """Create sample attention weights for demonstration."""
        n_words = len(words)
        
        # Create synthetic attention patterns
        attention_weights = np.zeros((n_words, n_words))
        
        # Simulate different attention patterns
        for i in range(n_words):
            for j in range(n_words):
                # Self-attention gets higher weight
                if i == j:
                    attention_weights[i, j] = 0.3
                # Adjacent words get medium weight
                elif abs(i - j) == 1:
                    attention_weights[i, j] = 0.2
                # First word (often subject) gets attention
                elif j == 0:
                    attention_weights[i, j] = 0.15
                # Last word (often object) gets some attention
                elif j == n_words - 1:
                    attention_weights[i, j] = 0.1
                else:
                    attention_weights[i, j] = 0.05
        
        # Normalize rows to sum to 1
        attention_weights = attention_weights / attention_weights.sum(axis=1, keepdims=True)
        
        return {
            'words': words,
            'attention_matrix': attention_weights,
            'num_heads': self.num_heads
        }
    
    def compute_attention_demo(self, query_idx: int, words: List[str]) -> Dict:
        """Demonstrate attention computation for a specific word."""
        n_words = len(words)
        
        # Create synthetic Q, K, V matrices
        np.random.seed(42)
        Q = np.random.randn(n_words, self.hidden_dim)
        K = np.random.randn(n_words, self.hidden_dim)
        V = np.random.randn(n_words, self.hidden_dim)
        
        # Compute attention scores: Q × Kᵀ
        query_vec = Q[query_idx:query_idx+1]  # Shape: (1, hidden_dim)
        key_matrix = K.T  # Shape: (hidden_dim, n_words)
        
        # Raw attention scores
        attention_scores = np.dot(query_vec, key_matrix) / np.sqrt(self.hidden_dim)
        
        # Apply softmax
        attention_weights = np.exp(attention_scores) / np.sum(np.exp(attention_scores))
        
        # Compute output: weights × V
        output = np.dot(attention_weights, V)
        
        return {
            'query_word': words[query_idx],
            'attention_scores': attention_scores.flatten(),
            'attention_weights': attention_weights.flatten(),
            'output_vector': output.flatten(),
            'words': words
        }
    
    def create_heatmap_data(self, text: str) -> Dict:
        """Create data for attention heatmap visualization."""
        words = text.split()[:10]  # Limit to 10 words for clarity
        
        if len(words) < 2:
            words = ['The', 'quick', 'brown', 'fox', 'jumps']
        
        attention_data = self.create_sample_attention(words)
        
        return {
            'text': text,
            'words': words,
            'attention_matrix': attention_data['attention_matrix'].tolist(),
            'matrix_shape': list(attention_data['attention_matrix'].shape)
        }
    
    def animate_attention_flow(self, words: List[str]) -> List[Dict]:
        """Create frames for attention flow animation."""
        frames = []
        n_words = len(words)
        
        for step in range(n_words):
            frame_data = self.compute_attention_demo(step, words)
            
            frames.append({
                'step': step,
                'focus_word': words[step],
                'weights': frame_data['attention_weights'].tolist(),
                'scores': frame_data['attention_scores'].tolist()
            })
        
        return frames


def demo_attention(text: str = "The quick brown fox jumps over the lazy dog"):
    """Demonstrate attention mechanism."""
    visualizer = AttentionVisualizer()
    words = text.split()[:8]  # Use first 8 words
    
    print("\n🎯 Attention Mechanism Demo")
    print("=" * 50)
    print(f"Text: {' '.join(words)}\n")
    
    # Show attention from one word
    query_idx = 0
    result = visualizer.compute_attention_demo(query_idx, words)
    
    print(f"Query word: '{result['query_word']}'")
    print(f"\nAttention weights (what '{result['query_word']}' pays attention to):")
    
    for word, weight in zip(words, result['attention_weights']):
        bar = '█' * int(weight * 20)
        print(f"   {word:10s} {weight:.3f} {bar}")
    
    # Create heatmap data
    print("\n\n📊 Attention Matrix:")
    heatmap_data = visualizer.create_heatmap_data(text)
    
    return heatmap_data


if __name__ == "__main__":
    demo_attention()
