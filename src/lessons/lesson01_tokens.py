"""
AI-Viz-Lab: Lesson 01 - Tokens
Interactive visualization of how text becomes tokens.
"""

import numpy as np
from typing import List, Tuple, Dict


class TokenVisualizer:
    """Visualizes tokenization process."""
    
    def __init__(self, vocab_size: int = 1000):
        self.vocab_size = vocab_size
        self.sample_vocab = self._create_sample_vocab()
    
    def _create_sample_vocab(self) -> Dict[str, int]:
        """Create a sample vocabulary for demonstration."""
        # Common words and their token IDs
        vocab = {
            '<pad>': 0, '<unk>': 1, '<bos>': 2, '<eos>': 3,
            'the': 10, 'a': 11, 'an': 12, 'and': 13, 'or': 14,
            'cat': 20, 'dog': 21, 'bird': 22, 'fish': 23,
            'hello': 30, 'world': 31, 'ai': 32, 'model': 33,
            'token': 40, 'embedding': 41, 'attention': 42,
            'I': 50, 'you': 51, 'we': 52, 'they': 53,
            'is': 60, 'are': 61, 'was': 62, 'were': 63,
            'run': 70, 'running': 71, 'ran': 72,
            'quick': 80, 'brown': 81, 'fox': 82,
            'jumps': 90, 'over': 91, 'lazy': 92,
        }
        return vocab
    
    def tokenize_text(self, text: str) -> List[Tuple[str, int]]:
        """Tokenize text into (token, id) pairs."""
        # Simple whitespace tokenization for demo
        words = text.lower().split()
        tokens = []
        
        for word in words:
            # Remove punctuation for simplicity
            clean_word = ''.join(c for c in word if c.isalnum())
            if clean_word:
                token_id = self.sample_vocab.get(clean_word, len(self.sample_vocab) + len(tokens))
                tokens.append((clean_word, token_id))
        
        return tokens
    
    def get_token_data(self, text: str) -> Dict:
        """Get comprehensive token data for visualization."""
        tokens = self.tokenize_text(text)
        
        return {
            'original_text': text,
            'tokens': [t[0] for t in tokens],
            'token_ids': [t[1] for t in tokens],
            'num_tokens': len(tokens),
            'vocab_size': self.vocab_size
        }
    
    def create_visualization_data(self, text: str) -> Dict:
        """Create data for interactive visualization."""
        data = self.get_token_data(text)
        
        # Create bar chart data
        x_positions = list(range(len(data['tokens'])))
        
        return {
            **data,
            'x_positions': x_positions,
            'bar_heights': data['token_ids'],
            'colors': ['#4CAF50' if tid < 50 else '#FF9800' for tid in data['token_ids']]
        }


def demo_tokenize(text: str = "The quick brown fox jumps over the lazy dog"):
    """Demonstrate tokenization with example text."""
    visualizer = TokenVisualizer()
    result = visualizer.create_visualization_data(text)
    
    print("\n🔢 Tokenization Demo")
    print("=" * 50)
    print(f"Original text: {result['original_text']}")
    print(f"Number of tokens: {result['num_tokens']}")
    print(f"\nTokens and IDs:")
    
    for i, (token, token_id) in enumerate(zip(result['tokens'], result['token_ids'])):
        print(f"  [{i}] '{token}' → ID: {token_id}")
    
    return result


if __name__ == "__main__":
    demo_tokenize()
