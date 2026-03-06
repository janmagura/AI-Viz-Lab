"""
AI-Viz-Lab: Lesson 02 - Embeddings
Interactive visualization of word embeddings and semantic similarity.
"""

import numpy as np
from typing import List, Dict, Tuple
from sklearn.manifold import TSNE


class EmbeddingVisualizer:
    """Visualizes word embeddings in 2D/3D space."""
    
    def __init__(self, embedding_dim: int = 50):
        self.embedding_dim = embedding_dim
        self.word_embeddings = self._create_sample_embeddings()
    
    def _create_sample_embeddings(self) -> Dict[str, np.ndarray]:
        """Create sample word embeddings for demonstration."""
        np.random.seed(42)
        
        # Create semantic clusters
        animals = ['cat', 'dog', 'bird', 'fish', 'lion', 'tiger']
        colors = ['red', 'blue', 'green', 'yellow', 'orange', 'purple']
        actions = ['run', 'jump', 'walk', 'swim', 'fly', 'sleep']
        emotions = ['happy', 'sad', 'angry', 'excited', 'calm', 'nervous']
        
        embeddings = {}
        
        # Animals cluster (centered around [1, 0, ...])
        base_animal = np.zeros(self.embedding_dim)
        base_animal[0] = 1.0
        for word in animals:
            embeddings[word] = base_animal + np.random.randn(self.embedding_dim) * 0.3
        
        # Colors cluster (centered around [0, 1, ...])
        base_color = np.zeros(self.embedding_dim)
        base_color[1] = 1.0
        for word in colors:
            embeddings[word] = base_color + np.random.randn(self.embedding_dim) * 0.3
        
        # Actions cluster (centered around [0, 0, 1, ...])
        base_action = np.zeros(self.embedding_dim)
        base_action[2] = 1.0
        for word in actions:
            embeddings[word] = base_action + np.random.randn(self.embedding_dim) * 0.3
        
        # Emotions cluster (centered around [-1, 0, ...])
        base_emotion = np.zeros(self.embedding_dim)
        base_emotion[0] = -1.0
        for word in emotions:
            embeddings[word] = base_emotion + np.random.randn(self.embedding_dim) * 0.3
        
        return embeddings
    
    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def find_similar_words(self, word: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Find words most similar to the given word."""
        if word not in self.word_embeddings:
            return []
        
        target_vec = self.word_embeddings[word]
        similarities = []
        
        for other_word, other_vec in self.word_embeddings.items():
            if other_word != word:
                sim = self.cosine_similarity(target_vec, other_vec)
                similarities.append((other_word, sim))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]
    
    def reduce_dimensions(self, words: List[str] = None, n_components: int = 2) -> np.ndarray:
        """Reduce embeddings to 2D or 3D using t-SNE."""
        if words is None:
            words = list(self.word_embeddings.keys())
        
        # Get embeddings for selected words
        embeddings_matrix = np.array([self.word_embeddings[word] for word in words])
        
        # Apply t-SNE
        tsne = TSNE(n_components=n_components, random_state=42, perplexity=min(5, len(words)-1))
        reduced = tsne.fit_transform(embeddings_matrix)
        
        return reduced
    
    def get_embedding_data(self, word: str) -> Dict:
        """Get embedding data for a specific word."""
        if word not in self.word_embeddings:
            return {'error': f'Word "{word}" not in vocabulary'}
        
        vec = self.word_embeddings[word]
        
        return {
            'word': word,
            'embedding_dim': self.embedding_dim,
            'vector_norm': float(np.linalg.norm(vec)),
            'first_10_dims': vec[:10].tolist(),
            'similar_words': self.find_similar_words(word)
        }
    
    def create_visualization_data(self, n_components: int = 2) -> Dict:
        """Create data for interactive visualization."""
        words = list(self.word_embeddings.keys())
        reduced_coords = self.reduce_dimensions(words, n_components)
        
        # Assign colors based on semantic clusters
        animals = ['cat', 'dog', 'bird', 'fish', 'lion', 'tiger']
        colors_list = ['red', 'blue', 'green', 'yellow', 'orange', 'purple']
        actions = ['run', 'jump', 'walk', 'swim', 'fly', 'sleep']
        emotions = ['happy', 'sad', 'angry', 'excited', 'calm', 'nervous']
        
        point_colors = []
        for word in words:
            if word in animals:
                point_colors.append('#FF6B6B')  # Red
            elif word in colors_list:
                point_colors.append('#4ECDC4')  # Teal
            elif word in actions:
                point_colors.append('#45B7D1')  # Blue
            elif word in emotions:
                point_colors.append('#FFA07A')  # Light salmon
            else:
                point_colors.append('#95A5A6')  # Gray
        
        return {
            'words': words,
            'x_coords': reduced_coords[:, 0].tolist(),
            'y_coords': reduced_coords[:, 1].tolist(),
            'z_coords': reduced_coords[:, 2].tolist() if n_components == 3 else None,
            'colors': point_colors,
            'clusters': {
                'animals': animals,
                'colors': colors_list,
                'actions': actions,
                'emotions': emotions
            }
        }


def demo_embeddings():
    """Demonstrate embeddings with examples."""
    visualizer = EmbeddingVisualizer()
    
    print("\n🧭 Embeddings Demo")
    print("=" * 50)
    
    # Show similar words
    test_words = ['cat', 'red', 'run', 'happy']
    
    for word in test_words:
        print(f"\n📍 Similar to '{word}':")
        similar = visualizer.find_similar_words(word, top_k=3)
        for sim_word, score in similar:
            print(f"   - {sim_word}: {score:.3f}")
    
    # Get embedding info
    print("\n\n📊 Embedding Information:")
    data = visualizer.get_embedding_data('cat')
    print(f"   Word: {data['word']}")
    print(f"   Dimension: {data['embedding_dim']}")
    print(f"   Vector norm: {data['vector_norm']:.3f}")
    
    return visualizer.create_visualization_data()


if __name__ == "__main__":
    demo_embeddings()
