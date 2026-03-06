"""
Lesson 04: Multilingual - One space, many languages
Interactive demo comparing English/Chinese vectors in shared embedding space
"""

import gradio as gr
import numpy as np
from typing import Tuple, List
import plotly.graph_objects as go


class MultilingualDemo:
    """Demonstrates cross-lingual alignment in embedding spaces"""
    
    def __init__(self):
        # Simulated multilingual embeddings (in real app, use LaBSE or similar)
        self.word_pairs = {
            'hello': ('你好', [0.8, 0.2, 0.1], [0.75, 0.25, 0.15]),
            'world': ('世界', [0.6, 0.7, 0.3], [0.65, 0.68, 0.28]),
            'love': ('爱', [0.3, 0.8, 0.5], [0.35, 0.78, 0.48]),
            'peace': ('和平', [0.5, 0.6, 0.7], [0.52, 0.58, 0.72]),
            'friend': ('朋友', [0.7, 0.4, 0.6], [0.68, 0.42, 0.58]),
            'water': ('水', [0.2, 0.3, 0.9], [0.22, 0.28, 0.88]),
            'fire': ('火', [0.9, 0.1, 0.2], [0.88, 0.12, 0.18]),
            'sun': ('太阳', [0.85, 0.5, 0.3], [0.82, 0.52, 0.28]),
        }
    
    def calculate_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        v1 = np.array(vec1)
        v2 = np.array(vec2)
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    
    def create_comparison_plot(self, english_word: str) -> go.Figure:
        """Create 3D plot showing English-Chinese word pair alignment"""
        if english_word.lower() not in self.word_pairs:
            # Create empty plot with message
            fig = go.Figure()
            fig.add_annotation(text="Word not in vocabulary<br>Try: hello, world, love, peace, friend, water, fire, sun",
                             xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            return fig
        
        en_word = english_word.lower()
        cn_word, en_vec, cn_vec = self.word_pairs[en_word]
        
        # Create 3D scatter plot
        fig = go.Figure()
        
        # Add English word
        fig.add_trace(go.Scatter3d(
            x=[en_vec[0]], y=[en_vec[1]], z=[en_vec[2]],
            mode='markers+text',
            marker=dict(size=10, color='blue', symbol='circle'),
            text=[f'EN: {en_word}'],
            textposition='top center',
            name='English'
        ))
        
        # Add Chinese word
        fig.add_trace(go.Scatter3d(
            x=[cn_vec[0]], y=[cn_vec[1]], z=[cn_vec[2]],
            mode='markers+text',
            marker=dict(size=10, color='red', symbol='circle'),
            text=[f'CN: {cn_word}'],
            textposition='top center',
            name='Chinese'
        ))
        
        # Add line connecting the pair
        fig.add_trace(go.Scatter3d(
            x=[en_vec[0], cn_vec[0]],
            y=[en_vec[1], cn_vec[1]],
            z=[en_vec[2], cn_vec[2]],
            mode='lines',
            line=dict(color='gray', width=3, dash='dash'),
            name='Alignment',
            showlegend=False
        ))
        
        # Calculate and display similarity
        similarity = self.calculate_similarity(en_vec, cn_vec)
        
        fig.update_layout(
            title=f'Multilingual Alignment: "{en_word}" ↔ "{cn_word}"<br><span style="font-size: 14px">Cosine Similarity: {similarity:.3f}</span>',
            scene=dict(
                xaxis=dict(title='Dimension 1', range=[0, 1]),
                yaxis=dict(title='Dimension 2', range=[0, 1]),
                zaxis=dict(title='Dimension 3', range=[0, 1]),
                aspectmode='cube'
            ),
            height=500,
            legend=dict(x=0.02, y=0.98)
        )
        
        return fig
    
    def compare_words(self, english_word: str, custom_chinese: str = "") -> Tuple[str, go.Figure, str]:
        """Compare English word with its Chinese counterpart"""
        en_word = english_word.lower().strip()
        
        if not en_word:
            return "Please enter an English word", go.Figure(), ""
        
        if en_word not in self.word_pairs:
            available = ', '.join(self.word_pairs.keys())
            return f"Word '{en_word}' not in demo vocabulary.<br>Available: {available}", go.Figure(), ""
        
        cn_word, en_vec, cn_vec = self.word_pairs[en_word]
        similarity = self.calculate_similarity(en_vec, cn_vec)
        
        # Create visualization
        fig = self.create_comparison_plot(en_word)
        
        # Generate explanation
        explanation = f"""
        **Cross-Lingual Alignment Analysis**
        
        🔤 **Word Pair**: {en_word} (English) ↔ {cn_word} (Chinese)
        
        📊 **Vector Similarity**: {similarity:.3f}
        - Values closer to 1.0 indicate stronger semantic alignment
        - Modern multilingual models achieve 0.7-0.9 for common concepts
        
        🌍 **Key Concepts**:
        • **Shared Embedding Space**: Both languages mapped to same vector space
        • **Semantic Preservation**: Meaning preserved across translation
        • **Cultural Nuances**: Some concepts may have lower alignment
        
        💡 **Real Applications**:
        - Machine translation without parallel corpora
        - Cross-lingual search engines
        - Multilingual chatbots
        """
        
        return f"✅ Analyzed: {en_word} ↔ {cn_word}", fig, explanation
    
    def create_interface(self) -> gr.Blocks:
        """Create Gradio interface for multilingual lesson"""
        with gr.Blocks(title="🌍 Lesson 04: Multilingual Embeddings") as demo:
            gr.Markdown("""
            # 🌍 Lesson 04: Multilingual - One Space, Many Languages
            
            Discover how AI models represent different languages in a **shared embedding space**.
            Words with similar meanings across languages are positioned close together!
            
            ### Key Concepts:
            - **Cross-lingual Alignment**: Mapping different languages to same vector space
            - **Semantic Preservation**: Meaning transcends language barriers
            - **Zero-shot Translation**: Translate without direct training pairs
            """)
            
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 🔤 Word Comparison")
                    english_input = gr.Textbox(
                        label="Enter English Word",
                        placeholder="Try: hello, world, love, peace, friend, water, fire, sun",
                        value="hello"
                    )
                    chinese_input = gr.Textbox(
                        label="Custom Chinese (optional)",
                        placeholder="Leave empty for automatic pairing",
                        visible=False  # Hidden for this demo
                    )
                    compare_btn = gr.Button("🔍 Compare Words", variant="primary")
                    
                    gr.Markdown("""
                    ### 📚 Available Words:
                    - hello → 你好
                    - world → 世界  
                    - love → 爱
                    - peace → 和平
                    - friend → 朋友
                    - water → 水
                    - fire → 火
                    - sun → 太阳
                    """)
                
                with gr.Column(scale=2):
                    output_plot = gr.Plot(label="Embedding Space Visualization")
            
            with gr.Row():
                status_output = gr.Textbox(label="Status", interactive=False)
            
            with gr.Row():
                explanation_output = gr.Markdown(label="Analysis")
            
            # Event handlers
            compare_btn.click(
                fn=self.compare_words,
                inputs=[english_input, chinese_input],
                outputs=[status_output, output_plot, explanation_output]
            )
            
            # Auto-run on load
            demo.load(
                fn=self.compare_words,
                inputs=[english_input, chinese_input],
                outputs=[status_output, output_plot, explanation_output]
            )
        
        return demo


def launch():
    """Launch the multilingual lesson interface"""
    demo = MultilingualDemo()
    interface = demo.create_interface()
    interface.launch(server_name="0.0.0.0", server_port=7864, share=False)


if __name__ == "__main__":
    launch()
