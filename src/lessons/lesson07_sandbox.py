"""
Lesson 07: Sandbox - Free experimentation
Interactive playground to mix & match concepts and build custom AI pipelines
"""

import gradio as gr
import numpy as np
from typing import Dict, Any, List
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class SandboxDemo:
    """Interactive sandbox for experimenting with AI concepts"""
    
    def __init__(self):
        self.pipeline_history = []
        
    def create_token_visualization(self, text: str, vocab_size: int) -> go.Figure:
        """Visualize tokenization with configurable vocab size"""
        if not text:
            fig = go.Figure()
            fig.add_annotation(text="Enter text to tokenize",
                             xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            return fig
        
        # Simulate tokenization
        words = text.split()
        tokens = []
        token_ids = []
        
        for i, word in enumerate(words):
            # Simulate subword tokenization
            if len(word) > 5 and vocab_size < 10000:
                # Split long words for small vocab
                mid = len(word) // 2
                tokens.extend([word[:mid], word[mid:]])
                token_ids.extend([i * 2, i * 2 + 1])
            else:
                tokens.append(word)
                token_ids.append(i)
        
        # Create visualization
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=list(range(len(tokens))),
            y=[len(t) for t in tokens],
            text=tokens,
            textposition='outside',
            marker_color='steelblue',
            name='Tokens'
        ))
        
        fig.update_layout(
            title=f'Tokenization: {len(tokens)} tokens (vocab size: {vocab_size})',
            xaxis_title='Token Position',
            yaxis_title='Token Length',
            height=300
        )
        
        return fig
    
    def create_embedding_projection(self, num_words: int, dimensions: int) -> go.Figure:
        """Create 2D/3D projection of word embeddings"""
        np.random.seed(42)
        
        # Generate random embeddings
        embeddings = np.random.randn(num_words, dimensions)
        
        # Simple PCA-like projection to 2D
        if dimensions >= 2:
            projected = embeddings[:, :2]
        else:
            projected = np.column_stack([embeddings[:, 0], np.zeros(num_words)])
        
        # Create scatter plot
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=projected[:, 0],
            y=projected[:, 1],
            mode='markers+text',
            marker=dict(size=12, color='coral', opacity=0.7),
            text=[f'W{i}' for i in range(num_words)],
            textposition='top center',
            name='Words'
        ))
        
        fig.update_layout(
            title=f'Embedding Space: {num_words} words in {dimensions}D → 2D projection',
            xaxis_title='Dimension 1',
            yaxis_title='Dimension 2',
            height=400,
            showlegend=False
        )
        
        return fig
    
    def create_attention_matrix(self, seq_len: int, num_heads: int) -> go.Figure:
        """Generate attention heatmap with multiple heads"""
        np.random.seed(42)
        
        # Create simulated attention matrices for each head
        fig = make_subplots(
            rows=1, cols=num_heads,
            subplot_titles=[f'Head {i+1}' for i in range(num_heads)],
            horizontal_spacing=0.1
        )
        
        for head in range(num_heads):
            # Generate structured attention pattern
            attention = np.random.rand(seq_len, seq_len) * 0.3
            
            # Add diagonal bias (self-attention)
            attention += np.eye(seq_len) * 0.5
            
            # Add some structure
            for i in range(seq_len):
                if i > 0:
                    attention[i, i-1] += 0.4  # Attend to previous token
            
            attention = np.clip(attention, 0, 1)
            
            fig.add_trace(
                go.Heatmap(
                    z=attention,
                    colorscale='Viridis',
                    showscale=False,
                    x=list(range(seq_len)),
                    y=list(range(seq_len))
                ),
                row=1, col=head+1
            )
        
        fig.update_layout(
            title=f'Multi-Head Attention: {seq_len} tokens × {num_heads} heads',
            height=300
        )
        
        return fig
    
    def run_pipeline(self, text: str, vocab_size: int, embed_dim: int, 
                     num_heads: int, quant_bits: int) -> Dict[str, Any]:
        """Run a complete pipeline with all selected parameters"""
        results = {}
        
        # Step 1: Tokenization
        tokens_fig = self.create_token_visualization(text, vocab_size)
        num_tokens = len(text.split()) if text else 0
        
        # Step 2: Embeddings
        embed_fig = self.create_embedding_projection(num_tokens, embed_dim)
        
        # Step 3: Attention
        if num_tokens > 0:
            attention_fig = self.create_attention_matrix(min(num_tokens, 8), num_heads)
        else:
            attention_fig = self.create_attention_matrix(5, num_heads)
        
        # Calculate metrics
        model_size_estimate = (vocab_size * embed_dim * 4) / (1e9)  # GB
        if quant_bits < 32:
            model_size_estimate *= (quant_bits / 32)
        
        inference_speed = 1000 / (num_tokens * embed_dim * num_heads / 1000)
        if quant_bits < 32:
            inference_speed *= (32 / quant_bits)
        
        stats = f"""
        ### 📊 Pipeline Statistics
        
        **Configuration:**
        - Vocabulary Size: {vocab_size:,}
        - Embedding Dimensions: {embed_dim}
        - Attention Heads: {num_heads}
        - Quantization: {quant_bits}-bit
        
        **Estimates:**
        - Model Size: ~{model_size_estimate:.2f} GB
        - Tokens/sec: ~{inference_speed:.0f}
        - Memory: ~{model_size_estimate * 1.5:.2f} GB (with overhead)
        
        **Pipeline Flow:**
        ```
        Input Text → Tokenization ({num_tokens} tokens) 
                   → Embeddings ({embed_dim}D) 
                   → Attention ({num_heads} heads) 
                   → Output
        ```
        """
        
        return {
            'tokens_fig': tokens_fig,
            'embed_fig': embed_fig,
            'attention_fig': attention_fig,
            'stats': stats
        }
    
    def create_interface(self) -> gr.Blocks:
        """Create Gradio interface for sandbox lesson"""
        with gr.Blocks(title="🧪 Lesson 07: AI Sandbox") as demo:
            gr.Markdown("""
            # 🧪 Lesson 07: Sandbox - Build Your Own AI Pipeline
            
            **Experiment freely** with all the concepts you've learned! Mix and match parameters,
            visualize the effects, and understand how different choices impact model behavior.
            
            ### 🎛️ Configure Your Pipeline:
            Adjust parameters below and see real-time visualizations of each stage.
            """)
            
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### ⚙️ Pipeline Configuration")
                    
                    text_input = gr.Textbox(
                        label="Input Text",
                        placeholder="Enter your text here...",
                        value="The quick brown fox jumps over the lazy dog",
                        lines=3
                    )
                    
                    vocab_slider = gr.Slider(
                        minimum=1000, maximum=100000, step=1000,
                        value=30000, label="Vocabulary Size"
                    )
                    
                    embed_slider = gr.Slider(
                        minimum=64, maximum=1024, step=64,
                        value=512, label="Embedding Dimensions"
                    )
                    
                    heads_slider = gr.Slider(
                        minimum=1, maximum=16, step=1,
                        value=8, label="Attention Heads"
                    )
                    
                    quant_slider = gr.Slider(
                        minimum=4, maximum=32, step=4,
                        value=8, label="Quantization Bits",
                        info="4, 8, 16, or 32"
                    )
                    
                    run_btn = gr.Button("🚀 Run Pipeline", variant="primary", size="lg")
                    
                    gr.Markdown("""
                    ### 💡 Tips:
                    - Larger vocab → better coverage but bigger model
                    - More dimensions → richer representations
                    - More heads → capture different relationships
                    - Lower bits → faster but less precise
                    """)
                
                with gr.Column(scale=2):
                    gr.Markdown("### 🔢 Step 1: Tokenization")
                    tokens_plot = gr.Plot(label="Token Visualization")
                    
                    gr.Markdown("### 🧭 Step 2: Embedding Space")
                    embed_plot = gr.Plot(label="Embedding Projection")
                    
                    gr.Markdown("### 🎯 Step 3: Attention Mechanism")
                    attention_plot = gr.Plot(label="Attention Heatmaps")
                    
                    stats_output = gr.Markdown(label="Pipeline Statistics")
            
            # Event handlers
            run_btn.click(
                fn=lambda t, v, e, h, q: self.run_pipeline(t, v, e, h, q),
                inputs=[text_input, vocab_slider, embed_slider, heads_slider, quant_slider],
                outputs=[tokens_plot, embed_plot, attention_plot, stats_output]
            )
            
            # Auto-run on load
            demo.load(
                fn=lambda t, v, e, h, q: self.run_pipeline(t, v, e, h, q),
                inputs=[text_input, vocab_slider, embed_slider, heads_slider, quant_slider],
                outputs=[tokens_plot, embed_plot, attention_plot, stats_output]
            )
        
        return demo


def launch():
    """Launch the sandbox lesson interface"""
    demo = SandboxDemo()
    interface = demo.create_interface()
    interface.launch(server_name="0.0.0.0", server_port=7867, share=False)


if __name__ == "__main__":
    launch()
