"""
Lesson 05: Vision - Images meet text
Interactive demo showing patch encoding and joint attention for vision-language models
"""

import gradio as gr
import numpy as np
from typing import Tuple, List
import plotly.graph_objects as go
from PIL import Image
import io


class VisionDemo:
    """Demonstrates how images are processed in vision-language models"""
    
    def __init__(self):
        self.patch_size = 16
        self.num_patches = 196  # 14x14 grid
    
    def create_patch_grid(self, image: np.ndarray) -> Tuple[go.Figure, str]:
        """Simulate patch encoding of an image"""
        if image is None:
            # Create placeholder
            fig = go.Figure()
            fig.add_annotation(text="Upload an image to see patch encoding",
                             xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            return fig, "No image uploaded"
        
        # Resize to standard size for demo
        img_resized = Image.fromarray(image).resize((224, 224))
        img_array = np.array(img_resized)
        
        # Calculate patches
        h, w = img_array.shape[:2]
        patches_h = h // self.patch_size
        patches_w = w // self.patch_size
        
        # Create visualization
        fig = go.Figure()
        
        # Add original image
        fig.add_trace(go.Image(z=img_array))
        
        # Overlay grid
        for i in range(patches_h + 1):
            y = i * self.patch_size
            fig.add_shape(type='line',
                         x0=0, y0=y, x1=w, y1=y,
                         line=dict(color='yellow', width=1, dash='dot'))
        
        for j in range(patches_w + 1):
            x = j * self.patch_size
            fig.add_shape(type='line',
                         x0=x, y0=0, x1=x, y1=h,
                         line=dict(color='yellow', width=1, dash='dot'))
        
        # Highlight a sample patch
        sample_i, sample_j = 3, 3
        x0 = sample_j * self.patch_size
        y0 = sample_i * self.patch_size
        x1 = x0 + self.patch_size
        y1 = y0 + self.patch_size
        
        fig.add_shape(type='rect',
                     x0=x0, y0=y0, x1=x1, y1=y1,
                     fillcolor='rgba(255, 0, 0, 0.3)',
                     line=dict(color='red', width=3))
        
        fig.update_layout(
            title=f'Patch Encoding: {patches_h}×{patches_w} = {patches_h*patches_w} patches<br><span style="font-size: 12px">Each {self.patch_size}×{self.patch_size} patch becomes a token</span>',
            width=500,
            height=500,
            xaxis=dict(range=[0, w], showticklabels=False),
            yaxis=dict(range=[h, 0], showticklabels=False, scaleanchor="x")
        )
        
        info = f"""
        **Patch Encoding Analysis**
        
        📐 **Image Size**: {w}×{h} pixels
        🔲 **Patch Size**: {self.patch_size}×{self.patch_size} pixels
        📊 **Total Patches**: {patches_h * patches_w} ({patches_h}×{patches_w} grid)
        
        💡 **How it works**:
        1. Image divided into fixed-size patches
        2. Each patch flattened and linearly projected
        3. Patch embeddings fed to transformer
        4. Joint attention with text tokens
        
        🎯 **Highlighted**: Patch ({sample_i}, {sample_j}) shown in red
        """
        
        return fig, info
    
    def simulate_joint_attention(self, image: np.ndarray, text: str) -> Tuple[go.Figure, str]:
        """Simulate joint attention between image patches and text tokens"""
        if image is None or not text.strip():
            fig = go.Figure()
            fig.add_annotation(text="Upload image and enter text to see joint attention",
                             xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            return fig, "Need both image and text"
        
        # Tokenize text (simplified)
        words = text.split()[:10]  # Limit to 10 words
        num_words = len(words)
        
        # Simulated attention matrix (patches × words)
        patches_h = patches_w = 14
        num_patches = patches_h * patches_w
        
        # Create synthetic attention pattern
        np.random.seed(42)
        attention_matrix = np.random.rand(num_patches, num_words) * 0.3
        
        # Add some structure - make certain patches attend to certain words
        for i in range(min(num_patches, num_words)):
            attention_matrix[i, i % num_words] += 0.7
        
        # Normalize
        attention_matrix = attention_matrix / attention_matrix.max()
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=attention_matrix,
            x=words,
            y=[f'P{i}' for i in range(num_patches)],
            colorscale='Viridis',
            zmin=0,
            zmax=1
        ))
        
        fig.update_layout(
            title=f'Joint Attention Map: {num_patches} patches × {num_words} words',
            xaxis_title='Text Tokens',
            yaxis_title='Image Patches',
            width=700,
            height=500
        )
        
        info = f"""
        **Joint Attention Analysis**
        
        📝 **Text Tokens**: {num_words} words
        🖼️ **Image Patches**: {num_patches} patches
        🔗 **Attention Matrix**: {num_patches} × {num_words} = {num_patches * num_words} connections
        
        💡 **What this shows**:
        - Each cell shows how much a patch "attends to" a word
        - Brighter colors = stronger attention
        - Model learns to focus on relevant image regions for each word
        
        🎯 **Key Insight**:
        Vision-language models use the same attention mechanism
        for both text-text and image-text interactions!
        """
        
        return fig, info
    
    def process_image_and_text(self, image: np.ndarray, text: str) -> Tuple[go.Figure, str, go.Figure, str]:
        """Process both patch encoding and joint attention"""
        patch_fig, patch_info = self.create_patch_grid(image)
        attention_fig, attention_info = self.simulate_joint_attention(image, text)
        return patch_fig, patch_info, attention_fig, attention_info
    
    def create_interface(self) -> gr.Blocks:
        """Create Gradio interface for vision lesson"""
        with gr.Blocks(title="🖼️ Lesson 05: Vision-Language Models") as demo:
            gr.Markdown("""
            # 🖼️ Lesson 05: Vision - Images Meet Text
            
            Explore how modern AI models process images alongside text using **patch encoding** 
            and **joint attention** mechanisms—the foundation of models like CLIP, Flamingo, and GPT-4V!
            
            ### Key Concepts:
            - **Patch Encoding**: Images split into grid of tokens (like words)
            - **Joint Attention**: Unified attention over image + text tokens
            - **Multimodal Fusion**: Seamless integration of vision and language
            """)
            
            with gr.Row():
                with gr.Column(scale=1):
                    image_input = gr.Image(label="📷 Upload Image", type="numpy")
                    text_input = gr.Textbox(
                        label="📝 Enter Description",
                        placeholder="Describe the image...",
                        value="a cat sitting on a couch"
                    )
                    process_btn = gr.Button("🔍 Analyze", variant="primary")
                    
                    gr.Markdown("""
                    ### 💡 How Vision Transformers Work:
                    1. **Patchify**: Split image into fixed-size patches
                    2. **Embed**: Convert each patch to vector
                    3. **Attend**: Apply self-attention across all tokens
                    4. **Fuse**: Joint reasoning over image + text
                    """)
                
                with gr.Column(scale=2):
                    gr.Markdown("### 🔲 Step 1: Patch Encoding")
                    patch_plot = gr.Plot(label="Image Grid")
                    patch_info = gr.Markdown(label="Encoding Info")
                    
                    gr.Markdown("### 🔗 Step 2: Joint Attention")
                    attention_plot = gr.Plot(label="Attention Heatmap")
                    attention_info = gr.Markdown(label="Attention Analysis")
            
            # Event handlers
            process_btn.click(
                fn=self.process_image_and_text,
                inputs=[image_input, text_input],
                outputs=[patch_plot, patch_info, attention_plot, attention_info]
            )
            
            # Auto-run on load with placeholder
            demo.load(
                fn=self.process_image_and_text,
                inputs=[image_input, text_input],
                outputs=[patch_plot, patch_info, attention_plot, attention_info]
            )
        
        return demo


def launch():
    """Launch the vision lesson interface"""
    demo = VisionDemo()
    interface = demo.create_interface()
    interface.launch(server_name="0.0.0.0", server_port=7865, share=False)


if __name__ == "__main__":
    launch()
