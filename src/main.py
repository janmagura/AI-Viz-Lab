"""
AI-Viz-Lab: Main Application Entry Point
Launches the Gradio-based interactive UI.
"""

import gradio as gr
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from utils.config import get_config
from lessons.lesson01_tokens import TokenVisualizer, demo_tokenize
from lessons.lesson02_embeddings import EmbeddingVisualizer, demo_embeddings
from lessons.lesson03_attention import AttentionVisualizer, demo_attention


def create_lesson_tokens(text: str):
    """Lesson 1: Tokenization demo."""
    visualizer = TokenVisualizer()
    result = visualizer.create_visualization_data(text)
    
    output_text = f"Original: {result['original_text']}\n"
    output_text += f"Tokens: {len(result['tokens'])}\n\n"
    
    for i, (token, token_id) in enumerate(zip(result['tokens'], result['token_ids'])):
        output_text += f"[{i}] '{token}' → ID: {token_id}\n"
    
    return output_text


def create_lesson_embeddings(word: str):
    """Lesson 2: Embeddings demo."""
    visualizer = EmbeddingVisualizer()
    
    if word.lower() not in visualizer.word_embeddings:
        return f"Word '{word}' not in demo vocabulary. Try: cat, dog, red, blue, run, jump, happy, sad"
    
    similar = visualizer.find_similar_words(word.lower(), top_k=5)
    
    output_text = f"Embedding analysis for '{word}':\n\n"
    output_text += "Most similar words:\n"
    
    for sim_word, score in similar:
        output_text += f"  - {sim_word}: {score:.3f}\n"
    
    return output_text


def create_lesson_attention(text: str):
    """Lesson 3: Attention demo."""
    visualizer = AttentionVisualizer()
    words = text.split()[:8]
    
    if len(words) < 2:
        return "Please enter at least 2 words."
    
    result = visualizer.compute_attention_demo(0, words)
    
    output_text = f"Attention from '{result['query_word']}':\n\n"
    
    for word, weight in zip(words, result['attention_weights']):
        bar = '█' * int(weight * 20)
        output_text += f"{word:10s} {weight:.3f} {bar}\n"
    
    return output_text


def create_main_interface():
    """Create the main Gradio interface."""
    
    with gr.Blocks(title="AI-Viz-Lab", theme=gr.themes.Soft()) as app:
        gr.Markdown("""
        # 🎓 AI-Viz-Lab: Interactive AI Education Platform
        
        Learn how AI works through interactive visualizations!
        """)
        
        with gr.Tabs():
            # Lesson 1: Tokens
            with gr.TabItem("🔢 Lesson 1: Tokens"):
                gr.Markdown("""
                ## How Text Becomes Numbers
                
                AI models don't read text directly - they convert it to numbers called **tokens**.
                Type some text below to see how it gets tokenized!
                """)
                
                with gr.Row():
                    token_input = gr.Textbox(
                        label="Enter text",
                        value="The quick brown fox jumps",
                        lines=2
                    )
                    token_output = gr.Textbox(
                        label="Tokenization Result",
                        lines=10
                    )
                
                token_btn = gr.Button("Tokenize", variant="primary")
                token_btn.click(
                    fn=create_lesson_tokens,
                    inputs=[token_input],
                    outputs=[token_output]
                )
            
            # Lesson 2: Embeddings
            with gr.TabItem("🧭 Lesson 2: Embeddings"):
                gr.Markdown("""
                ## Words as Vectors
                
                Words are represented as vectors in high-dimensional space.
                Similar words have similar vectors!
                
                Try: cat, dog, red, blue, run, jump, happy, sad
                """)
                
                with gr.Row():
                    embed_input = gr.Textbox(
                        label="Enter a word",
                        value="cat",
                        lines=1
                    )
                    embed_output = gr.Textbox(
                        label="Similar Words",
                        lines=10
                    )
                
                embed_btn = gr.Button("Find Similar Words", variant="primary")
                embed_btn.click(
                    fn=create_lesson_embeddings,
                    inputs=[embed_input],
                    outputs=[embed_output]
                )
            
            # Lesson 3: Attention
            with gr.TabItem("🎯 Lesson 3: Attention"):
                gr.Markdown("""
                ## How Models "Focus"
                
                Attention mechanisms let models focus on relevant parts of the input.
                See how each word pays attention to other words!
                """)
                
                with gr.Row():
                    attention_input = gr.Textbox(
                        label="Enter text (max 8 words)",
                        value="The cat sits on the mat",
                        lines=2
                    )
                    attention_output = gr.Textbox(
                        label="Attention Weights",
                        lines=10
                    )
                
                attention_btn = gr.Button("Show Attention", variant="primary")
                attention_btn.click(
                    fn=create_lesson_attention,
                    inputs=[attention_input],
                    outputs=[attention_output]
                )
            
            # Info tab
            with gr.TabItem("ℹ️ About"):
                gr.Markdown("""
                ### About AI-Viz-Lab
                
                This interactive platform teaches AI concepts through visualization:
                
                - **Tokens**: How text becomes numbers
                - **Embeddings**: Words as vectors in space
                - **Attention**: How models focus on information
                - **Vision**: Images meet text
                - **Quantization**: Speed vs precision tradeoffs
                
                ### Configuration
                
                Edit `config.yaml` to customize:
                - Model selection
                - Hardware settings
                - Performance options
                
                ### Learn More
                
                Check out the README.md for detailed documentation.
                """)
    
    return app


def main():
    """Main entry point."""
    print("\n🎓 AI-Viz-Lab: Starting Application...")
    
    # Load configuration
    config = get_config()
    config.validate()
    
    # Create and launch interface
    app = create_main_interface()
    
    print("✅ Application ready!")
    print("   Opening in browser...")
    
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )


if __name__ == "__main__":
    main()
