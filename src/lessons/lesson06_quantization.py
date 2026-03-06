"""
Lesson 06: Quantization - Speed vs precision
Interactive demo benchmarking different quantization precisions (int8/int4, GGUF)
"""

import gradio as gr
import numpy as np
from typing import Tuple, Dict
import plotly.graph_objects as go
import time


class QuantizationDemo:
    """Demonstrates quantization tradeoffs between speed and precision"""
    
    def __init__(self):
        self.benchmark_results = {}
        # Simulated model sizes (in GB)
        self.model_size_fp32 = 14.0
        self.model_size_fp16 = 7.0
        
    def simulate_quantization(self, value: float, bits: int) -> Tuple[float, float]:
        """Simulate quantization and dequantization with error"""
        if bits == 32:
            return value, 0.0
        elif bits == 16:
            # FP16 simulation
            quantized = np.float16(value)
            dequantized = float(quantized)
            error = abs(value - dequantized)
            return dequantized, error
        elif bits == 8:
            # INT8 simulation
            scale = 127.0
            quantized = int(np.clip(value * scale, -128, 127))
            dequantized = quantized / scale
            error = abs(value - dequantized)
            return dequantized, error
        elif bits == 4:
            # INT4 simulation
            scale = 7.0
            quantized = int(np.clip(value * scale, -8, 7))
            dequantized = quantized / scale
            error = abs(value - dequantized)
            return dequantized, error
        else:
            return value, 0.0
    
    def benchmark_precision(self, precision: str) -> Dict[str, float]:
        """Simulate benchmark for different precisions"""
        benchmarks = {
            'FP32': {'size_gb': 14.0, 'speed': 1.0, 'accuracy': 100.0, 'memory': 100.0},
            'FP16': {'size_gb': 7.0, 'speed': 1.8, 'accuracy': 99.9, 'memory': 52.0},
            'INT8': {'size_gb': 3.5, 'speed': 3.2, 'accuracy': 99.5, 'memory': 26.0},
            'INT4': {'size_gb': 1.75, 'speed': 5.5, 'accuracy': 98.0, 'memory': 13.0},
            'GGUF-Q4_K_M': {'size_gb': 2.1, 'speed': 4.8, 'accuracy': 98.5, 'memory': 15.0},
            'GGUF-Q8_0': {'size_gb': 4.2, 'speed': 2.8, 'accuracy': 99.7, 'memory': 30.0},
        }
        return benchmarks.get(precision, benchmarks['FP32'])
    
    def create_comparison_chart(self) -> go.Figure:
        """Create comprehensive comparison chart of all precisions"""
        precisions = ['FP32', 'FP16', 'INT8', 'INT4', 'GGUF-Q4_K_M', 'GGUF-Q8_0']
        
        sizes = [self.benchmark_precision(p)['size_gb'] for p in precisions]
        speeds = [self.benchmark_precision(p)['speed'] for p in precisions]
        accuracies = [self.benchmark_precision(p)['accuracy'] for p in precisions]
        
        # Create subplots
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=('Model Size (GB)', 'Relative Speed', 'Accuracy (%)'),
            specs=[[{"type": "bar"}, {"type": "bar"}, {"type": "bar"}]]
        )
        
        # Size bar chart
        fig.add_trace(
            go.Bar(x=precisions, y=sizes, name='Size', marker_color='steelblue'),
            row=1, col=1
        )
        
        # Speed bar chart
        fig.add_trace(
            go.Bar(x=precisions, y=speeds, name='Speed', marker_color='forestgreen'),
            row=1, col=2
        )
        
        # Accuracy bar chart
        fig.add_trace(
            go.Bar(x=precisions, y=accuracies, name='Accuracy', marker_color='coral'),
            row=1, col=3
        )
        
        fig.update_layout(
            title='Quantization Tradeoffs: Size vs Speed vs Accuracy',
            height=400,
            showlegend=False
        )
        
        return fig
    
    def analyze_precision(self, precision: str) -> Tuple[go.Figure, str]:
        """Analyze a specific precision level"""
        metrics = self.benchmark_precision(precision)
        
        # Create gauge chart for accuracy
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=metrics['accuracy'],
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': f"{precision} Accuracy", 'font': {'size': 24}},
            delta={'reference': 100, 'increasing': None, 'decreasing': {'color': "red"}},
            gauge={
                'axis': {'range': [95, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [95, 97], 'color': "lightgray"},
                    {'range': [97, 99], 'color': "gray"},
                    {'range': [99, 100], 'color': "lightgray"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 98
                }
            }
        ))
        
        fig.update_layout(height=300)
        
        analysis = f"""
        ### 📊 {precision} Analysis
        
        **Key Metrics:**
        - 💾 **Model Size**: {metrics['size_gb']:.2f} GB ({metrics['size_gb']/14.0*100:.1f}% of FP32)
        - ⚡ **Speed**: {metrics['speed']:.1f}× faster than FP32
        - 🎯 **Accuracy**: {metrics['accuracy']:.1f}% (vs FP32 baseline)
        - 🧠 **Memory**: {metrics['memory']:.1f}% of FP32 memory usage
        
        **Tradeoff Summary:**
        """
        
        if 'INT4' in precision or 'Q4' in precision:
            analysis += """
            ✅ **Best for**: Edge devices, mobile deployment, low-memory systems
            ⚠️ **Watch for**: Slight accuracy drop on complex tasks
            💡 **Tip**: Use Q4_K_M variant for better quality/size balance
            """
        elif 'INT8' in precision or 'Q8' in precision:
            analysis += """
            ✅ **Best for**: Production servers, good quality/speed balance
            ⚠️ **Watch for**: Minimal accuracy loss, often imperceptible
            💡 **Tip**: Excellent default choice for most applications
            """
        elif 'FP16' in precision:
            analysis += """
            ✅ **Best for**: GPU inference, high-quality generation
            ⚠️ **Watch for**: Still requires significant VRAM
            💡 **Tip**: Use when quality is paramount and resources available
            """
        else:  # FP32
            analysis += """
            ✅ **Best for**: Training, research, maximum precision needs
            ⚠️ **Watch for**: Very high memory requirements, slow inference
            💡 **Tip**: Rarely needed for inference; consider FP16 or lower
            """
        
        return fig, analysis
    
    def compare_two_precisions(self, prec1: str, prec2: str) -> Tuple[go.Figure, str]:
        """Compare two precision levels side by side"""
        m1 = self.benchmark_precision(prec1)
        m2 = self.benchmark_precision(prec2)
        
        # Create comparison bar chart
        metrics_names = ['Size\\n(GB)', 'Speed\\n(relative)', 'Accuracy\\n(%)']
        metrics_vals1 = [m1['size_gb'], m1['speed'], m1['accuracy']]
        metrics_vals2 = [m2['size_gb'], m2['speed'], m2['accuracy']]
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name=prec1,
            x=metrics_names,
            y=metrics_vals1,
            marker_color='steelblue'
        ))
        
        fig.add_trace(go.Bar(
            name=prec2,
            x=metrics_names,
            y=metrics_vals2,
            marker_color='coral'
        ))
        
        fig.update_layout(
            title=f'Comparison: {prec1} vs {prec2}',
            barmode='group',
            height=400,
            yaxis_title='Value'
        )
        
        # Calculate improvements
        size_improvement = (m1['size_gb'] - m2['size_gb']) / m1['size_gb'] * 100
        speed_improvement = (m2['speed'] - m1['speed']) / m1['speed'] * 100
        accuracy_change = m2['accuracy'] - m1['accuracy']
        
        summary = f"""
        ### 🔄 Head-to-Head Comparison
        
        **{prec1} → {prec2}:**
        
        📈 **Changes:**
        - 💾 Model Size: {size_improvement:+.1f}% ({'smaller' if size_improvement > 0 else 'larger'})
        - ⚡ Inference Speed: {speed_improvement:+.1f}% ({'faster' if speed_improvement > 0 else 'slower'})
        - 🎯 Accuracy: {accuracy_change:+.2f} percentage points
        
        💡 **Recommendation:**
        """
        
        if accuracy_change < -1:
            summary += f"Consider if the speed/size gains justify the {abs(accuracy_change):.1f}% accuracy loss."
        elif accuracy_change < 0:
            summary += f"Excellent tradeoff! Only {abs(accuracy_change):.1f}% accuracy loss for significant gains."
        else:
            summary += f"{prec2} provides both better accuracy AND efficiency!"
        
        return fig, summary
    
    def create_interface(self) -> gr.Blocks:
        """Create Gradio interface for quantization lesson"""
        from plotly.subplots import make_subplots
        
        with gr.Blocks(title="⚡ Lesson 06: Quantization") as demo:
            gr.Markdown("""
            # ⚡ Lesson 06: Quantization - Speed vs Precision
            
            Learn how **quantization** reduces model size and increases speed while managing 
            accuracy tradeoffs. Explore formats like **INT8**, **INT4**, and **GGUF**!
            
            ### Key Concepts:
            - **Precision Reduction**: FP32 → FP16 → INT8 → INT4
            - **GGUF Format**: Optimized quantization for CPU inference
            - **Tradeoffs**: Smaller/faster vs. slight accuracy loss
            """)
            
            with gr.Tab("Overview"):
                overview_chart = gr.Plot(label="All Precisions Comparison")
                gr.Markdown("""
                **Understanding the Chart:**
                - **Left**: Model size decreases with lower precision
                - **Middle**: Inference speed increases dramatically
                - **Right**: Accuracy remains high even at low precision
                """)
            
            with gr.Tab("Single Analysis"):
                with gr.Row():
                    precision_select = gr.Dropdown(
                        choices=['FP32', 'FP16', 'INT8', 'INT4', 'GGUF-Q4_K_M', 'GGUF-Q8_0'],
                        value='INT8',
                        label="Select Precision"
                    )
                    analyze_btn = gr.Button("🔍 Analyze", variant="primary")
                
                with gr.Row():
                    gauge_plot = gr.Plot(label="Accuracy Gauge")
                    analysis_text = gr.Markdown(label="Detailed Analysis")
            
            with gr.Tab("Compare Two"):
                with gr.Row():
                    prec1_select = gr.Dropdown(
                        choices=['FP32', 'FP16', 'INT8', 'INT4', 'GGUF-Q4_K_M', 'GGUF-Q8_0'],
                        value='FP32',
                        label="Precision A"
                    )
                    prec2_select = gr.Dropdown(
                        choices=['FP32', 'FP16', 'INT8', 'INT4', 'GGUF-Q4_K_M', 'GGUF-Q8_0'],
                        value='INT4',
                        label="Precision B"
                    )
                    compare_btn = gr.Button("⚖️ Compare", variant="primary")
                
                comparison_chart = gr.Plot(label="Side-by-Side Comparison")
                comparison_summary = gr.Markdown(label="Comparison Summary")
            
            # Event handlers
            demo.load(fn=lambda: self.create_comparison_chart(), outputs=[overview_chart])
            
            analyze_btn.click(
                fn=self.analyze_precision,
                inputs=[precision_select],
                outputs=[gauge_plot, analysis_text]
            )
            
            compare_btn.click(
                fn=self.compare_two_precisions,
                inputs=[prec1_select, prec2_select],
                outputs=[comparison_chart, comparison_summary]
            )
        
        return demo


def launch():
    """Launch the quantization lesson interface"""
    demo = QuantizationDemo()
    interface = demo.create_interface()
    interface.launch(server_name="0.0.0.0", server_port=7866, share=False)


if __name__ == "__main__":
    launch()
