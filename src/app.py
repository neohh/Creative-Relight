import gradio as gr
import torch
from pathlib import Path
import os

from processors.image_processor import ImageProcessor
from processors.video_processor import VideoProcessor
from processors.sequence_processor import SequenceProcessor

# Set up temporary directory
TEMP_DIR = Path("temp_uploads")
TEMP_DIR.mkdir(exist_ok=True)
os.environ["GRADIO_TEMP_DIR"] = str(TEMP_DIR.absolute())

# Global processors (initialized once)
image_processor = None
video_processor = None
sequence_processor = None

def create_ui():
    """Create the main Gradio interface for Creative Relight"""
    
    # Initialize processors with device
    global image_processor, video_processor, sequence_processor
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    image_processor = ImageProcessor(device=device)
    video_processor = VideoProcessor(device=device)
    sequence_processor = SequenceProcessor(device=device)
    
    with gr.Blocks(
        title="Creative Relight",
        css="""
        .gradio-container {
            max-width: 1200px;
            margin: auto;
        }
        h1 {
            text-align: center;
            color: #2c3e50;
        }
        .preview-gallery {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 10px;
            margin-top: 20px;
        }
        .export-options {
            background-color: #f0f0f0;
            padding: 10px;
            border-radius: 5px;
            margin: 10px 0;
        }
        """,
    ) as app:
        
        gr.Markdown("""
        # 🎨 Creative Relight
        
        **Generate five essential passes for relighting and compositing:**
        - 🖼️ **Albedo** - Base color without lighting
        - 🌑 **Shading** - Light and shadow information
        - ✨ **Specular** - Highlights and reflections
        - 📏 **Depth** - Distance information
        - 🧭 **Normal** - Surface orientation
        """)
        
        with gr.Tabs() as tabs:
            # Single Image Tab
            with gr.Tab("🖼️ Single Image"):
                with gr.Row():
                    with gr.Column(scale=1):
                        single_image = gr.Image(
                            label="Input Image",
                            type="pil",
                            height=400
                        )
                        single_output_dir = gr.Textbox(
                            label="Output Directory",
                            placeholder="e.g., C:/Users/output/image_results",
                            value="./output/single_image"
                        )
                        
                        # Export options
                        with gr.Group():
                            gr.Markdown("### Export Options")
                            single_export_albedo = gr.Checkbox(label="Export Albedo", value=True)
                            single_export_shading = gr.Checkbox(label="Export Shading", value=True)
                            single_export_specular = gr.Checkbox(label="Export Specular", value=True)
                            single_export_depth = gr.Checkbox(label="Export Depth", value=True)
                            single_export_normal = gr.Checkbox(label="Export Normal", value=True)
                        
                        single_process_btn = gr.Button("Process Image", variant="primary", size="lg")
                    
                    with gr.Column(scale=2):
                        single_status = gr.Textbox(label="Status", interactive=False)
                        with gr.Row():
                            single_albedo = gr.Image(label="Albedo", height=150)
                            single_shading = gr.Image(label="Shading", height=150)
                            single_specular = gr.Image(label="Specular", height=150)
                        with gr.Row():
                            single_depth = gr.Image(label="Depth", height=150)
                            single_normal = gr.Image(label="Normal", height=150)
            
            # Video Tab
            with gr.Tab("🎥 Video"):
                with gr.Row():
                    with gr.Column():
                        video_input = gr.Video(label="Input Video")
                        video_output_dir = gr.Textbox(
                            label="Output Directory",
                            placeholder="e.g., C:/Users/output/video_results",
                            value="./output/video"
                        )
                        video_frame_step = gr.Slider(
                            minimum=1,
                            maximum=30,
                            value=1,
                            step=1,
                            label="Process every N frames"
                        )
                        
                        # Export options
                        with gr.Group():
                            gr.Markdown("### Export Options")
                            video_export_albedo = gr.Checkbox(label="Export Albedo", value=True)
                            video_export_shading = gr.Checkbox(label="Export Shading", value=True)
                            video_export_specular = gr.Checkbox(label="Export Specular", value=True)
                            video_export_depth = gr.Checkbox(label="Export Depth", value=True)
                            video_export_normal = gr.Checkbox(label="Export Normal", value=True)
                            
                        with gr.Row():
                            video_process_btn = gr.Button("Process Video", variant="primary")
                            video_stop_btn = gr.Button("Stop", variant="stop")
                    
                    with gr.Column():
                        video_status = gr.Textbox(label="Status", interactive=False)
                        video_progress = gr.Progress()
            
            # Image Sequence Tab
            with gr.Tab("📁 Image Sequence"):
                with gr.Row():
                    with gr.Column():
                        sequence_folder = gr.Textbox(
                            label="Input Folder Path",
                            placeholder="e.g., C:/Users/image_sequence"
                        )
                        sequence_output_dir = gr.Textbox(
                            label="Output Directory",
                            placeholder="e.g., C:/Users/output/sequence_results",
                            value="./output/sequence"
                        )
                        
                        # Export options
                        with gr.Group():
                            gr.Markdown("### Export Options")
                            sequence_export_albedo = gr.Checkbox(label="Export Albedo", value=True)
                            sequence_export_shading = gr.Checkbox(label="Export Shading", value=True)
                            sequence_export_specular = gr.Checkbox(label="Export Specular", value=True)
                            sequence_export_depth = gr.Checkbox(label="Export Depth", value=True)
                            sequence_export_normal = gr.Checkbox(label="Export Normal", value=True)
                        
                        with gr.Row():
                            sequence_process_btn = gr.Button("Process Sequence", variant="primary")
                            sequence_stop_btn = gr.Button("Stop", variant="stop")
                    
                    with gr.Column():
                        sequence_status = gr.Textbox(label="Status", interactive=False)
                        sequence_progress = gr.Progress()
        
        # Event handlers
        def process_single_image(image, output_dir, export_albedo, export_shading, export_specular, 
                               export_depth, export_normal, progress=gr.Progress()):
            try:
                if image is None:
                    return "❌ No image provided", None, None, None, None, None
                
                if not any([export_albedo, export_shading, export_specular, export_depth, export_normal]):
                    return "❌ Please select at least one pass to export", None, None, None, None, None
                
                progress(0.1, desc="Processing image...")
                
                # Create export config
                export_config = {
                    'albedo': export_albedo,
                    'shading': export_shading,
                    'specular': export_specular,
                    'depth': export_depth,
                    'normal': export_normal
                }
                
                result = image_processor.process(
                    image, output_dir,
                    export_config=export_config
                )
                
                progress(1.0, desc="Complete!")
                
                previews = result.get('previews', {})
                return (
                    f"✅ Successfully processed! Files saved to: {output_dir}",
                    previews.get('albedo') if export_albedo else None,
                    previews.get('shading') if export_shading else None,
                    previews.get('specular') if export_specular else None,
                    previews.get('depth') if export_depth else None,
                    previews.get('normal') if export_normal else None
                )
            except Exception as e:
                return f"❌ Error: {str(e)}", None, None, None, None, None
        
        def process_video(video_path, output_dir, frame_step, export_albedo, export_shading, 
                         export_specular, export_depth, export_normal, progress=gr.Progress()):
            try:
                if not any([export_albedo, export_shading, export_specular, export_depth, export_normal]):
                    return "❌ Please select at least one pass to export"
                
                export_config = {
                    'albedo': export_albedo,
                    'shading': export_shading,
                    'specular': export_specular,
                    'depth': export_depth,
                    'normal': export_normal
                }
                return video_processor.process(
                    video_path, output_dir, frame_step,
                    export_config=export_config,
                    progress=progress
                )
            except Exception as e:
                return f"❌ Error: {str(e)}"
        
        def process_sequence(folder_path, output_dir, export_albedo, export_shading, export_specular,
                           export_depth, export_normal, progress=gr.Progress()):
            try:
                if not any([export_albedo, export_shading, export_specular, export_depth, export_normal]):
                    return "❌ Please select at least one pass to export"
                
                export_config = {
                    'albedo': export_albedo,
                    'shading': export_shading,
                    'specular': export_specular,
                    'depth': export_depth,
                    'normal': export_normal
                }
                return sequence_processor.process(
                    folder_path, output_dir,
                    export_config=export_config,
                    progress=progress
                )
            except Exception as e:
                return f"❌ Error: {str(e)}"
        
        def stop_processing():
            video_processor.stop()
            sequence_processor.stop()
            return "🛑 Stop requested - processing will halt"
        
        # Connect events
        single_process_btn.click(
            process_single_image,
            inputs=[single_image, single_output_dir, single_export_albedo, single_export_shading,
                   single_export_specular, single_export_depth, single_export_normal],
            outputs=[single_status, single_albedo, single_shading, single_specular, single_depth, single_normal]
        )
        
        video_process_btn.click(
            process_video,
            inputs=[video_input, video_output_dir, video_frame_step, video_export_albedo,
                   video_export_shading, video_export_specular, video_export_depth, video_export_normal],
            outputs=[video_status]
        )
        
        video_stop_btn.click(
            stop_processing,
            outputs=[video_status]
        )
        
        sequence_process_btn.click(
            process_sequence,
            inputs=[sequence_folder, sequence_output_dir, sequence_export_albedo,
                   sequence_export_shading, sequence_export_specular, sequence_export_depth,
                   sequence_export_normal],
            outputs=[sequence_status]
        )
        
        sequence_stop_btn.click(
            stop_processing,
            outputs=[sequence_status]
        )
    
    return app

def main():
    """Main entry point"""
    # Check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create and launch UI
    app = create_ui()
    app.launch(show_error=True, ssr_mode=False)

if __name__ == "__main__":
    main()