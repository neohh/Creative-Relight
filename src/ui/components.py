from pathlib import Path
import gradio as gr

def create_ui(process_image, process_video, process_sequence):
    with gr.Blocks(title="Creative Relight") as app:
        gr.Markdown("""
        # Creative Relight
        
        Generate albedo, shading, specular, depth, and normal maps from images and videos.
        """)
        
        with gr.Tab("Single Image"):
            image_input = gr.Image(label="Input Image", type="numpy")
            output_dir = gr.Textbox(label="Output Directory", placeholder="Path to save output files")
            process_btn = gr.Button("Process Image")
            image_output = gr.Textbox(label="Status")

            process_btn.click(
                fn=process_image,
                inputs=[image_input, output_dir],
                outputs=image_output
            )
        
        with gr.Tab("Video"):
            video_input = gr.Video(label="Input Video")
            video_output_dir = gr.Textbox(label="Output Directory", placeholder="Path to save output files")
            video_process_btn = gr.Button("Process Video")
            video_output = gr.Textbox(label="Status")

            video_process_btn.click(
                fn=process_video,
                inputs=[video_input, video_output_dir],
                outputs=video_output
            )
        
        with gr.Tab("Image Sequence"):
            folder_input = gr.Textbox(label="Image Sequence Folder Path", placeholder="Path to folder containing images")
            sequence_output_dir = gr.Textbox(label="Output Directory", placeholder="Path to save output files")
            sequence_process_btn = gr.Button("Process Sequence")
            sequence_output = gr.Textbox(label="Status")

            sequence_process_btn.click(
                fn=process_sequence,
                inputs=[folder_input, sequence_output_dir],
                outputs=sequence_output
            )

    return app