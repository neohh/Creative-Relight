import os
import cv2
import torch
import gradio as gr
import numpy as np
from pathlib import Path
from typing import Union, List, Optional
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt

from moge.model.v2 import MoGeModel

# Create a custom temporary directory in the workspace
TEMP_DIR = Path("temp_uploads")
TEMP_DIR.mkdir(exist_ok=True)
os.environ["GRADIO_TEMP_DIR"] = str(TEMP_DIR.absolute())

# Global state for process control
class ProcessState:
    def __init__(self):
        self.should_stop = False

    def stop(self):
        self.should_stop = True

    def reset(self):
        self.should_stop = False

process_state = ProcessState()

def ensure_dir(path: Union[str, Path]) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path

def process_video(video_path: str, output_dir: str, frame_step: int = 1, resize_depth: bool = False, depth_width: int = 1920, depth_height: int = 1080) -> tuple[List[str], List[str]]:
    """Process video file frame by frame"""
    global process_state
    process_state.reset()
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    # Create output directories
    output_dir = Path(output_dir)
    depth_dir = ensure_dir(output_dir / "depth")
    normal_dir = ensure_dir(output_dir / "normal")
    
    # Setup model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal").to(device)
    model.eval()
    
    depth_files = []
    normal_files = []
    frame_count = 0
    
    with torch.no_grad():
        while cap.isOpened() and not process_state.should_stop:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % frame_step == 0:
                # Convert BGR to RGB and normalize
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                input_tensor = torch.tensor(frame_rgb / 255.0, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)
                
                # Inference
                output = model.infer(input_tensor[0])
                
                # Save depth map with exact MoGe colorization
                depth_map = output["depth"].cpu().numpy()
                # Handle NaN and infinite values
                depth_map = np.nan_to_num(depth_map, nan=0.0, posinf=1.0, neginf=0.0)
                # Convert to colored depth map using MoGe's function
                # disp = 1/depth, normalized to [0,1], then use Spectral colormap
                disp = 1 / np.where(depth_map > 0, depth_map, np.nan)
                min_disp, max_disp = np.nanquantile(disp, 0.001), np.nanquantile(disp, 0.99)
                disp = (disp - min_disp) / (max_disp - min_disp)
                colored_depth = np.nan_to_num(plt.cm.Spectral(1.0 - disp)[..., :3], 0)
                colored_depth = np.ascontiguousarray((colored_depth.clip(0, 1) * 255).astype(np.uint8))
                
                if resize_depth:
                    colored_depth = cv2.resize(colored_depth, (depth_width, depth_height), interpolation=cv2.INTER_LINEAR)
                    
                # Convert colored depth to grayscale (weighted average of RGB channels)
                gray_depth = cv2.cvtColor(colored_depth, cv2.COLOR_RGB2GRAY)
                    
                depth_file = str(depth_dir / f"depth_{frame_count:06d}.png")
                cv2.imwrite(depth_file, gray_depth)
                depth_files.append(depth_file)
                
                # Save normal map with exact MoGe colorization
                normal_map = output["normal"].cpu().numpy()
                # MoGe normal colorization: scale to [0,1] range with specific scaling
                normal_map = normal_map * [0.5, -0.5, -0.5] + 0.5
                normal_map = (normal_map.clip(0, 1) * 255).astype(np.uint8)
                
                normal_file = str(normal_dir / f"normal_{frame_count:06d}.png")
                cv2.imwrite(normal_file, cv2.cvtColor(normal_map, cv2.COLOR_RGB2BGR))
                normal_files.append(normal_file)
            
            frame_count += 1
            
    cap.release()
    return depth_files, normal_files

def process_image_sequence(input_dir: str, output_dir: str, resize_depth: bool = False, depth_width: int = 1920, depth_height: int = 1080) -> tuple[List[str], List[str]]:
    """Process a sequence of images in a directory"""
    global process_state
    process_state.reset()
    
    input_dir = Path(input_dir)
    # Support common image formats
    image_files = sorted([f for f in input_dir.glob("*") if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}])
    
    if not image_files:
        raise ValueError(f"No supported image files found in directory: {input_dir}")
    
    # Create output directories
    output_dir = Path(output_dir)
    depth_dir = ensure_dir(output_dir / "depth")
    normal_dir = ensure_dir(output_dir / "normal")
    
    # Setup model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal").to(device)
    model.eval()
    
    depth_files = []
    normal_files = []
    
    with torch.no_grad():
        for idx, img_path in enumerate(image_files):
            if process_state.should_stop:
                break
            # Read and convert image to RGB
            frame = cv2.imread(str(img_path))
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Prepare input
            input_tensor = torch.tensor(frame_rgb / 255.0, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)
            
            # Inference
            output = model.infer(input_tensor[0])
            
            # Save depth map with exact MoGe colorization
            depth_map = output["depth"].cpu().numpy()
            # Handle NaN and infinite values
            depth_map = np.nan_to_num(depth_map, nan=0.0, posinf=1.0, neginf=0.0)
            # Convert to colored depth map using MoGe's function
            # disp = 1/depth, normalized to [0,1], then use Spectral colormap
            disp = 1 / np.where(depth_map > 0, depth_map, np.nan)
            min_disp, max_disp = np.nanquantile(disp, 0.001), np.nanquantile(disp, 0.99)
            disp = (disp - min_disp) / (max_disp - min_disp)
            colored_depth = np.nan_to_num(plt.cm.Spectral(1.0 - disp)[..., :3], 0)
            colored_depth = np.ascontiguousarray((colored_depth.clip(0, 1) * 255).astype(np.uint8))
            
            if resize_depth:
                colored_depth = cv2.resize(colored_depth, (depth_width, depth_height), interpolation=cv2.INTER_LINEAR)
                
            # Convert colored depth to grayscale (weighted average of RGB channels)
            gray_depth = cv2.cvtColor(colored_depth, cv2.COLOR_RGB2GRAY)
                
            depth_file = str(depth_dir / f"depth_{idx:06d}.png")
            cv2.imwrite(depth_file, gray_depth)
            depth_files.append(depth_file)
            
            # Save normal map with exact MoGe colorization
            normal_map = output["normal"].cpu().numpy()
            # MoGe normal colorization: scale to [0,1] range with specific scaling
            normal_map = normal_map * [0.5, -0.5, -0.5] + 0.5
            normal_map = (normal_map.clip(0, 1) * 255).astype(np.uint8)
            
            normal_file = str(normal_dir / f"normal_{idx:06d}.png")
            cv2.imwrite(normal_file, cv2.cvtColor(normal_map, cv2.COLOR_RGB2BGR))
            normal_files.append(normal_file)
    
    return depth_files, normal_files

def batch_process(
    input_path: str,
    output_dir: str,
    input_type: str = "video",
    frame_step: int = 1,
    resize_depth: bool = False,
    depth_width: int = 1920,
    depth_height: int = 1080,
) -> tuple[str, str]:
    """Main processing function"""
    global process_state
    process_state.reset()
    
    if not input_path:
        return "No input provided", "No input provided"
        
    if not output_dir:
        return "No output directory specified", "No output directory specified"
    
    try:
        if input_type == "video":
            depth_files, normal_files = process_video(input_path, output_dir, frame_step, resize_depth, depth_width, depth_height)
        else:  # image_sequence
            depth_files, normal_files = process_image_sequence(input_path, output_dir, resize_depth, depth_width, depth_height)
        
        if process_state.should_stop:
            return "Processing stopped by user", "Processing stopped by user"
            
        depth_msg = f"Saved {len(depth_files)} depth maps to {output_dir}/depth"
        normal_msg = f"Saved {len(normal_files)} normal maps to {output_dir}/normal"
        
        return depth_msg, normal_msg
        
    except Exception as e:
        return f"Error: {str(e)}", f"Error: {str(e)}"

# Create Gradio interface
def create_ui():
    def cleanup_temp_files():
        """Clean up temporary files"""
        import time
        if TEMP_DIR.exists():
            for file in TEMP_DIR.glob("*"):
                try:
                    if file.exists():
                        if os.name == 'nt':  # Windows
                            import subprocess
                            # Use del command with force option
                            subprocess.run(['del', '/F', str(file)], shell=True, check=False)
                        else:
                            file.unlink()
                except Exception as e:
                    print(f"Warning: Could not delete {file}: {e}")

    def stop_processing():
        global process_state
        process_state.stop()
        return "Stopping...", "Stopping..."

    with gr.Blocks(title="MoGe Batch Processor") as app:
        app.load(cleanup_temp_files)
        
        gr.Markdown("""
        # MoGe Batch Processor
        Extract depth and normal maps from videos or image sequences.
        
        - Supports video files (e.g., .mp4)
        - Supports image sequences (jpg, png)
        - Exports depth maps as customizable 16-bit grayscale PNG
        - Exports normal maps as PNG with accurate colors
        """)
        
        with gr.Row():
            with gr.Column():
                input_type = gr.Radio(
                    choices=["video", "image_sequence"],
                    value="video",
                    label="Input Type"
                )
                
                # Video inputs
                video_input = gr.Video(label="Input Video", visible=True)
                frame_step = gr.Slider(
                    minimum=1,
                    maximum=10,
                    value=1,
                    step=1,
                    label="Frame Step",
                    visible=True
                )
                
                # Image sequence inputs
                folder_input = gr.Textbox(
                    label="Image Sequence Folder Path",
                    placeholder="Paste or select folder path containing image sequence",
                    lines=1,
                    visible=False
                )
                gr.Examples(
                    examples=["C:/path/to/your/images"],
                    inputs=folder_input,
                    label="Example path format"
                )
                folder_browse = gr.Button("📁 Browse for Folder", variant="secondary", visible=False)
                
                output_dir = gr.Textbox(
                    label="Output Directory",
                    placeholder="Path where depth and normal maps will be saved"
                )
                
                # Depth map dimensions controls
                with gr.Row():
                    resize_depth = gr.Checkbox(label="Custom Depth Map Dimensions", value=False)
                    depth_width = gr.Number(label="Depth Width", value=1920, minimum=1, precision=0, visible=False)
                    depth_height = gr.Number(label="Depth Height", value=1080, minimum=1, precision=0, visible=False)
                
                with gr.Row():
                    process_btn = gr.Button("Process", variant="primary")
                    stop_btn = gr.Button("Stop", variant="secondary")
            
            with gr.Column():
                depth_output = gr.Textbox(label="Depth Maps Status")
                normal_output = gr.Textbox(label="Normal Maps Status")
        
        def browse_folder():
            from tkinter import filedialog, Tk
            root = Tk()
            root.withdraw()  # Hide the main window
            folder_path = filedialog.askdirectory(title="Select Image Sequence Folder")
            root.destroy()
            if folder_path:
                return folder_path
            return None
        
        def toggle_input_mode(input_type):
            is_video = input_type == "video"
            return [
                gr.update(visible=is_video),      # video_input
                gr.update(visible=is_video),      # frame_step
                gr.update(visible=not is_video),  # folder_input
                gr.update(visible=not is_video)   # folder_browse
            ]
        
        input_type.change(
            toggle_input_mode,
            inputs=[input_type],
            outputs=[video_input, frame_step, folder_input, folder_browse]
        )
        
        folder_browse.click(
            browse_folder,
            outputs=[folder_input]
        )
        
        def on_process(input_type, video, folder, output_dir, frame_step, resize_depth, depth_width, depth_height):
            if not output_dir:
                return "No output directory specified", "No output directory specified"
            
            if input_type == "video":
                if not video:
                    return "No video file provided", "No video file provided"
                input_path = video
            else:  # image_sequence
                if not folder or not folder.strip():
                    return "No folder path provided", "No folder path provided"
                input_path = folder.strip()
            
            if resize_depth and (depth_width <= 0 or depth_height <= 0):
                return "Invalid depth dimensions", "Invalid depth dimensions"
                
            try:
                return batch_process(input_path, output_dir, input_type, frame_step, resize_depth, int(depth_width), int(depth_height))
            except Exception as e:
                return f"Error: {str(e)}", f"Error: {str(e)}"
        
        # Connect the process and stop buttons
        process_btn.click(
            on_process,
            inputs=[input_type, video_input, folder_input, output_dir, frame_step,
                   resize_depth, depth_width, depth_height],
            outputs=[depth_output, normal_output]
        )
        
        stop_btn.click(
            stop_processing,
            outputs=[depth_output, normal_output]
        )
        
        def toggle_depth_dimensions(use_custom):
            return [
                gr.update(visible=use_custom),  # depth_width
                gr.update(visible=use_custom)   # depth_height
            ]
        
        resize_depth.change(
            toggle_depth_dimensions,
            inputs=[resize_depth],
            outputs=[depth_width, depth_height]
        )
    
    return app

if __name__ == "__main__":
    app = create_ui()
    app.launch()

