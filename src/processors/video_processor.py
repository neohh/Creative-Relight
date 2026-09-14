import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys

# Handle imports for both development and frozen (PyInstaller) environments
try:
    # Try direct import first (dev mode - models/ is in sys.path)
    from lighting_model import LightingModel
    from geometry_model import GeometryModel
except ModuleNotFoundError:
    # Fallback to package import (frozen/built mode)
    from models.lighting_model import LightingModel
    from models.geometry_model import GeometryModel

try:
    # Try direct import first (dev mode - utils/ is in sys.path)
    from file_handler import ensure_dir
    from image_utils import resize_to_original, ensure_uint8
except ModuleNotFoundError:
    # Fallback to package import (frozen mode)
    from utils.file_handler import ensure_dir
    from utils.image_utils import resize_to_original, ensure_uint8

# Lazy import - TemporalStabilizer will be imported only when needed
# This prevents crashes when loading the module even if TC is disabled
TemporalStabilizer = None

from PIL import Image

class VideoProcessor:
    """Process videos frame by frame to generate all 5 passes"""
    
    def __init__(self, device='cuda', use_temporal_consistency=True, temporal_window_size=3):
        self.device = device
        self.lighting_model = LightingModel(device)
        self.geometry_model = GeometryModel(device)
        self.should_stop = False
        
        # Temporal consistency settings
        self.use_temporal_consistency = use_temporal_consistency
        self.temporal_stabilizer = None
        if use_temporal_consistency:
            # Lazy import - only load TemporalStabilizer when actually needed
            global TemporalStabilizer
            if TemporalStabilizer is None:
                try:
                    from temporal_consistency import TemporalStabilizer
                    print("[TEMPORAL] Imported TemporalStabilizer successfully")
                except ModuleNotFoundError:
                    try:
                        from processors.temporal_consistency import TemporalStabilizer
                        print("[TEMPORAL] Imported TemporalStabilizer from processors package")
                    except Exception as e:
                        print(f"[TEMPORAL] ❌ Failed to import TemporalStabilizer: {e}")
                        print("[TEMPORAL] Disabling temporal consistency")
                        self.use_temporal_consistency = False
                        return
            
            try:
                self.temporal_stabilizer = TemporalStabilizer(window_size=temporal_window_size)
                print(f"[TEMPORAL] Temporal consistency enabled (window size: {temporal_window_size})")
            except Exception as e:
                print(f"[TEMPORAL] ❌ Failed to initialize TemporalStabilizer: {e}")
                print("[TEMPORAL] Disabling temporal consistency")
                self.use_temporal_consistency = False
                self.temporal_stabilizer = None
    
    def stop(self):
        """Stop processing"""
        self.should_stop = True
    
    def process(self, video_path, output_dir, frame_step=1, export_config=None, progress=None):
        """Process video and save selected passes for each frame"""
        self.should_stop = False
        
        # Reset temporal stabilizer for new video
        if self.temporal_stabilizer:
            self.temporal_stabilizer.reset()
            print("[TEMPORAL] Temporal stabilizer reset for new video")
        
        # Default export all if not specified
        if export_config is None:
            export_config = {
                'albedo': True,
                'specular': True,
                'depth': True,
                'normal': True
            }
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return f"❌ Could not open video file: {video_path}"
        
        # Get video info
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Create output directories only for selected passes
        output_path = Path(output_dir)
        dirs = {}
        for component in ['albedo', 'specular', 'depth', 'normal']:
            if export_config.get(component, False):
                dirs[component] = ensure_dir(output_path / component)
        
        frame_count = 0
        processed_count = 0
        
        print(f"[VIDEO_PROC] Starting video processing: {total_frames} frames total")
        print(f"[VIDEO_PROC] Frame step: {frame_step}, should_stop: {self.should_stop}")
        
        try:
            while cap.isOpened() and not self.should_stop:
                ret, frame = cap.read()
                if not ret:
                    print(f"[VIDEO_PROC] End of video reached at frame {frame_count}")
                    break
                
                if frame_count % frame_step == 0:
                    print(f"[VIDEO_PROC] Processing frame {frame_count}/{total_frames} (processed: {processed_count})")
                    
                    # Update progress
                    if progress:
                        progress(frame_count / total_frames, f"Processing frame {frame_count}/{total_frames}")
                    
                    # Convert BGR to RGB and normalize
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image_array = frame_rgb.astype(np.float32) / 255.0
                    
                    # Get original frame dimensions for resizing outputs
                    original_size = (width, height)
                    
                    # Process with models only if needed
                    lighting_results = {}
                    if any(export_config.get(comp, False) for comp in ['albedo', 'specular']):
                        lighting_results = self.lighting_model.process(image_array)
                    
                    geometry_results = {}
                    if any(export_config.get(comp, False) for comp in ['depth', 'normal']):
                        geometry_results = self.geometry_model.process(image_array)
                    
                    # Apply temporal consistency if enabled
                    if self.temporal_stabilizer:
                        try:
                            # Debug logging for shapes
                            print(f"[TEMPORAL] Frame {frame_count} - frame_rgb shape: {frame_rgb.shape}, original_size: {original_size}")
                            
                            # Compute optical flow for current frame (needs uint8 RGB at original resolution)
                            flow = self.temporal_stabilizer.compute_optical_flow(frame_rgb)
                            if flow is not None:
                                print(f"[TEMPORAL] Frame {frame_count} - flow shape: {flow.shape}")
                            
                            # Stabilize albedo - resize to original size first, then stabilize
                            if 'albedo' in lighting_results and lighting_results['albedo'] is not None:
                                print(f"[TEMPORAL] Frame {frame_count} - albedo before resize: {lighting_results['albedo'].shape}")
                                
                                # Resize albedo to original resolution for stabilization
                                albedo_full_res = resize_to_original(lighting_results['albedo'], original_size)
                                print(f"[TEMPORAL] Frame {frame_count} - albedo after resize: {albedo_full_res.shape}")
                                
                                # Stabilize at full resolution
                                albedo_stabilized = self.temporal_stabilizer.stabilize_albedo(
                                    albedo_full_res, flow
                                )
                                
                                # Store the stabilized version (already at original size)
                                lighting_results['albedo'] = albedo_stabilized
                            
                            # Stabilize specular - resize to original size first, then stabilize
                            if 'specular' in lighting_results and lighting_results['specular'] is not None:
                                print(f"[TEMPORAL] Frame {frame_count} - specular before resize: {lighting_results['specular'].shape}")
                                
                                # Resize specular to original resolution for stabilization
                                specular_full_res = resize_to_original(lighting_results['specular'], original_size)
                                print(f"[TEMPORAL] Frame {frame_count} - specular after resize: {specular_full_res.shape}")
                                
                                # Stabilize at full resolution
                                specular_stabilized = self.temporal_stabilizer.stabilize_specular(
                                    specular_full_res, flow
                                )
                                
                                # Store the stabilized version (already at original size)
                                lighting_results['specular'] = specular_stabilized
                                
                        except Exception as e:
                            import traceback
                            print(f"[TEMPORAL] Warning: Temporal stabilization failed for frame {frame_count}: {e}")
                            print(f"[TEMPORAL] Traceback: {traceback.format_exc()}")
                            # Continue without temporal consistency for this frame
                    
                    # Save selected components
                    frame_num = f"{processed_count:06d}"
                    
                    # Save albedo - already at original size if temporal consistency was applied
                    if export_config.get('albedo', False) and 'albedo' in lighting_results and lighting_results['albedo'] is not None:
                        albedo = lighting_results['albedo']
                        # Only resize if temporal consistency didn't already do it
                        if albedo.shape[:2] != (height, width):
                            albedo = resize_to_original(albedo, original_size)
                        albedo = ensure_uint8(albedo)
                        cv2.imwrite(str(dirs['albedo'] / f"albedo_{frame_num}.png"), 
                                   cv2.cvtColor(albedo, cv2.COLOR_RGB2BGR))
                    
                    # Save specular - already at original size if temporal consistency was applied
                    if export_config.get('specular', False) and 'specular' in lighting_results and lighting_results['specular'] is not None:
                        specular = lighting_results['specular']
                        # Only resize if temporal consistency didn't already do it
                        if specular.shape[:2] != (height, width):
                            specular = resize_to_original(specular, original_size)
                        specular = ensure_uint8(specular)
                        # Save as grayscale
                        if len(specular.shape) == 2:
                            cv2.imwrite(str(dirs['specular'] / f"specular_{frame_num}.png"), specular)
                        else:
                            # Convert to grayscale if needed
                            specular_gray = cv2.cvtColor(specular, cv2.COLOR_RGB2GRAY) if len(specular.shape) == 3 else specular
                            cv2.imwrite(str(dirs['specular'] / f"specular_{frame_num}.png"), specular_gray)
                    
                    # Save depth - resize to original dimensions
                    if export_config.get('depth', False) and 'depth' in geometry_results and geometry_results['depth'] is not None:
                        depth = cv2.resize(geometry_results['depth'], original_size, interpolation=cv2.INTER_LINEAR)
                        cv2.imwrite(str(dirs['depth'] / f"depth_{frame_num}.png"), depth)
                    
                    # Save normal - resize to original dimensions
                    if export_config.get('normal', False) and 'normal' in geometry_results and geometry_results['normal'] is not None:
                        normal = cv2.resize(geometry_results['normal'], original_size, interpolation=cv2.INTER_LINEAR)
                        cv2.imwrite(str(dirs['normal'] / f"normal_{frame_num}.png"), 
                                   cv2.cvtColor(normal, cv2.COLOR_RGB2BGR))
                    
                    processed_count += 1
                    print(f"[VIDEO_PROC] Frame {frame_count} processed successfully. Total processed: {processed_count}")
                
                frame_count += 1
                
                # Check stop condition
                if self.should_stop:
                    print(f"[VIDEO_PROC] Stop requested! Breaking loop at frame {frame_count}")
                    break
            
            cap.release()
            
            print(f"[VIDEO_PROC] Loop ended. Processed {processed_count}/{total_frames} frames")
            print(f"[VIDEO_PROC] should_stop={self.should_stop}, cap.isOpened()={cap.isOpened()}")
            
            if self.should_stop:
                return f"⚠️ Processing stopped. Processed {processed_count} frames"
            else:
                selected_passes = [k for k, v in export_config.items() if v]
                return f"✅ Successfully processed {processed_count} frames! Exported: {', '.join(selected_passes)}. Files saved to: {output_dir}"
                
        except Exception as e:
            cap.release()
            return f"❌ Error processing video: {str(e)}"