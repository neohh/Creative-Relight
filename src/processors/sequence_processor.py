import numpy as np
from pathlib import Path
from PIL import Image
import cv2
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
    from file_handler import ensure_dir, get_image_files
    from image_utils import ensure_uint8
except ModuleNotFoundError:
    # Fallback to package import (frozen mode)
    from utils.file_handler import ensure_dir, get_image_files
    from utils.image_utils import ensure_uint8

# Lazy import - TemporalStabilizer will be imported only when needed
TemporalStabilizer = None

class SequenceProcessor:
    """Process image sequences to generate all 5 passes"""
    
    def __init__(self, device='cuda', use_temporal_consistency=False, temporal_window_size=3):
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
                    print("[SEQUENCE-TEMPORAL] Imported TemporalStabilizer successfully")
                except ModuleNotFoundError:
                    try:
                        from processors.temporal_consistency import TemporalStabilizer
                        print("[SEQUENCE-TEMPORAL] Imported TemporalStabilizer from processors package")
                    except Exception as e:
                        print(f"[SEQUENCE-TEMPORAL] ❌ Failed to import TemporalStabilizer: {e}")
                        print("[SEQUENCE-TEMPORAL] Disabling temporal consistency")
                        self.use_temporal_consistency = False
                        return
            
            try:
                self.temporal_stabilizer = TemporalStabilizer(window_size=temporal_window_size)
                print(f"[SEQUENCE-TEMPORAL] Temporal consistency enabled (window size: {temporal_window_size})")
            except Exception as e:
                print(f"[SEQUENCE-TEMPORAL] ❌ Failed to initialize TemporalStabilizer: {e}")
                print("[SEQUENCE-TEMPORAL] Disabling temporal consistency")
                self.use_temporal_consistency = False
                self.temporal_stabilizer = None
    
    def stop(self):
        """Stop processing"""
        self.should_stop = True
    
    def process_file_list(self, image_files, output_dir, export_config=None, progress=None):
        """Process a list of image files and save selected passes"""
        print("[SEQUENCE] process_file_list called")
        print(f"[SEQUENCE] Image files count: {len(image_files) if image_files else 0}")
        print(f"[SEQUENCE] Output dir: {output_dir}")
        print(f"[SEQUENCE] Export config: {export_config}")
        
        self.should_stop = False
        
        # Validate inputs
        if not image_files:
            error_msg = "No image files provided"
            print(f"[SEQUENCE ERROR] {error_msg}")
            return f"❌ {error_msg}"
        
        if not output_dir:
            error_msg = "No output directory specified"
            print(f"[SEQUENCE ERROR] {error_msg}")
            return f"❌ {error_msg}"
        
        # Default export all if not specified
        if export_config is None:
            export_config = {
                'albedo': True,
                'specular': True,
                'depth': True,
                'normal': True
            }
        
        print(f"[SEQUENCE] Will process {len(image_files)} images")
        
        total_images = len(image_files)
        
        # Create output directories only for selected passes
        output_path = Path(output_dir)
        print(f"[SEQUENCE] Creating output directories at: {output_path}")
        
        dirs = {}
        try:
            for component in ['albedo', 'specular', 'depth', 'normal']:
                if export_config.get(component, False):
                    dirs[component] = ensure_dir(output_path / component)
                    print(f"[SEQUENCE] Created directory for {component}")
        except Exception as e:
            error_msg = f"Failed to create output directories: {str(e)}"
            print(f"[SEQUENCE ERROR] {error_msg}")
            import traceback
            traceback.print_exc()
            return f"❌ {error_msg}"
        
        processed_count = 0
        
        try:
            print("[SEQUENCE] Starting image processing loop...")
            for idx, img_path in enumerate(image_files):
                if self.should_stop:
                    print("[SEQUENCE] Stop requested, breaking loop")
                    break
                
                print(f"[SEQUENCE] Processing image {idx + 1}/{total_images}: {img_path}")
                
                # Update progress and check if we should stop
                if progress:
                    try:
                        should_continue = progress((idx + 1) / total_images, f"Processing image {idx + 1}/{total_images}")
                        if should_continue == False:  # Check explicit False return
                            print("[SEQUENCE] Progress callback returned False, stopping")
                            self.should_stop = True
                            break
                    except Exception as e:
                        print(f"[SEQUENCE WARNING] Progress callback error: {e}")
                
                try:
                    # Load image
                    print(f"[SEQUENCE] Loading image: {img_path}")
                    image = Image.open(img_path).convert('RGB')
                    # Get original dimensions for resizing outputs
                    original_size = image.size
                    print(f"[SEQUENCE] Image loaded, size: {original_size}")
                    image_array = np.array(image).astype(np.float32) / 255.0
                    
                    # Process with models only if needed
                    lighting_results = {}
                    if any(export_config.get(comp, False) for comp in ['albedo', 'specular']):
                        print("[SEQUENCE] Processing with lighting model...")
                        lighting_results = self.lighting_model.process(image_array)
                        print(f"[SEQUENCE] Lighting results: {list(lighting_results.keys())}")
                    
                    geometry_results = {}
                    if any(export_config.get(comp, False) for comp in ['depth', 'normal']):
                        print("[SEQUENCE] Processing with geometry model...")
                        geometry_results = self.geometry_model.process(image_array)
                        print(f"[SEQUENCE] Geometry results: {list(geometry_results.keys())}")
                    
                    # Apply temporal consistency if enabled
                    if self.temporal_stabilizer:
                        try:
                            from image_utils import resize_to_original
                            print(f"[SEQUENCE-TEMPORAL] Applying temporal consistency to frame {idx + 1}")
                            
                            # Convert image to uint8 RGB for optical flow
                            frame_rgb = (np.array(image) * 255).astype(np.uint8) if isinstance(image, Image.Image) else np.array(image).astype(np.uint8)
                            
                            # Compute optical flow for current frame
                            flow = self.temporal_stabilizer.compute_optical_flow(frame_rgb)
                            if flow is not None:
                                print(f"[SEQUENCE-TEMPORAL] Optical flow computed for frame {idx + 1}")
                            
                            # Stabilize albedo
                            if 'albedo' in lighting_results and lighting_results['albedo'] is not None:
                                print(f"[SEQUENCE-TEMPORAL] Stabilizing albedo for frame {idx + 1}")
                                # Resize to original resolution first
                                albedo_full_res = resize_to_original(lighting_results['albedo'], original_size)
                                # Stabilize at full resolution
                                albedo_stabilized = self.temporal_stabilizer.stabilize_albedo(albedo_full_res, flow)
                                # Store stabilized version
                                lighting_results['albedo'] = albedo_stabilized
                                print(f"[SEQUENCE-TEMPORAL] Albedo stabilized for frame {idx + 1}")
                            
                            # Stabilize specular
                            if 'specular' in lighting_results and lighting_results['specular'] is not None:
                                print(f"[SEQUENCE-TEMPORAL] Stabilizing specular for frame {idx + 1}")
                                # Resize to original resolution first
                                specular_full_res = resize_to_original(lighting_results['specular'], original_size)
                                # Stabilize at full resolution
                                specular_stabilized = self.temporal_stabilizer.stabilize_specular(specular_full_res, flow)
                                # Store stabilized version
                                lighting_results['specular'] = specular_stabilized
                                print(f"[SEQUENCE-TEMPORAL] Specular stabilized for frame {idx + 1}")
                        
                        except Exception as e:
                            import traceback
                            print(f"[SEQUENCE-TEMPORAL] Warning: Temporal stabilization failed for frame {idx + 1}: {e}")
                            print(f"[SEQUENCE-TEMPORAL] Traceback: {traceback.format_exc()}")
                            # Continue without temporal consistency for this frame
                    
                    # Save selected components
                    frame_num = f"{idx:06d}"
                    
                    # Save albedo - resize to original dimensions (if not already resized by temporal consistency)
                    if export_config.get('albedo', False) and 'albedo' in lighting_results and lighting_results['albedo'] is not None:
                        print("[SEQUENCE] Saving albedo...")
                        from image_utils import resize_to_original
                        albedo = lighting_results['albedo']
                        # Only resize if temporal consistency didn't already do it
                        if albedo.shape[:2] != original_size[::-1]:  # Compare (height, width) vs (width, height)
                            albedo = resize_to_original(albedo, original_size)
                        albedo = ensure_uint8(albedo)
                        Image.fromarray(albedo).save(dirs['albedo'] / f"albedo_{frame_num}.png")
                    
                    # Save specular - resize to original dimensions (if not already resized by temporal consistency)
                    if export_config.get('specular', False) and 'specular' in lighting_results and lighting_results['specular'] is not None:
                        print("[SEQUENCE] Saving specular...")
                        from image_utils import resize_to_original
                        specular = lighting_results['specular']
                        # Only resize if temporal consistency didn't already do it
                        if specular.shape[:2] != original_size[::-1]:  # Compare (height, width) vs (width, height)
                            specular = resize_to_original(specular, original_size)
                        specular = ensure_uint8(specular)
                        # Save as grayscale if 2D
                        if len(specular.shape) == 2:
                            Image.fromarray(specular, mode='L').save(dirs['specular'] / f"specular_{frame_num}.png")
                        else:
                            Image.fromarray(specular).save(dirs['specular'] / f"specular_{frame_num}.png")
                    
                    # Save depth - resize to original dimensions
                    if export_config.get('depth', False) and 'depth' in geometry_results and geometry_results['depth'] is not None:
                        print("[SEQUENCE] Saving depth...")
                        depth = cv2.resize(geometry_results['depth'], original_size, interpolation=cv2.INTER_LINEAR)
                        cv2.imwrite(str(dirs['depth'] / f"depth_{frame_num}.png"), depth)
                    
                    # Save normal - resize to original dimensions
                    if export_config.get('normal', False) and 'normal' in geometry_results and geometry_results['normal'] is not None:
                        print("[SEQUENCE] Saving normal...")
                        normal = cv2.resize(geometry_results['normal'], original_size, interpolation=cv2.INTER_LINEAR)
                        cv2.imwrite(str(dirs['normal'] / f"normal_{frame_num}.png"), 
                                   cv2.cvtColor(normal, cv2.COLOR_RGB2BGR))
                    
                    processed_count += 1
                    print(f"[SEQUENCE] Successfully processed image {idx + 1}")
                    
                except Exception as e:
                    error_msg = f"Error processing image {idx + 1} ({img_path}): {str(e)}"
                    print(f"[SEQUENCE ERROR] {error_msg}")
                    import traceback
                    traceback.print_exc()
                    # Continue to next image instead of failing completely
                    continue
            
            if self.should_stop:
                result_msg = f"⚠️ Processing stopped. Processed {processed_count} images"
                print(f"[SEQUENCE] {result_msg}")
                return result_msg
            else:
                selected_passes = [k for k, v in export_config.items() if v]
                result_msg = f"✅ Successfully processed {processed_count} images! Exported: {', '.join(selected_passes)}. Files saved to: {output_dir}"
                print(f"[SEQUENCE] {result_msg}")
                return result_msg
                
        except Exception as e:
            error_msg = f"❌ Error processing sequence: {str(e)}"
            print(f"[SEQUENCE ERROR] {error_msg}")
            import traceback
            traceback.print_exc()
            return error_msg
    
    def process(self, input_folder, output_dir, export_config=None, progress=None):
        """Process image sequence and save selected passes"""
        self.should_stop = False
        
        # Default export all if not specified
        if export_config is None:
            export_config = {
                'albedo': True,
                'specular': True,
                'depth': True,
                'normal': True
            }
        
        # Get image files
        try:
            image_files = get_image_files(input_folder)
            if not image_files:
                return f"❌ No image files found in: {input_folder}"
        except Exception as e:
            return f"❌ Error accessing folder: {str(e)}"
        
        total_images = len(image_files)
        
        # Create output directories only for selected passes
        output_path = Path(output_dir)
        dirs = {}
        for component in ['albedo', 'specular', 'depth', 'normal']:
            if export_config.get(component, False):
                dirs[component] = ensure_dir(output_path / component)
        
        processed_count = 0
        
        try:
            for idx, img_path in enumerate(image_files):
                if self.should_stop:
                    break
                
                # Update progress and check if we should stop
                if progress:
                    should_continue = progress((idx + 1) / total_images, f"Processing image {idx + 1}/{total_images}")
                    if should_continue == False:  # Check explicit False return
                        self.should_stop = True
                        break
                
                # Load image
                image = Image.open(img_path).convert('RGB')
                # Get original dimensions for resizing outputs
                original_size = image.size
                image_array = np.array(image).astype(np.float32) / 255.0
                
                # Process with models only if needed
                lighting_results = {}
                if any(export_config.get(comp, False) for comp in ['albedo', 'specular']):
                    lighting_results = self.lighting_model.process(image_array)
                
                geometry_results = {}
                if any(export_config.get(comp, False) for comp in ['depth', 'normal']):
                    geometry_results = self.geometry_model.process(image_array)
                
                # Save selected components
                frame_num = f"{idx:06d}"
                
                # Save albedo - resize to original dimensions
                if export_config.get('albedo', False) and 'albedo' in lighting_results and lighting_results['albedo'] is not None:
                    from image_utils import resize_to_original
                    albedo = resize_to_original(lighting_results['albedo'], original_size)
                    albedo = ensure_uint8(albedo)
                    Image.fromarray(albedo).save(dirs['albedo'] / f"albedo_{frame_num}.png")
                
                # Save specular - resize to original dimensions
                if export_config.get('specular', False) and 'specular' in lighting_results and lighting_results['specular'] is not None:
                    from image_utils import resize_to_original
                    specular = resize_to_original(lighting_results['specular'], original_size)
                    specular = ensure_uint8(specular)
                    # Save as grayscale if 2D
                    if len(specular.shape) == 2:
                        Image.fromarray(specular, mode='L').save(dirs['specular'] / f"specular_{frame_num}.png")
                    else:
                        Image.fromarray(specular).save(dirs['specular'] / f"specular_{frame_num}.png")
                
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
            
            if self.should_stop:
                return f"⚠️ Processing stopped. Processed {processed_count} images"
            else:
                selected_passes = [k for k, v in export_config.items() if v]
                return f"✅ Successfully processed {processed_count} images! Exported: {', '.join(selected_passes)}. Files saved to: {output_dir}"
                
        except Exception as e:
            return f"❌ Error processing sequence: {str(e)}"