"""
Creative Relight - Processing Worker
Background thread worker for AI processing tasks
"""

import time
from pathlib import Path
from PyQt6.QtCore import QObject, pyqtSignal


class ProcessingWorker(QObject):
    """Worker class for processing files in a background thread"""
    
    finished = pyqtSignal()
    progress = pyqtSignal(int, str)
    preview = pyqtSignal(object)
    status = pyqtSignal(str)
    files_progress = pyqtSignal(int, int)  # processed_items, total_items (frames for video, files for others)
    speed = pyqtSignal(float)  # processing speed in items per second
    frame_progress = pyqtSignal(int, int, str)  # current_frame, total_frames, filename
    time_remaining = pyqtSignal(float)  # estimated time remaining in seconds

    def __init__(self, main_window, export_config, output_dir):
        super().__init__()
        self.main_window = main_window
        self.export_config = export_config
        self.output_dir = output_dir
        self._running = True
        self.start_time = None
        
        # Enhanced speed tracking
        self.processing_times = []  # Store processing times for recent items
        self.max_samples = 10  # Keep last 10 samples for rolling average
        self.total_processing_units = 0
        self.processed_units = 0
        self.enabled_passes_count = 0

    def stop(self):
        """Stop the processing worker"""
        self._running = False

    def calculate_enabled_passes_count(self):
        """Calculate how many passes are enabled for processing"""
        if not self.export_config:
            return 4  # Default to all passes if config is missing
        return sum(1 for enabled in self.export_config.values() if enabled)

    def add_processing_time_sample(self, processing_time_per_unit):
        """Add a processing time sample for rolling average calculation"""
        self.processing_times.append(processing_time_per_unit)
        # Keep only the most recent samples
        if len(self.processing_times) > self.max_samples:
            self.processing_times.pop(0)

    def get_current_speed(self):
        """Calculate current processing speed based on recent samples"""
        if not self.processing_times:
            return 0.0
        
        # Use rolling average of recent processing times
        avg_time_per_unit = sum(self.processing_times) / len(self.processing_times)
        if avg_time_per_unit <= 0:
            return 0.0
        
        # Speed = units per second
        return 1.0 / avg_time_per_unit

    def calculate_time_remaining(self):
        """Calculate estimated time remaining based on current speed"""
        current_speed = self.get_current_speed()
        if current_speed <= 0:
            return None
        
        remaining_units = self.total_processing_units - self.processed_units
        if remaining_units <= 0:
            return 0.0
        
        # Time remaining = remaining units / current speed
        return remaining_units / current_speed

    def emit_speed_and_time_updates(self):
        """Emit current speed and time remaining signals"""
        current_speed = self.get_current_speed()
        self.speed.emit(current_speed)
        
        time_remaining = self.calculate_time_remaining()
        if time_remaining is not None:
            self.time_remaining.emit(time_remaining)

    def run(self):
        """Main processing loop"""
        self.start_time = time.time()
        
        # Calculate enabled passes count for speed calculation
        self.enabled_passes_count = self.calculate_enabled_passes_count()
        
        try:
            file_type = self.main_window.file_type_combo.currentText()
            
            # Handle Image Sequence differently - process as a list of files
            if file_type == "Image Sequence":
                if not self.main_window.current_files:
                    self.status.emit("❌ No sequence files selected")
                    return
                
                total_files = len(self.main_window.current_files)
                self.total_processing_units = total_files
                self.processed_units = 0
                
                self.status.emit(f"🚀 Starting processing of image sequence ({total_files} frames)...")
                
                # Process entire sequence with sequence processor
                processor = self.main_window.sequence_processor
                if not processor:
                    self.status.emit("❌ Sequence processor not initialized")
                    return
                
                sequence_start_time = time.time()
                
                # Define progress callback for sequence processing
                def sequence_progress(progress_value, message):
                    if not self._running:
                        return False
                    
                    # Update progress
                    self.progress.emit(int(progress_value * 100), message)
                    
                    # Estimate current frame from progress
                    current_frame = int(progress_value * total_files)
                    self.processed_units = current_frame
                    self.files_progress.emit(current_frame, total_files)
                    
                    # Emit speed updates periodically
                    if current_frame > 0:
                        elapsed_time = time.time() - sequence_start_time
                        time_per_frame = elapsed_time / current_frame
                        self.add_processing_time_sample(time_per_frame)
                        self.emit_speed_and_time_updates()
                    
                    return True  # Continue processing
                
                # Process the sequence by passing the file list instead of folder
                try:
                    # Add comprehensive error handling and logging
                    print(f"[DEBUG] Starting sequence processing")
                    print(f"[DEBUG] File count: {len(self.main_window.current_files)}")
                    print(f"[DEBUG] Output dir: {self.output_dir}")
                    print(f"[DEBUG] Export config: {self.export_config}")
                    print(f"[DEBUG] Processor type: {type(processor).__name__}")
                    
                    # Validate processor has the method
                    if not hasattr(processor, 'process_file_list'):
                        self.status.emit(f"❌ Processor missing process_file_list method")
                        print("[ERROR] Processor object does not have process_file_list method")
                        return
                    
                    print("[DEBUG] Calling process_file_list...")
                    
                    # Get temporal consistency settings from UI
                    use_temporal = self.main_window.enable_temporal.isChecked()
                    temporal_window = self.main_window.temporal_strength_slider.value()
                    
                    # Update processor with temporal consistency settings
                    if use_temporal != processor.use_temporal_consistency:
                        # Reinitialize processor with new settings if temporal consistency changed
                        try:
                            from sequence_processor import SequenceProcessor
                        except ModuleNotFoundError:
                            try:
                                from processors.sequence_processor import SequenceProcessor
                            except ModuleNotFoundError:
                                from src.processors.sequence_processor import SequenceProcessor
                        
                        processor = SequenceProcessor(
                            device=processor.device,
                            use_temporal_consistency=use_temporal,
                            temporal_window_size=temporal_window
                        )
                        self.main_window.sequence_processor = processor
                        print(f"[SEQUENCE-TEMPORAL] Reinitialized with use_temporal={use_temporal}, window={temporal_window}")
                    
                    result = processor.process_file_list(
                        self.main_window.current_files,  # Pass the actual file list
                        self.output_dir,
                        export_config=self.export_config,
                        progress=sequence_progress
                    )
                    
                    print(f"[DEBUG] Processing completed with result: {result}")
                    
                    sequence_end_time = time.time()
                    processing_time = sequence_end_time - sequence_start_time
                    
                    if self._running:
                        self.status.emit(f"✅ Processed {total_files} frames in {processing_time:.1f}s!")
                        self.progress.emit(100, "All sequence processing completed")
                    else:
                        self.status.emit(f"⏹️ Sequence processing stopped")
                
                except AttributeError as e:
                    error_msg = f"❌ Processor method error: {str(e)}"
                    self.status.emit(error_msg)
                    print(f"[ERROR] {error_msg}")
                    import traceback
                    traceback.print_exc()
                except MemoryError as e:
                    error_msg = f"❌ Out of memory: {str(e)}"
                    self.status.emit(error_msg)
                    print(f"[ERROR] {error_msg}")
                except RuntimeError as e:
                    error_msg = f"❌ Runtime error (GPU/CUDA issue?): {str(e)}"
                    self.status.emit(error_msg)
                    print(f"[ERROR] {error_msg}")
                    import traceback
                    traceback.print_exc()
                except Exception as e:
                    error_msg = f"❌ Error processing sequence: {str(e)}"
                    self.status.emit(error_msg)
                    print(f"[ERROR] {error_msg}")
                    import traceback
                    traceback.print_exc()
                
            else:
                # Handle Single Image and Video File
                total_files = len(self.main_window.current_files)
                processed_files = 0
                
                self.status.emit(f"🚀 Starting processing of {total_files} files...")
                
                # For videos, we need to track frames instead of files
                if file_type == "Video File":
                    # For videos, we'll estimate total frames or get from video metadata
                    self.total_processing_units = 0  # Will be updated when we know frame count
                    self.processed_units = 0
                    unit_name = "frames"
                else:
                    # For images, track files
                    self.total_processing_units = total_files
                    self.processed_units = 0
                    unit_name = "files"
                
                for i, file_path in enumerate(self.main_window.current_files):
                    # Check if processing should stop
                    if not self._running:
                        self.status.emit("⏹️ Processing cancelled by user")
                        break
                    
                    file_name = Path(file_path).name
                    self.status.emit(f"🔄 Processing {file_name}... ({i+1}/{total_files})")
                    self.progress.emit(0, f"Starting {file_name}")
                    
                    # Get processor for this file
                    processor = self.main_window.get_processor_for_file(file_path, file_type)
                    if not processor:
                        self.status.emit(f"❌ No suitable processor for {file_name}")
                        continue
                    
                    # Update processor with temporal consistency settings for videos
                    if file_type == "Video File":
                        use_temporal = self.main_window.enable_temporal.isChecked()
                        temporal_window = self.main_window.temporal_strength_slider.value()
                        
                        # Check if we need to reinitialize the video processor
                        if use_temporal != processor.use_temporal_consistency:
                            try:
                                from video_processor import VideoProcessor
                            except ModuleNotFoundError:
                                try:
                                    from processors.video_processor import VideoProcessor
                                except ModuleNotFoundError:
                                    from src.processors.video_processor import VideoProcessor
                            
                            processor = VideoProcessor(
                                device=processor.device,
                                use_temporal_consistency=use_temporal,
                                temporal_window_size=temporal_window
                            )
                            self.main_window.video_processor = processor
                            print(f"[VIDEO-TEMPORAL] Reinitialized with use_temporal={use_temporal}, window={temporal_window}")
                    
                    # Check if still running before starting processing
                    if not self._running:
                        break
                    
                    file_start_time = time.time()
                    
                    # Process the file with the correct processor
                    try:
                        # Create a callback for frame progress if it's a video
                        if file_type == "Video File":
                            def frame_callback(frame_progress, frame_info):
                                if not self._running:
                                    return False  # Signal to stop processing
                                
                                # Extract frame numbers from progress (assuming it's a ratio 0-1)
                                current_frame = int(frame_progress * 1000)  # Will be updated with actual frames
                                total_frames = 1000  # Will be updated with actual total
                                
                                # Try to extract frame info from the string if provided
                                if isinstance(frame_info, str) and "frame" in frame_info.lower():
                                    try:
                                        # Parse "Processing frame X/Y" format
                                        import re
                                        match = re.search(r'(\d+)/(\d+)', frame_info)
                                        if match:
                                            current_frame = int(match.group(1))
                                            total_frames = int(match.group(2))
                                            
                                            # Update total processing units on first frame
                                            if self.total_processing_units == 0:
                                                self.total_processing_units = total_frames * total_files
                                    except:
                                        pass
                                
                                # Update current task
                                task_desc = f"Processing frame {current_frame}/{total_frames} of {file_name}"
                                self.progress.emit(int(frame_progress * 100), task_desc)
                                
                                # Update overall progress based on frames
                                self.processed_units = (processed_files * total_frames) + current_frame
                                global_total = self.total_processing_units if self.total_processing_units > 0 else total_frames
                                if global_total > 0:
                                    overall_progress = int((self.processed_units / global_total) * 100)
                                    self.files_progress.emit(self.processed_units, global_total)
                                
                                # Emit frame-specific progress
                                self.frame_progress.emit(current_frame, total_frames, file_name)
                                
                                # Calculate and emit speed/time updates
                                if current_frame > 0:
                                    elapsed = time.time() - file_start_time
                                    time_per_frame = elapsed / current_frame
                                    self.add_processing_time_sample(time_per_frame)
                                    self.emit_speed_and_time_updates()
                                
                                return True  # Continue processing
                            
                            # Pass the callback using the correct parameter name
                            result = processor.process(
                                file_path,
                                self.output_dir,
                                export_config=self.export_config,
                                progress=frame_callback
                            )
                            
                            # Update total processing units for videos after first file
                            if self.total_processing_units == 0 and hasattr(result, 'total_frames'):
                                self.total_processing_units = result.total_frames * total_files
                            elif self.total_processing_units == 0:
                                # Estimate frames from video duration (fallback)
                                self.total_processing_units = 24 * 16 * total_files  # Rough estimate
                                
                        else:
                            # For images, just process normally
                            result = processor.process(
                                file_path,
                                self.output_dir,
                                export_config=self.export_config
                            )
                        
                        # Check if processing was stopped during file processing
                        if not self._running:
                            break
                            
                    except Exception as e:
                        self.status.emit(f"❌ Error processing {file_name}: {str(e)}")
                        continue
                    
                    file_end_time = time.time()
                    processing_time = file_end_time - file_start_time
                    
                    processed_files += 1
                    
                    # Add processing time sample for speed calculation
                    if file_type == "Video File":
                        # For videos, calculate time per frame if we have frame info
                        # This is a rough estimate - we'll get more accurate data from frame callbacks
                        estimated_frames = 24 * 16  # Rough estimate for video duration
                        time_per_frame = processing_time / estimated_frames if estimated_frames > 0 else processing_time
                        self.add_processing_time_sample(time_per_frame)
                    else:
                        # For images/sequences, use time per file
                        self.add_processing_time_sample(processing_time)
                    
                    # Update progress based on files completed
                    if file_type != "Video File":
                        self.processed_units = processed_files
                        if self.total_processing_units > 0:
                            overall_progress = int((self.processed_units / self.total_processing_units) * 100)
                            self.progress.emit(overall_progress, f"Completed {file_name}")
                            self.files_progress.emit(self.processed_units, self.total_processing_units)
                    
                    # Emit speed and time remaining updates
                    self.emit_speed_and_time_updates()
                    
                    # Update preview with results if available and still running
                    if self._running and result:
                        if 'previews' in result:
                            self.preview.emit(result['previews'])
                        elif hasattr(result, 'get'):
                            # Try to extract preview data from result
                            preview_data = {}
                            for pass_type in ['albedo', 'normal', 'specular', 'depth']:
                                if pass_type in result:
                                    preview_data[pass_type] = result[pass_type]
                            if preview_data:
                                self.preview.emit(preview_data)
                    
                    # Emit status with timing information  
                    current_speed = self.get_current_speed()
                    self.status.emit(f"✅ Processed {file_name} in {processing_time:.1f}s | "
                                   f"Speed: {current_speed:.2f} {unit_name}/sec")
                    
                    # Brief pause to allow UI updates and check for stop
                    time.sleep(0.1)
                
                # Final status update
                if self._running:
                    total_time = time.time() - self.start_time
                    self.status.emit(f"✅ All {processed_files} files processed successfully in {total_time:.1f}s!")
                    self.progress.emit(100, "All processing completed")
                else:
                    self.status.emit(f"⏹️ Processing stopped after {processed_files}/{total_files} files")
                
        except Exception as e:
            self.status.emit(f"❌ Error during processing: {str(e)}")
            import traceback
            traceback.print_exc()
        finally:
            self.finished.emit()
