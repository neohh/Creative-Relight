"""
Creative Relight - Temporal Consistency Module
Reduces flickering and improves temporal stability for video processing
Uses advanced optical flow and frame blending techniques
"""

import cv2
import numpy as np
from collections import deque

class TemporalStabilizer:
    """
    Provides temporal consistency for video pass extraction
    
    Features:
    - Optical flow computation
    - Temporal smoothing with frame warping
    - Deflickering post-processing
    """
    
    def __init__(self, window_size=3, deflicker_strength=2):
        """
        Initialize temporal stabilizer
        
        Args:
            window_size: Number of frames to use for temporal smoothing (default: 3)
            deflicker_strength: Strength of deflickering (1-4, default: 2)
        """
        self.window_size = window_size
        self.deflicker_strength = deflicker_strength
        
        # Frame buffers for each pass type
        self.albedo_buffer = deque(maxlen=window_size)
        self.specular_buffer = deque(maxlen=window_size)
        self.depth_buffer = deque(maxlen=window_size)
        self.normal_buffer = deque(maxlen=window_size)
        
        # Optical flow buffer
        self.flow_buffer = deque(maxlen=window_size-1)
        self.prev_frame = None
        
        print(f"⏱️ Temporal stabilizer initialized (window={window_size}, deflicker={deflicker_strength})")
    
    def compute_optical_flow(self, current_frame):
        """
        Compute optical flow from previous frame to current frame
        
        Args:
            current_frame: Current frame (RGB uint8 or float32)
        
        Returns:
            Optical flow (H, W, 2) or None if no previous frame
        """
        if self.prev_frame is None:
            self.prev_frame = current_frame.copy()
            return None
        
        # Convert to grayscale for optical flow
        if current_frame.dtype != np.uint8:
            current_frame = (current_frame * 255).astype(np.uint8)
        if self.prev_frame.dtype != np.uint8:
            self.prev_frame = (self.prev_frame * 255).astype(np.uint8)
        
        gray_prev = cv2.cvtColor(self.prev_frame, cv2.COLOR_RGB2GRAY) if len(self.prev_frame.shape) == 3 else self.prev_frame
        gray_curr = cv2.cvtColor(current_frame, cv2.COLOR_RGB2GRAY) if len(current_frame.shape) == 3 else current_frame
        
        # Compute dense optical flow using Farneback method
        flow = cv2.calcOpticalFlowFarneback(
            gray_prev, gray_curr,
            None,
            pyr_scale=0.5,      # Pyramid scale factor
            levels=3,            # Number of pyramid layers
            winsize=15,          # Averaging window size
            iterations=3,        # Iterations at each pyramid level
            poly_n=5,            # Size of pixel neighborhood
            poly_sigma=1.2,      # Gaussian sigma for polynomial expansion
            flags=0
        )
        
        # Store current frame for next iteration
        self.prev_frame = current_frame.copy()
        
        return flow
    
    def warp_frame(self, frame, flow):
        """
        Warp frame using optical flow
        
        Args:
            frame: Frame to warp (H, W, C) or (H, W)
            flow: Optical flow (H, W, 2)
        
        Returns:
            Warped frame
        """
        if flow is None:
            return frame
        
        h, w = flow.shape[:2]
        
        # Create flow map for remapping
        flow_map = np.copy(flow)
        flow_map[:, :, 0] += np.arange(w)  # Add x coordinates
        flow_map[:, :, 1] += np.arange(h)[:, np.newaxis]  # Add y coordinates
        
        # Warp the frame
        warped = cv2.remap(
            frame,
            flow_map[:, :, 0].astype(np.float32),
            flow_map[:, :, 1].astype(np.float32),
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE
        )
        
        return warped
    
    def stabilize_albedo(self, albedo, flow=None):
        """
        Apply temporal stabilization to albedo pass
        
        Args:
            albedo: Current frame's albedo (H, W, 3)
            flow: Optical flow from previous frame (H, W, 2)
        
        Returns:
            Stabilized albedo (H, W, 3)
        """
        self.albedo_buffer.append(albedo.copy())
        
        if flow is not None:
            self.flow_buffer.append(flow)
        
        # Need at least 2 frames for stabilization
        if len(self.albedo_buffer) < 2:
            return albedo
        
        return self._temporal_smooth(self.albedo_buffer, self.flow_buffer)
    
    def stabilize_specular(self, specular, flow=None):
        """
        Apply temporal stabilization to specular pass
        
        Args:
            specular: Current frame's specular (H, W) or (H, W, C)
            flow: Optical flow from previous frame (H, W, 2)
        
        Returns:
            Stabilized specular
        """
        self.specular_buffer.append(specular.copy())
        
        # Need at least 2 frames for stabilization
        if len(self.specular_buffer) < 2:
            return specular
        
        return self._temporal_smooth(self.specular_buffer, self.flow_buffer)
    
    def stabilize_depth(self, depth, flow=None):
        """
        Apply temporal stabilization to depth pass
        
        Args:
            depth: Current frame's depth (H, W) or (H, W, 3)
            flow: Optical flow from previous frame (H, W, 2)
        
        Returns:
            Stabilized depth
        """
        self.depth_buffer.append(depth.copy())
        
        # Need at least 2 frames for stabilization
        if len(self.depth_buffer) < 2:
            return depth
        
        return self._temporal_smooth(self.depth_buffer, self.flow_buffer)
    
    def stabilize_normal(self, normal, flow=None):
        """
        Apply temporal stabilization to normal pass
        
        Args:
            normal: Current frame's normal (H, W, 3)
            flow: Optical flow from previous frame (H, W, 2)
        
        Returns:
            Stabilized normal (H, W, 3)
        """
        self.normal_buffer.append(normal.copy())
        
        # Need at least 2 frames for stabilization
        if len(self.normal_buffer) < 2:
            return normal
        
        return self._temporal_smooth(self.normal_buffer, self.flow_buffer)
    
    def _temporal_smooth(self, frame_buffer, flow_buffer):
        """
        Apply temporal smoothing using frame warping
        
        Args:
            frame_buffer: Deque of recent frames
            flow_buffer: Deque of optical flows
        
        Returns:
            Smoothed frame
        """
        current_frame = frame_buffer[-1]
        
        # Debug: log shapes
        print(f"[TEMPORAL_SMOOTH] Current frame shape: {current_frame.shape}")
        if len(flow_buffer) > 0:
            print(f"[TEMPORAL_SMOOTH] Latest flow shape: {flow_buffer[-1].shape}")
        
        # Warp previous frames to current frame
        warped_frames = []
        weights = []
        
        for i in range(len(frame_buffer)):
            if i == len(frame_buffer) - 1:
                # Current frame - highest weight
                warped_frames.append(current_frame)
                weights.append(1.0)
            else:
                # Warp older frames if we have flow
                if i < len(flow_buffer):
                    # Use the corresponding flow
                    flow_idx = -(len(frame_buffer) - 1 - i)
                    if flow_idx < 0 and abs(flow_idx) <= len(flow_buffer):
                        flow = flow_buffer[flow_idx]
                        # Validate shapes before warping
                        if flow.shape[:2] != current_frame.shape[:2]:
                            print(f"[TEMPORAL_SMOOTH] Shape mismatch! flow: {flow.shape}, frame: {current_frame.shape}")
                            # Skip warping if shapes don't match
                            warped_frames.append(frame_buffer[i])
                        else:
                            warped = self.warp_frame(frame_buffer[i], flow)
                            warped_frames.append(warped)
                    else:
                        warped_frames.append(frame_buffer[i])
                else:
                    warped_frames.append(frame_buffer[i])
                
                # Exponential decay for older frames
                weight = 0.5 ** (len(frame_buffer) - 1 - i)
                weights.append(weight)
        
        # Normalize weights
        weights = np.array(weights)
        weights /= weights.sum()
        
        # Weighted average - validate all frames have same shape
        shapes = [f.shape for f in warped_frames]
        if len(set(shapes)) > 1:
            print(f"[TEMPORAL_SMOOTH] Multiple frame shapes detected: {shapes}")
            # Fall back to just returning current frame
            return current_frame
        
        # Weighted average
        smoothed = np.zeros_like(current_frame, dtype=np.float32)
        for frame, weight in zip(warped_frames, weights):
            smoothed += frame.astype(np.float32) * weight
        
        # Clip to valid range
        smoothed = np.clip(smoothed, 0, 255 if current_frame.dtype == np.uint8 else 1.0)
        
        return smoothed.astype(current_frame.dtype)
    
    def reset(self):
        """Reset all buffers (call between videos)"""
        self.albedo_buffer.clear()
        self.specular_buffer.clear()
        self.depth_buffer.clear()
        self.normal_buffer.clear()
        self.flow_buffer.clear()
        self.prev_frame = None
        print("⏱️ Temporal stabilizer reset")


class Deflickerer:
    """
    Post-processing deflickering for entire sequences
    Apply after all frames have been processed
    """
    
    def __init__(self, window_size=5):
        """
        Initialize deflickerer
        
        Args:
            window_size: Temporal window for median filtering (default: 5)
        """
        self.window_size = window_size
    
    def deflicker_sequence(self, frames):
        """
        Apply temporal median filtering to remove flicker
        
        Args:
            frames: Array of frames [N, H, W, C] or [N, H, W]
        
        Returns:
            Deflickered frames [N, H, W, C] or [N, H, W]
        """
        frames = np.array(frames, dtype=np.float32)
        n_frames = len(frames)
        
        deflickered = np.zeros_like(frames)
        
        half_window = self.window_size // 2
        
        print(f"🎬 Deflickering {n_frames} frames (window={self.window_size})...")
        
        for i in range(n_frames):
            # Get temporal window
            start_idx = max(0, i - half_window)
            end_idx = min(n_frames, i + half_window + 1)
            
            window = frames[start_idx:end_idx]
            
            # Temporal median (removes outliers/flicker)
            deflickered[i] = np.median(window, axis=0)
        
        print("✅ Deflickering complete")
        
        return deflickered


def test_temporal_consistency():
    """Test function to verify temporal consistency module"""
    print("Testing temporal consistency module...")
    
    # Create test frames
    test_frames = []
    for i in range(10):
        frame = np.random.rand(256, 256, 3).astype(np.float32)
        test_frames.append(frame)
    
    # Test stabilizer
    stabilizer = TemporalStabilizer(window_size=3)
    
    for i, frame in enumerate(test_frames):
        flow = stabilizer.compute_optical_flow(frame)
        stabilized = stabilizer.stabilize_albedo(frame, flow)
        print(f"Frame {i}: stabilized shape = {stabilized.shape}")
    
    # Test deflickerer
    deflickerer = Deflickerer(window_size=5)
    deflickered = deflickerer.deflicker_sequence(test_frames)
    print(f"Deflickered: {deflickered.shape}")
    
    print("✅ Temporal consistency module test passed!")


if __name__ == "__main__":
    test_temporal_consistency()
