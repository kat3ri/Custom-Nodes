# face_align_warp_node.py
# Refactored for MediaPipe 0.10.x compatibility
# Compatible with MediaPipe tasks API and vision module

import os
import cv2
import numpy as np
import torch
from scipy.spatial import Delaunay
from PIL import Image
import onnxruntime as ort
import mediapipe as mp
from pathlib import Path
import logging
import urllib.request
import urllib.error

# MediaPipe imports - using consolidated, non-deprecated paths
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision.face_landmarker import (
    FaceLandmarker,
    FaceLandmarkerOptions,
    FaceLandmarkerResult
)

# Configure logging
logger = logging.getLogger(__name__)


def safe_model_path(model_path_or_name="face_landmarker.task", max_retries=3):
    """
    Safely resolve and download the MediaPipe face landmarker model.
    
    Args:
        model_path_or_name: Full path or just filename. If just filename, uses default location.
        max_retries: Number of download retry attempts
        
    Returns:
        str: Absolute path to the model file
        
    Raises:
        FileNotFoundError: If model cannot be downloaded or found
    """
    # Default model location in user's home directory
    if Path(model_path_or_name).is_absolute():
        path = Path(model_path_or_name).expanduser().resolve()
    else:
        # Use a portable default location
        default_dir = Path.home() / ".cache" / "mediapipe" / "models"
        path = (default_dir / model_path_or_name).resolve()
    
    if path.exists():
        logger.info(f"Model found at: {path}")
        return str(path)
    
    # Auto-download if missing - try multiple URLs
    urls = [
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
        "https://storage.googleapis.com/mediapipe-assets/face_landmarker.task",
    ]
    
    logger.info(f"Model not found at {path}. Attempting download...")
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Try each URL
    for url in urls:
        logger.info(f"Trying URL: {url}")
        
        # Download with retry logic
        for attempt in range(max_retries):
            try:
                logger.info(f"Download attempt {attempt + 1}/{max_retries}")
                
                # Add user agent to avoid 403 errors
                req = urllib.request.Request(
                    url,
                    headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
                )
                
                with urllib.request.urlopen(req) as response:
                    with open(path, 'wb') as out_file:
                        out_file.write(response.read())
                
                # Verify download
                if path.exists() and path.stat().st_size > 0:
                    logger.info(f"Model successfully downloaded to: {path}")
                    return str(path)
                else:
                    raise ValueError(f"Downloaded file is empty or invalid")
                    
            except (urllib.error.URLError, urllib.error.HTTPError, ValueError) as e:
                logger.warning(f"Download attempt {attempt + 1} from {url} failed: {e}")
                if path.exists():
                    path.unlink()
                    
                if attempt == max_retries - 1:
                    # Try next URL
                    break
    
    # All URLs failed
    raise FileNotFoundError(
        f"Failed to download model after trying all URLs. "
        f"Please manually download the face_landmarker.task model and place it at {path}\n"
        f"Tried URLs:\n" + "\n".join(f"  - {url}" for url in urls)
    )

class FaceAlignWarpNode:
    """
    Face alignment and warping node for ComfyUI.
    Uses MediaPipe Face Landmarker for facial landmark detection.
    Compatible with MediaPipe 0.10.x
    
    Note:
        Requires face_landmarker.task model file. The model will be automatically downloaded
        on first use if not found. If automatic download fails, manually place the model at:
        ~/.cache/mediapipe/models/face_landmarker.task
        
        Model can be downloaded from:
        https://developers.google.com/mediapipe/solutions/vision/face_landmarker#models
    """
    
    def __init__(self):
        """
        Initialize the Face Landmarker with proper error handling.
        
        The initialization is deferred - if model cannot be loaded during __init__,
        it will be retried during first use in the run() method.
        """
        self.landmarker = None
        self.parsing_session = None  # ONNX session for face parsing, initialized separately
        self._init_error = None
        
        try:
            self._initialize_landmarker()
        except Exception as e:
            # Store error but don't fail init - allow retry during run()
            self._init_error = e
            logger.warning(f"Failed to initialize landmarker during __init__: {e}")
            logger.warning("Will retry initialization during first use")
    
    def _initialize_landmarker(self):
        """Initialize the landmarker - can be called multiple times."""
        # Try to use environment variable or default path
        model_path_env = os.environ.get('FACE_LANDMARKER_MODEL_PATH', 'face_landmarker.task')
        model_path = safe_model_path(model_path_env)
        
        # Configure Face Landmarker options
        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            num_faces=1,
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            min_face_detection_confidence=0.5,  # Added for robustness
            min_face_presence_confidence=0.5,   # Added for robustness
            min_tracking_confidence=0.5          # Added for robustness
        )
        
        self.landmarker = FaceLandmarker.create_from_options(options)
        self._init_error = None  # Clear any previous errors
        logger.info("FaceLandmarker initialized successfully")


    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "original_image": ("IMAGE",),
                "donor_image": ("IMAGE",),
            },
            "optional": {
                "donor_mask": ("MASK",),  # externally provided
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("aligned_donor_image", "face_mask")
    FUNCTION = "run"
    CATEGORY = "image"

    def run(self, original_image, donor_image, donor_mask=None):
        """
        Align and warp donor face to match original image face.
        
        Args:
            original_image: Target image tensor
            donor_image: Source face image tensor
            donor_mask: Optional mask tensor for donor face region
            
        Returns:
            tuple: (aligned_donor_image, face_mask)
        """
        # Retry initialization if it failed during __init__
        if self.landmarker is None:
            if self._init_error:
                logger.warning("Retrying landmarker initialization...")
                try:
                    self._initialize_landmarker()
                except Exception as e:
                    logger.error(f"Failed to initialize landmarker: {e}")
                    # Return donor image unchanged with empty mask
                    h, w = self.tensor2np(original_image).shape[:2]
                    return donor_image, torch.zeros((1, 1, h, w), dtype=torch.uint8)
        
        try:
            base = self.tensor2np(original_image)
            donor = self.tensor2np(donor_image)
            donor_mask_input = self.tensor2np(donor_mask, gray=True) if donor_mask is not None else None
            
            # Get landmarks for both images
            base_lm = self.get_landmarks(base)
            donor_lm = self.get_landmarks(donor)
            
            # Handle case where landmarks are not detected
            if base_lm is None or donor_lm is None:
                logger.warning("Failed to detect landmarks in one or both images, returning original donor image")
                h, w = base.shape[:2]
                # Return donor image unchanged with empty mask
                return donor_image, torch.zeros((1, 1, h, w), dtype=torch.uint8)
            
            base_lm_2d = base_lm[:, :2]
            donor_lm_2d = donor_lm[:, :2]
            
            # Get or generate donor mask
            if donor_mask_input is None:
                donor_mask_np = self.generate_parsing_mask(donor)
            else:
                donor_mask_np = donor_mask_input

            # Step 1: Get top contour of donor mask
            contours, _ = cv2.findContours(donor_mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                logger.warning("No contours found in donor mask, returning donor image with empty mask")
                h, w = base.shape[:2]
                return donor_image, torch.zeros((1, 1, h, w), dtype=torch.uint8)
            
            mask_contour = max(contours, key=cv2.contourArea)

            # Step 2: Select top-most points in the contour (e.g., forehead arc)
            # Get bounding box of contour
            x_min = np.min(mask_contour[:, 0, 0])
            x_max = np.max(mask_contour[:, 0, 0])
            x_vals = np.linspace(x_min, x_max, 20).astype(int)

            # For each x, find the topmost y in the contour
            top_points = []
            for x in x_vals:
                candidates = mask_contour[:, 0, :]
                y_candidates = candidates[candidates[:, 0] == x]
                if len(y_candidates) > 0:
                    top_y = np.min(y_candidates[:, 1])
                    top_points.append([x, top_y])

            top_points = np.array(top_points)

            # Step 3: Extrapolate those upward
            forehead_offset = 60  # How far up you want to stretch
            extrapolated_points = np.array([[x, max(0, y - forehead_offset)] for x, y in top_points])

            # Step 4: Concatenate with landmarks
            base_lm_aug = np.vstack([base_lm_2d, extrapolated_points])
            donor_lm_aug = np.vstack([donor_lm_2d, extrapolated_points])  # You can share extrapolated pts

            # Focus on eyes, nose, mouth fidelity
            shape_sensitive_indices = list(set([
                *range(33, 133),  # eyes and brows
                *range(61, 68), *range(291, 296),  # lips
                *range(1, 20),  # inner face for general shape control
                195, 5, 4, 51, 280, 287  # key nasal points
            ]))

            donor_lm_aug[shape_sensitive_indices] = donor_lm_aug[shape_sensitive_indices] * 0.92 + base_lm_aug[shape_sensitive_indices] * 0.08

            # Reduce pull from outer face areas (apply soft blend before triangulation)
            jaw_indices = list(range(0, 17))
            side_face_indices = list(range(234, 454))
            soft_blend_indices = jaw_indices + side_face_indices
            for idx in soft_blend_indices:
                donor_lm_aug[idx] = donor_lm_aug[idx] * 0.7 + base_lm_aug[idx] * 0.3

            tri = Delaunay(base_lm_aug)
            h, w = base.shape[:2]
            warped_image = np.zeros_like(base)
            warped_mask = np.zeros((h, w), dtype=np.float32)

            for tri_indices in tri.simplices:
                t1 = np.float32([donor_lm_aug[i] for i in tri_indices])
                t2 = np.float32([base_lm_aug[i] for i in tri_indices])
                matrix = cv2.getAffineTransform(t1, t2)

                tri_mask = np.zeros((h, w), dtype=np.uint8)
                cv2.fillConvexPoly(tri_mask, np.int32(t2), 255)

                assert donor_mask_np.ndim == 2

                warped_part = cv2.warpAffine(donor, matrix, (w, h), flags=cv2.INTER_LINEAR,
                                             borderMode=cv2.BORDER_REFLECT_101)
                warped_mask_part = cv2.warpAffine(donor_mask_np, matrix, (w, h), flags=cv2.INTER_LINEAR,
                                                  borderMode=cv2.BORDER_REFLECT_101)

                warped_image[tri_mask == 255] = warped_part[tri_mask == 255]
                warped_mask[tri_mask == 255] = warped_mask_part[tri_mask == 255]

            warped_mask = np.clip(warped_mask, 0, 255).astype(np.uint8)

            # === Smooth transition with distance blending ===
            mask_bin = (warped_mask > 0.05).astype(np.uint8)
            dist_out = cv2.distanceTransform(1 - mask_bin, cv2.DIST_L2, 5)
            dist_in = cv2.distanceTransform(mask_bin, cv2.DIST_L2, 5)
            feather_out = np.clip(dist_out / 25, 0, 1)
            feather_in = np.clip(dist_in / 25, 0, 1)
            soft_mask = np.clip(1.0 - feather_out, 0, 1) * np.clip(feather_in, 0, 1)
            warped_mask = np.clip(soft_mask, 0, 1)

            # Fill missing pixels
            missing_pixels = (warped_image.sum(axis=-1) == 0)
            warped_image[missing_pixels] = base[missing_pixels]

            # Blend with soft mask
            blended = base.astype(np.float32) * (1 - warped_mask[..., None]) + \
                      warped_image.astype(np.float32) * warped_mask[..., None]
            blended = np.clip(blended, 0, 255).astype(np.uint8)

            # Optional: Polish seam edges
            edge = cv2.Canny((warped_mask * 255).astype(np.uint8), 20, 60)
            edge = cv2.dilate(edge, None, iterations=1)
            blurred = cv2.bilateralFilter(blended, d=7, sigmaColor=60, sigmaSpace=60)
            blended[edge > 0] = blurred[edge > 0]

            output_mask = warped_mask[None, None, ...] * 255.0
            return self.np2tensor(blended), self.np2tensor_mask(output_mask)
        
        except Exception as e:
            logger.error(f"Error in face alignment process: {e}", exc_info=True)
            # Return original donor image with empty mask on error
            h, w = self.tensor2np(original_image).shape[:2]
            return donor_image, torch.zeros((1, 1, h, w), dtype=torch.uint8)

    def generate_parsing_mask(self, image):
        """
        Generate face parsing mask using ONNX model.
        
        Args:
            image: numpy array in RGB format
            
        Returns:
            numpy array mask in uint8 format
            
        Note:
            Requires self.parsing_session to be initialized with an ONNX model.
            Falls back to a simple threshold mask if session is not available.
        """
        try:
            if not hasattr(self, 'parsing_session') or self.parsing_session is None:
                logger.warning("ONNX parsing session not initialized, creating fallback mask")
                # Create a simple fallback mask based on face landmarks
                return self._create_fallback_mask(image)
            
            original_size = (image.shape[1], image.shape[0])

            # Resize and normalize [-1, 1]
            pil_image = Image.fromarray(image.astype(np.uint8)).resize((512, 512))
            image_np = np.array(pil_image).astype(np.float32) / 127.5 - 1.0
            image_np = image_np[None, ...]  # [1, 512, 512, 3] — NHWC

            input_name = self.parsing_session.get_inputs()[0].name
            output_name = self.parsing_session.get_outputs()[0].name
            result = self.parsing_session.run([output_name], {input_name: image_np})

            # [1, 512, 512, num_classes] → argmax over channels
            result = np.array(result[0]).argmax(axis=3).squeeze(0)

            # Use class 13 only (face region)
            mask = (result == 13).astype(np.uint8) * 255

            # Resize back to original
            mask = Image.fromarray(mask, mode="L").resize(original_size)
            return np.array(mask).astype(np.uint8)
            
        except Exception as e:
            logger.error(f"Error generating parsing mask: {e}")
            return self._create_fallback_mask(image)
    
    def _create_fallback_mask(self, image):
        """Create a simple fallback mask when ONNX parsing is not available."""
        logger.debug("Creating fallback mask")
        # Create a simple oval mask based on image dimensions
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        center = (w // 2, h // 2)
        axes = (w // 3, h // 2)
        cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
        return mask

    def get_landmarks(self, image):
        """
        Extract facial landmarks from an image using MediaPipe Face Landmarker.
        
        Args:
            image: numpy array in RGB format, shape (H, W, 3), dtype uint8
            
        Returns:
            numpy array of shape (num_landmarks, 3) with x, y, z coordinates,
            or None if no face is detected
        """
        try:
            # Ensure image is in the correct format
            if not isinstance(image, np.ndarray):
                raise TypeError(f"Expected numpy array, got {type(image)}")
            
            if image.dtype != np.uint8:
                logger.warning(f"Image dtype is {image.dtype}, converting to uint8")
                image = np.clip(image, 0, 255).astype(np.uint8)
            
            # Resize to standard size for consistent processing
            pil_image = Image.fromarray(image.astype(np.uint8)).convert("RGB").resize((512, 512))
            image_np = np.array(pil_image)  # shape: (512, 512, 3), dtype=uint8
            
            # MediaPipe expects RGB format
            if image_np.shape[2] != 3:
                raise ValueError(f"Expected 3-channel RGB image, got shape {image_np.shape}")
            
            # Ensure contiguous memory for MediaPipe
            image_np = np.ascontiguousarray(image_np, dtype=np.uint8)
            
            # Create MediaPipe Image object using the latest API
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_np)
            
            # Detect landmarks
            result = self.landmarker.detect(mp_image)
            
            # Handle no face detected
            if not result.face_landmarks or len(result.face_landmarks) == 0:
                logger.warning("No face detected in image")
                return None
            
            # Extract landmarks from first face
            landmark_list = result.face_landmarks[0]
            
            # MediaPipe returns normalized coordinates (0-1 range)
            # Scale them to original image dimensions, not the resized 512x512
            h, w, _ = image.shape
            
            # Convert normalized coordinates to pixel coordinates in original image space
            landmarks = np.array(
                [[lm.x * w, lm.y * h, lm.z * w] for lm in landmark_list],
                dtype=np.float32
            )
            
            logger.debug(f"Successfully extracted {len(landmarks)} landmarks")
            return landmarks
            
        except Exception as e:
            logger.error(f"Failed to extract landmarks: {e}")
            return None

    def tensor2np(self, tensor, gray=False):
        """
        Convert tensor or numpy array to numpy array in uint8 format.
        
        Args:
            tensor: torch.Tensor or numpy array
            gray: If True, returns grayscale image
            
        Returns:
            numpy array in uint8 format
        """
        if tensor is None:
            return None
            
        if isinstance(tensor, torch.Tensor):
            arr = tensor[0].cpu().numpy()
        elif isinstance(tensor, np.ndarray):
            arr = tensor[0] if tensor.ndim == 4 else tensor
        else:
            raise TypeError(f"Expected tensor or np.ndarray, got {type(tensor)}")

        if arr.ndim == 3 and arr.shape[0] == 3 and not gray:
            arr = arr.transpose(1, 2, 0)
        elif gray:
            arr = arr[0] if arr.ndim == 3 else arr

        return np.clip(arr * 255.0, 0, 255).astype(np.uint8)

    def np2tensor(self, image):
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        elif image.ndim == 3 and image.shape[2] == 1:
            image = np.repeat(image, 3, axis=2)

        assert image.shape[2] == 3, f"Expected RGB image, got shape: {image.shape}"

        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0

        image = image[None, ...]  # [1, H, W, 3]
        return torch.from_numpy(image).float()

    def np2tensor_mask(self, mask):
        """
        Convert numpy mask to torch tensor with shape [1, 1, H, W].
        
        Args:
            mask: numpy array mask (2D, 3D, or 4D)
            
        Returns:
            torch.Tensor with shape [1, 1, H, W]
        """
        # Convert to grayscale if needed
        if mask.ndim == 3:
            if mask.shape[2] > 1:  # Multi-channel, take first channel
                mask = mask[:, :, 0]
            else:  # Single channel in 3D format
                mask = mask[:, :, 0]
        
        # Add batch and channel dimensions
        if mask.ndim == 2:
            mask = mask[None, None, ...]  # [H, W] → [1, 1, H, W]
        elif mask.ndim == 3:
            mask = mask[None, ...]  # [1, H, W] → [1, 1, H, W]
        elif mask.ndim == 4:
            # Already has batch and channel dimensions
            pass
        
        mask = np.clip(mask, 0, 255).astype(np.uint8)
        return torch.from_numpy(mask)


NODE_CLASS_MAPPINGS = {"FaceAlignWarpNode": FaceAlignWarpNode}
NODE_DISPLAY_NAME_MAPPINGS = {"FaceAlignWarpNode": "Face Pre-Align Warp"}
