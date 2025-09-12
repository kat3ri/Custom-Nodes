import cv2
import numpy as np
import torch
import os
from pathlib import Path
from scipy.spatial import Delaunay
from scipy.interpolate import LinearNDInterpolator
import mediapipe as mp
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarker, FaceLandmarkerOptions
from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarkerResult
from mediapipe.tasks.python.vision import RunningMode
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

def safe_model_path(relative_path):
    # Find ComfyUI root directory by walking up from current file location
    current_dir = Path(__file__).parent
    comfyui_root = current_dir
    
    # Walk up directories until we find ComfyUI root
    while comfyui_root.name != "ComfyUI" and comfyui_root.parent != comfyui_root:
        comfyui_root = comfyui_root.parent
    
    # If we didn't find ComfyUI in the path, use current working directory
    if comfyui_root.name != "ComfyUI":
        comfyui_root = Path(os.getcwd())
        # Try to find ComfyUI in current working directory
        for item in comfyui_root.iterdir():
            if item.is_dir() and item.name == "ComfyUI":
                comfyui_root = item
                break
    
    # Construct absolute path
    model_path = comfyui_root / relative_path
    model_path = model_path.resolve()
    
    if not model_path.exists():
        # Auto-download if missing
        url = "https://storage.googleapis.com/mediapipe-assets/face_landmarker.task"
        print(f"Downloading model from {url} to {model_path}")
        model_path.parent.mkdir(parents=True, exist_ok=True)
        import urllib.request
        urllib.request.urlretrieve(url, model_path)
    
    return str(model_path)



VisionRunningMode = mp.tasks.vision.RunningMode

class FaceDepthMapNode:
    def __init__(self):
        self.model_path = safe_model_path("models/face_alignment/face_landmarker.task")  # Relative to ComfyUI root
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"Model not found at: {self.model_path}")

        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=self.model_path),
            num_faces=1,
            running_mode=VisionRunningMode.IMAGE,
        )
        self.landmarker = FaceLandmarker.create_from_options(options)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate_face_depth_map"
    CATEGORY = "CustomNodes/FaceTools"

    def generate_face_depth_map(self, image):
        tensor = image[0]

        if tensor.ndim == 3 and tensor.shape[0] in [1, 3]:
            image_np = tensor.permute(1, 2, 0).numpy()
        elif tensor.ndim == 3 and tensor.shape[2] in [1, 3]:
            image_np = tensor.numpy()
        else:
            raise ValueError(f"Unsupported tensor shape: {tensor.shape}")
        image_np = (image_np * 255).astype(np.uint8)

        if image_np.shape[2] == 4:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)
        elif image_np.shape[2] == 1:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
        elif image_np.shape[2] == 3:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        else:
            raise ValueError(f"Unsupported channel count: {image_np.shape[2]}")


        print("Resolved model path:", self.model_path)


        image_np = np.ascontiguousarray(image_np, dtype=np.uint8)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_np)
        result = self.landmarker.detect(mp_image)

        if not result.face_landmarks:
            raise ValueError("No face landmarks detected.")

        landmarks = result.face_landmarks[0]
        h, w, _ = image_np.shape
        points_3d = np.array([(lm.x * w, lm.y * h, lm.z) for lm in landmarks])
        z = -points_3d[:, 2]  # invert (camera toward face)
        z_min = np.percentile(z, 5)
        z_max = np.percentile(z, 95)

        z = (z - z_min) / (z_max - z_min)
        z = np.clip(z, 0, 1)
        points_3d[:, 2] = z

        tri = Delaunay(points_3d[:, :2])
        interp = LinearNDInterpolator(tri, points_3d[:, 2])

        grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
        grid_points = np.vstack((grid_x.ravel(), grid_y.ravel())).T

        depth_map = interp(grid_points).reshape(h, w)
        valid_mask = ~np.isnan(depth_map)
        depth_map = np.nan_to_num(depth_map, nan=0.0)
        depth_map = cv2.GaussianBlur(depth_map, (11, 11), 0)
        depth_map = cv2.normalize(depth_map, None, 0, 1, cv2.NORM_MINMAX)
        depth_map = depth_map ** 0.7  # gamma boost (adjustable)

        depth_norm = cv2.normalize(depth_map, None, 0, 1, cv2.NORM_MINMAX)
        depth_rgb = np.stack([depth_norm] * 3, axis=-1)
        depth_rgb = (depth_rgb * 255).astype(np.uint8)

        dark_background = np.zeros_like(depth_rgb)
        depth_rgb = np.where(valid_mask[..., None], depth_rgb, dark_background)

        preview_tensor = self.np2tensor(depth_rgb)
        return (preview_tensor,)


    def tensor2np(self, tensor):
        arr = tensor[0].cpu().numpy()
        if arr.ndim == 3 and arr.shape[0] == 3:
            arr = arr.transpose(1, 2, 0)
        elif arr.ndim == 3 and arr.shape[2] == 3:
            pass
        elif arr.ndim == 2:
            pass
        else:
            raise ValueError(f"Unexpected image shape: {arr.shape}")
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


NODE_CLASS_MAPPINGS = {
    "FaceDepthMapNode": FaceDepthMapNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FaceDepthMapNode": "🧑 Face Depth Map Generator"
}
