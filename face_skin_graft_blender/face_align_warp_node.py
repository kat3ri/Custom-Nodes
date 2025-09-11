# face_align_warp_node.py

import cv2
import numpy as np
import torch
import copy
from scipy.spatial import Delaunay
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.python._framework_bindings.image import Image as MPImage
from PIL import Image
import onnxruntime as ort
import mediapipe as mp
from pathlib import Path
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarker, FaceLandmarkerOptions
from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarkerResult
from mediapipe.tasks.python.vision import RunningMode
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


VisionRunningMode = mp.tasks.vision.RunningMode
def safe_model_path(p):
    path = Path(p).expanduser().resolve()
    if not path.exists():
        # Auto-download if missing
        url = "https://storage.googleapis.com/mediapipe-assets/face_landmarker.task"
        print(f"Downloading model from {url} to {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        import urllib.request
        urllib.request.urlretrieve(url, path)
    return str(path)

class FaceAlignWarpNode:
    def __init__(self):
        model_path = safe_model_path(r"D:\ComfyUI\models\face_alignment\face_landmarker.task")
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model not found at: {model_path}")

        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            num_faces=1,
            running_mode=VisionRunningMode.IMAGE,
        )

        self.landmarker = FaceLandmarker.create_from_options(options)


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
        base = self.tensor2np(original_image)
        donor = self.tensor2np(donor_image)
        donor_mask = self.tensor2np(donor_mask, gray=True)

        base_lm = self.get_landmarks(base)
        donor_lm = self.get_landmarks(donor)
        if base_lm is None or donor_lm is None:
            h, w = base.shape[:2]
            return donor_image, torch.zeros((1, 1, h, w), dtype=torch.uint8)

        base_lm_2d = base_lm[:, :2]
        donor_lm_2d = donor_lm[:, :2]

        if donor_mask is None:
            donor_mask_np = self.generate_parsing_mask(donor)
            #mask_to_return = donor_mask_np
        else:
            donor_mask_np = self.tensor2np(donor_mask, gray=True)



        # Step 1: Get top contour of donor mask
        contours, _ = cv2.findContours(donor_mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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

        # Reduce pull from outer face areas
        jaw_indices = list(range(0, 17))
        side_face_indices = list(range(234, 454))
        soft_blend_indices = jaw_indices + side_face_indices
        for idx in soft_blend_indices:
            donor_lm_aug[idx] = donor_lm_aug[idx] * 0.7 + base_lm_aug[idx] * 0.3

        tri = Delaunay(base_lm_aug)
        h, w = base.shape[:2]
        warped_image = np.zeros_like(base)
        warped_mask = np.zeros((h, w), dtype=np.float32)




        # Apply soft blend BEFORE triangulation
        for idx in soft_blend_indices:
            donor_lm_aug[idx] = donor_lm_aug[idx] * 0.7 + base_lm_aug[idx] * 0.3


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

    def generate_parsing_mask(self, image):
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

        # Use class 13 only
        mask = (result == 13).astype(np.uint8) * 255

        # Resize back to original
        mask = Image.fromarray(mask, mode="L").resize(original_size)
        return np.array(mask).astype(np.uint8)

    def get_landmarks(self, image):

        # Resize and normalize [-1, 1]
        pil_image = Image.fromarray(image.astype(np.uint8)).convert("RGB").resize((512, 512))
        image_np = np.array(pil_image)  # shape: (512, 512, 3), dtype=uint8

        # Ensure RGB (if you're coming from BGR)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

        # Contiguous memory for MediaPipe
        image_np = np.ascontiguousarray(image_np, dtype=np.uint8)

        assert image_np.ndim == 3 and image_np.shape[2] == 3
        assert image_np.dtype == np.uint8

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_np)

        result = self.landmarker.detect(mp_image)

        if not result.face_landmarks or len(result.face_landmarks) == 0:
            return None

        landmark_list = result.face_landmarks[0]  # Only first face
        h, w, _ = image.shape

        landmarks = np.array([[lm.x * w, lm.y * h, lm.z * w] for lm in landmark_list], dtype=np.float32)
        return landmarks

    def tensor2np(self, tensor, gray=False):
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
        if mask.ndim == 2:
            mask = mask[None, None, ...]  # [1, 1, H, W]
        elif mask.ndim == 3:
            mask = mask[None, ...]  # [1, H, W] → [1, 1, H, W]
        mask = np.clip(mask, 0, 255).astype(np.uint8)
        return torch.from_numpy(mask)


NODE_CLASS_MAPPINGS = {"FaceAlignWarpNode": FaceAlignWarpNode}
NODE_DISPLAY_NAME_MAPPINGS = {"FaceAlignWarpNode": "Face Pre-Align Warp"}
