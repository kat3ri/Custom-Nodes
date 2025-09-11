import torch
import numpy as np
import cv2
print("✅ OpenCV version:", cv2.__version__)

from pathlib import Path
from PIL import Image
from nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
from PIL import Image
import mediapipe as mp


def safe_to_pil(image_np):
    # [B, C, H, W] → [H, W, C]
    if image_np.ndim == 4:
        image_np = image_np[0]
    if image_np.shape[0] in [1, 3]:
        image_np = image_np.transpose(1, 2, 0)

    # Grayscale
    if image_np.shape[-1] == 1:
        image_np = image_np[:, :, 0]

    # Normalize
    image_np = np.clip(image_np * 255.0, 0, 255).astype(np.uint8)

    return Image.fromarray(image_np)

def feather_mask(mask: np.ndarray, feather_px=10) -> np.ndarray:
    mask = np.clip(mask.astype(np.float32), 0, 1)
    mask_bin = (mask > 0.05).astype(np.uint8)
    dist_out = cv2.distanceTransform(1 - mask_bin, cv2.DIST_L2, 5)
    dist_in = cv2.distanceTransform(mask_bin, cv2.DIST_L2, 5)
    feather_out = np.clip(dist_out / feather_px, 0, 1)
    feather_in = np.clip(dist_in / feather_px, 0, 1)
    soft_mask = np.clip(1.0 - feather_out, 0, 1) * np.clip(feather_in, 0, 1)
    return soft_mask


class BasicSkinBlender:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "original_image": ("IMAGE",),
                "enhanced_image": ("IMAGE",),
                "mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("blended_image", "mask_image")
    FUNCTION = "run"
    CATEGORY = "image"

    def feather_mask(mask, feather_px=40):
        return cv2.GaussianBlur(mask, (feather_px | 1, feather_px | 1), feather_px / 2)

    @staticmethod
    def generate_skin_mask(image, exclusion_padding=10, landmarks=None):
        mp_face_mesh = mp.solutions.face_mesh
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.float32)

        # Use provided landmarks, or detect if missing
        if landmarks is None:
            with mp_face_mesh.FaceMesh(static_image_mode=True, refine_landmarks=True) as face_mesh:
                results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                if not results.multi_face_landmarks:
                    print("⚠️ No face landmarks detected!")
                    return mask  # This is just all zeros!
                lm = results.multi_face_landmarks[0].landmark
        else:
            lm = landmarks  # assumed to be list of NormalizedLandmark or np array of shape (468, 3)

        def lm_pt(idx):
            if isinstance(lm[idx], tuple) or isinstance(lm[idx], list) or isinstance(lm[idx], np.ndarray):
                return int(lm[idx][0]), int(lm[idx][1])
            else:
                return int(lm[idx].x * w), int(lm[idx].y * h)

            # === Base skin region
        SKIN_INDICES = [
            10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365,
            379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93,
            234, 127, 162, 21, 54, 103, 67, 109, 66, 107, 55, 65, 52, 53, 67, 69,
            108, 151, 337, 9, 8, 107  # 9, 8 add forehead top
        ]

        skin_pts = np.array([lm_pt(i) for i in SKIN_INDICES], dtype=np.int32)
        skin_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(skin_mask, skin_pts, 255)
        dist_transform = cv2.distanceTransform(skin_mask, cv2.DIST_L2, 5)
        edge_falloff = np.clip(dist_transform / dist_transform.max(), 0, 1)
        mask = np.maximum(mask, edge_falloff.astype(np.float32))

        # === Strong blend zones (cheeks, chin, forehead center)
        core_indices = [205, 424, 50, 280, 10, 152, 9, 8, 151]
        for idx in core_indices:
            cx, cy = lm_pt(idx)
            radius = int(0.12 * h)
            for y in range(max(cy - radius, 0), min(cy + radius, h)):
                for x in range(max(cx - radius, 0), min(cx + radius, w)):
                    dist = np.hypot(x - cx, y - cy)
                    if dist < radius:
                        blend_val = 1.0 - (dist / radius)
                        mask[y, x] = max(mask[y, x], blend_val)

        # Add boost around nose bridge, avoiding tip
        nose_center = lm_pt(6)
        nose_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.ellipse(
            nose_mask,
            nose_center,
            axes=(12, 40),
            angle=0,
            startAngle=0,
            endAngle=360,
            color=255,
            thickness=-1
        )
        nose_mask = cv2.GaussianBlur(nose_mask.astype(np.float32) / 255.0, (31, 31), sigmaX=10)
        mask = np.maximum(mask, nose_mask * 0.6)

        # === Erase regions for eyes, nose, mouth with lower opacity
        def erase_region(indices, pad=10, opacity=0.0):
            pts = np.array([lm_pt(i) for i in indices], np.int32)
            if len(pts) == 0: return
            rect = cv2.boundingRect(pts)
            center = (rect[0] + rect[2] // 2, rect[1] + rect[3] // 2)
            scale = 1 + pad / 100.0
            pts = ((pts - center) * scale + center).astype(np.int32)
            overlay = np.zeros_like(mask)
            cv2.fillConvexPoly(overlay, pts, 1.0)
            mask[overlay == 1] *= opacity

        erase_region([33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246])
        erase_region([362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398])
        erase_region([61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 415])
        erase_region([1, 327], pad=5, opacity=0.25) #blend nose
        erase_region([4, 5, 45, 275, 440, 98], pad=5, opacity=0) #erase nose tip
        erase_region([61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291], pad=5)
        jawline = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]
        erase_region(jawline, pad=5, opacity=0.5)  # More subtle

        anchor = lm[10]  # Between eyebrows
        fx = int(anchor.x * w)
        fy = int(anchor.y * h)

        forehead_radius = int(0.2 * h)
        max_y = max(fy - int(0.25 * h), 0)  # Limit upward extent

        for y in range(max_y, fy):
            dy = fy - y
            for x in range(w):
                dx = x - fx
                dist = np.hypot(dx, dy)
                if dist >= forehead_radius:
                    continue
                blend_val = 1.0 - (dist / forehead_radius)
                mask[y, x] = max(mask[y, x], blend_val * 0.8)

        return feather_mask(mask, feather_px=40)

    def run(self, original_image, enhanced_image, mask):
        original = self.tensor2np(original_image)
        enhanced = self.tensor2np(enhanced_image)
        mask = self.tensor2np(mask, gray=True)

        if mask.max() <= 1.0:
            mask = (mask * 255.0).astype(np.uint8)

        # Fallback if mask is completely empty
        if np.sum(mask) < 10:
            print("⚠️ Provided mask is empty. Generating fallback mask.")
            with mp.solutions.face_mesh.FaceMesh(static_image_mode=True, refine_landmarks=True) as face_mesh:
                results = face_mesh.process(cv2.cvtColor(original, cv2.COLOR_RGB2BGR))
            if results.multi_face_landmarks:
                base_lm = results.multi_face_landmarks[0].landmark
                mask = BasicSkinBlender.generate_skin_mask(original, landmarks=base_lm)
            else:
                h, w = original.shape[:2]
                mask = np.zeros((h, w), dtype=np.float32)

        # === Step 2: Direct paste using mask ===
        # If mask was passed in, still erase sensitive regions using landmarks
        mask = self.erase_sensitive_regions(original, mask)
        mask = feather_mask(mask)  # optionally feather here if not already
        mask_expanded = np.clip(mask, 0, 1)[..., None]  # [H, W, 1]

        result_rgb = original * (1 - mask_expanded) + enhanced * mask_expanded
        result_rgb = np.clip(result_rgb, 0, 255).astype(np.uint8)

        tensor = self.np2tensor_image(result_rgb)
        mask_tensor = self.np2tensor_mask(mask)

        return (tensor, mask_tensor)

    def tensor2np(self, tensor, gray=False):
        arr = tensor[0].cpu().numpy()
        if arr.ndim == 3 and arr.shape[0] == 3 and not gray:
            arr = arr.transpose(1, 2, 0)
        elif gray:
            arr = arr[0] if arr.ndim == 3 else arr
        arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
        return arr

    # For images (RGB), ComfyUI expects NHWC
    def np2tensor_image(self, image):
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        elif image.ndim == 3 and image.shape[2] == 1:
            image = np.repeat(image, 3, axis=2)

        assert image.shape[2] == 3, f"Expected RGB image, got shape: {image.shape}"
        image = image.astype(np.float32) / 255.0
        return torch.from_numpy(image[None, ...]).float()  # [1, H, W, 3]

    # For masks (grayscale), return [1, 1, H, W]
    def np2tensor_mask(self, mask):
        if mask.ndim == 2:
            mask = mask[None, None, ...]  # [1, 1, H, W]
        elif mask.ndim == 3 and mask.shape[0] == 1:
            mask = mask[None, ...]  # [1, 1, H, W]
        return torch.from_numpy(mask.astype(np.float32) / 255.0)

    def erase_sensitive_regions(self, image, mask):
        mask = mask.astype(np.float32) / 255.0  # Normalize and enable float math
        h, w = image.shape[:2]
        with mp.solutions.face_mesh.FaceMesh(static_image_mode=True, refine_landmarks=True) as face_mesh:
            results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            if not results.multi_face_landmarks:
                return mask  # can't erase if no landmarks
            lm = results.multi_face_landmarks[0].landmark

            def lm_pt(idx):
                return int(lm[idx].x * w), int(lm[idx].y * h)

            def erase(indices, pad=10, opacity=0.0):
                pts = np.array([lm_pt(i) for i in indices], np.int32)
                if len(pts) == 0: return
                rect = cv2.boundingRect(pts)
                center = (rect[0] + rect[2] // 2, rect[1] + rect[3] // 2)
                scale = 1 + pad / 100.0
                pts = ((pts - center) * scale + center).astype(np.int32)
                overlay = np.zeros_like(mask)
                cv2.fillConvexPoly(overlay, pts, 1.0)
                mask[overlay == 1] *= opacity

            # Left eye – inner contour
            erase([133, 173, 157, 158, 159, 160], pad=2, opacity=0.0)

            # Right eye – inner contour
            erase([362, 385, 386, 387, 388, 466], pad=2, opacity=0.0)
            # Inner mouth (tight erase)
            erase([78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 415], pad=1, opacity=0.0)

            # Slight feather around corners (less opacity)
            erase([61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308], pad=2, opacity=0.15)

            # Optional: Leave this one out if causing too much nose loss
            # erase([1, 327], pad=5, opacity=0.25)

            erase([1, 327], pad=5, opacity=0.25)  # nose
#            erase([4, 5, 45, 275, 440, 98], pad=5, opacity=0)  # nose tip
            erase([61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291], pad=5)  # jaw

            return mask


NODE_CLASS_MAPPINGS["BasicSkinBlender"] = BasicSkinBlender
NODE_DISPLAY_NAME_MAPPINGS["BasicSkinBlender"] = "Basic Skin Blender"

