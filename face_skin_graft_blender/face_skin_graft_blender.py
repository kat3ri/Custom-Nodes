import torch
import numpy as np
import cv2
import json
from PIL import Image
from pathlib import Path

from nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS


class FaceSkinGraftBlender:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "original_image": ("IMAGE",),
                "enhanced_image": ("IMAGE",),
                "homography_json": ("STRING", {"multiline": False}),
                "strength": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 10.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("blended_image",)
    FUNCTION = "run"

    CATEGORY = "image/face_tools"

    def generate_skin_mask(image):
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.float32)

        with mp.solutions.face_mesh.FaceMesh(static_image_mode=True, refine_landmarks=True) as face_mesh:
            results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            if not results.multi_face_landmarks:
                return mask  # early fallback

            lm = results.multi_face_landmarks[0].landmark

            def lm_pt(idx):
                return int(lm[idx].x * w), int(lm[idx].y * h)

            # === Base skin region
            SKIN_INDICES = [
                10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365,
                379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93,
                234, 127, 162, 21, 54, 103, 67, 109, 66, 107, 55, 65, 52, 53
            ]
            skin_points = np.array([lm_pt(i) for i in SKIN_INDICES])
            if len(skin_points) > 0:
                hull = cv2.convexHull(skin_points)
                cv2.fillConvexPoly(mask, hull, 1.0)

            # === Forehead soft fill
            fx, fy = lm_pt(10)
            forehead_radius = int(0.2 * h)
            for y in range(max(fy - int(0.25 * h), 0), fy):
                for x in range(w):
                    dx = abs(x - fx)
                    dy = fy - y
                    dist = np.sqrt(dx ** 2 + dy ** 2)
                    if dist < forehead_radius:
                        blend_val = 1.0 - (dist / forehead_radius)
                        mask[y, x] = max(mask[y, x], blend_val * 0.8)

            # === Erase unwanted regions
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

            erase_region([33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246])  # Left eye
            erase_region([362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398])  # Right eye
            erase_region([
                61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308,
                78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 415
            ], pad=5)  # Mouth
            erase_region([1, 2, 98, 327, 195, 5, 4, 275, 440], pad=5, opacity=0.25)  # Nose
            erase_region([61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291], pad=5)  # Around nose/center

            return cv2.GaussianBlur(mask, (41, 41), 0)

    def run(self, original_image, enhanced_image, homography_json, strength):
        from mediapipe import solutions as mp_solutions

        original = self.tensor2np(original_image)
        enhanced = self.tensor2np(enhanced_image)

        # Load homography
        h, w = original.shape[:2]

        # If no homography is provided, skip warp
        if homography_json.strip().lower() in ["", "none"]:
            warped_enhanced = enhanced.copy()
        else:
            try:
                if Path(homography_json).exists():
                    with open(homography_json, 'r') as f:
                        data = json.load(f)
                else:
                    data = json.loads(homography_json)

                H_inv = np.array(data['H_inv'], dtype=np.float32)
                warped_enhanced = cv2.warpPerspective(enhanced, H_inv, (w, h), flags=cv2.INTER_LANCZOS4)

            except Exception as e:
                print(f"[WARN] Failed to load homography, skipping warp: {e}")
                warped_enhanced = enhanced.copy()

        # === Generate skin mask ===
        mask = generate_skin_mask(original)


        # === LAB Detail Transfer ===
        lab = cv2.cvtColor(warped_enhanced, cv2.COLOR_RGB2LAB)
        L, _, _ = cv2.split(lab)
        smooth_L = cv2.GaussianBlur(L, (0, 0), 2.5)
        detail_map = L.astype(np.float32) - smooth_L.astype(np.float32)

        lab_orig = cv2.cvtColor(original, cv2.COLOR_RGB2LAB)
        L_orig, A_orig, B_orig = cv2.split(lab_orig)

        mask_bin = (mask > 0.6).astype(np.uint8)
        dist_transform = cv2.distanceTransform(mask_bin, distanceType=cv2.DIST_L2, maskSize=5)
        combined_mask = np.clip(dist_transform / 20.0, 0, 1).astype(np.float32)

        L_detail = L_orig.astype(np.float32) + detail_map * combined_mask * strength
        L_detail = np.clip(L_detail, 0, 255).astype(np.uint8)

        result = cv2.merge([L_detail, A_orig, B_orig])
        result_rgb = cv2.cvtColor(result, cv2.COLOR_LAB2RGB)

        return (self.np2tensor(result_rgb),)

    def tensor2np(self, tensor):
        image = tensor[0].cpu().numpy()
        image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
        return image.transpose(1, 2, 0).copy()

    def np2tensor(self, image):
        image = image.astype(np.float32) / 255.0
        image = image.transpose(2, 0, 1)[None, ...]
        return torch.from_numpy(image)


NODE_CLASS_MAPPINGS["FaceSkinGraftBlender"] = FaceSkinGraftBlender
NODE_DISPLAY_NAME_MAPPINGS["FaceSkinGraftBlender"] = "Face Skin Graft Blender"
