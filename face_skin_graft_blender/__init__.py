from .basic_skin_blender import BasicSkinBlender
from .face_align_warp_node import FaceAlignWarpNode  # If in separate file


NODE_CLASS_MAPPINGS = {
    "BasicSkinBlender": BasicSkinBlender,
    "FaceAlignWarpNode": FaceAlignWarpNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BasicSkinBlender": "Basic Skin Blender",
    "FaceAlignWarpNode": "Face Align Warp"
}






