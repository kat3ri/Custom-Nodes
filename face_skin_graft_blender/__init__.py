from .face_skin_graft_blender import FaceSkinGraftBlender
from .basic_skin_blender import BasicSkinBlender
from .face_align_warp_node import FaceAlignWarpNode  # If in separate file
from .feature_refine_node import  FeatureRefineWarpNode

NODE_CLASS_MAPPINGS = {
    "FaceSkinGraftBlender": FaceSkinGraftBlender,
    "BasicSkinBlender": BasicSkinBlender,
    "FeatureRefineWarpNode": FeatureRefineWarpNode,
    "FaceAlignWarpNode": FaceAlignWarpNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FaceSkinGraftBlender": "Face Skin Graft Blender",
    "BasicSkinBlender": "Basic Skin Blender",
    "FeatureRefineWarpNode": "Feature Refinement Warp",
    "FaceAlignWarpNode": "Face Align Warp"
}





