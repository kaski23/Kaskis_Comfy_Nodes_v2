from .loaders import *
from .input_conformer import *
from .string_tools import STRING_TOOLS_NODE_CLASS_MAPPINGS, STRING_TOOLS_NODE_DISPLAY_NAME_MAPPINGS
from .async_tools import *
from .nanobanana_pro_rewrite import *


NODE_CLASS_MAPPINGS = {
    **STRING_TOOLS_NODE_CLASS_MAPPINGS,
    
    "VideoSizeLengthConformer_KASKI": VideoSizeLengthConformer,
    "WanVaceInputConform_KASKI": WanVaceInputConform,
    
    "LoadVideoWithFilename_KASKI": LoadVideoWithFilename,
    "LoadImageWithFilename_KASKI": LoadImageWithFilename,
    
    "AsyncDelay_KASKI": AsyncDelay,
    
    "NanobananaPro_KASKI": GeminiImage2,
    "NanobananaSettings_KASKI": GeminiSettings,
    
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **STRING_TOOLS_NODE_DISPLAY_NAME_MAPPINGS,
    
    "VideoSizeLengthConformer_KASKI": "Conform Video Size and Length",
    "WanVaceInputConform_KASKI": "Conform Video for Wan 2.1",
    
    "LoadVideoWithFilename_KASKI": "Load Video with Filename",
    "LoadImageWithFilename_KASKI": "Load Image with Filename",
    
    "AsyncDelay_KASKI": "Async Delay",
    
    "NanobananaPro_KASKI": "Nanobanana Pro IO-unlocked",
    "NanobananaSettings_KASKI": "Nanobanana Settings",
}