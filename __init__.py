from .loaders import *
from .input_conformer import *
from .string_tools import STRING_TOOLS_NODE_CLASS_MAPPINGS, STRING_TOOLS_NODE_DISPLAY_NAME_MAPPINGS
from .id_tools import ID_TOOLS_NODE_CLASS_MAPPINGS, ID_TOOLS_NODE_DISPLAY_NAME_MAPPINGS
from .async_tools import *
from .nanobanana_pro_rewrite import NANOBANANA_REWRITE_NODE_CLASS_MAPPINGS, NANOBANANA_REWRITE_NODE_DISPLAY_NAME_MAPPINGS
from. gptimage_rewrite import OPENAI_GPT_IMAGE_REWRITE_NODE_CLASS_MAPPINGS, OPENAI_GPT_IMAGE_REWRITE_NODE_DISPLAY_NAME_MAPPINGS


NODE_CLASS_MAPPINGS = {
    **STRING_TOOLS_NODE_CLASS_MAPPINGS,
    **ID_TOOLS_NODE_CLASS_MAPPINGS,
    **NANOBANANA_REWRITE_NODE_CLASS_MAPPINGS,
    **OPENAI_GPT_IMAGE_REWRITE_NODE_CLASS_MAPPINGS,
    
    "VideoSizeLengthConformer_KASKI": VideoSizeLengthConformer,
    "WanVaceInputConform_KASKI": WanVaceInputConform,
    
    "LoadVideoWithFilename_KASKI": LoadVideoWithFilename,
    "LoadImageWithFilename_KASKI": LoadImageWithFilename,
    
    "AsyncDelay_KASKI": AsyncDelay,

}


NODE_DISPLAY_NAME_MAPPINGS = {
    **STRING_TOOLS_NODE_DISPLAY_NAME_MAPPINGS,
    **ID_TOOLS_NODE_DISPLAY_NAME_MAPPINGS,
    **NANOBANANA_REWRITE_NODE_DISPLAY_NAME_MAPPINGS,
    **OPENAI_GPT_IMAGE_REWRITE_NODE_DISPLAY_NAME_MAPPINGS,
    
    "VideoSizeLengthConformer_KASKI": "Conform Video Size and Length",
    "WanVaceInputConform_KASKI": "Conform Video for Wan 2.1",
    
    "LoadVideoWithFilename_KASKI": "Load Video with Filename",
    "LoadImageWithFilename_KASKI": "Load Image with Filename",
    
    "AsyncDelay_KASKI": "Async Delay",
}