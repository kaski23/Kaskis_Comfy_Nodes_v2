import re

### REFERENCE-ID-TOOLS

REGEX_REFERENCE_ID = re.compile(
    r"(?:[A-Za-z0-9-]+_)?(?:character|prop|location|material)_[A-Za-z0-9-]+_v(?:[0-9]+|N)"
)

class GenerateReferenceID:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "project_name": ("STRING", {"default": "NONE", "multiline": False}),
                "reference_type": (["character", "prop", "location", "material"],),
                "reference_name": ("STRING", {"default": "", "multiline": False}),
                "version": ("INT", {"default": -1, "min": -1, "max": 10000, "step": 1}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("generated ID",)
    FUNCTION = "generate"
    CATEGORY = "KASKI/ID-Tools"

    def generate(
        self,
        project_name: str,
        reference_type: str,
        reference_name: str,
        version: int,
    ):
        project_name = project_name.strip()
        reference_name = reference_name.strip()

        if "_" in project_name:
            raise ValueError(
                f"KASKI-Nodes: project_name must not contain underscores: {project_name}"
            )

        if "_" in reference_name:
            raise ValueError(
                f"KASKI-Nodes: reference_name must not contain underscores: {reference_name}"
            )
            
        if " " in project_name:
            raise ValueError(
                f"KASKI-Nodes: project_name must not contain spaces: {project_name}"
            )

        if " " in reference_name:
            raise ValueError(
                f"KASKI-Nodes: reference_name must not contain spaces: {reference_name}"
            )

        if reference_name == "":
            raise ValueError("KASKI-Nodes: reference_name cannot be empty")

        if version != -1:
            version_string = f"v{version}"
        else:
            version_string = "vN"

        if project_name == "" or project_name == "NONE":
            out = f"{reference_type}_{reference_name}_{version_string}"
        else:
            out = f"{project_name}_{reference_type}_{reference_name}_{version_string}"

        return (out,)
        
        
class ExtractReferenceID:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": False}),
                "fail_if_not_found": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("extracted ID",)
    FUNCTION = "extract"
    CATEGORY = "KASKI/ID-Tools"

    def extract(self, text: str, fail_if_not_found: bool):
        match = REGEX_REFERENCE_ID.search(text)

        if not match:
            if fail_if_not_found:
                raise ValueError(f"KASKI-Nodes: Couldn't extract Reference ID from: {text}")
            else:
                return (text,)

        return (match.group(0),)




### SHOT-ID-TOOLS

REGEX_ID = re.compile(
    r"(?:[A-Za-z0-9-]+_)?sh[0-9]+_(?:firstFrame|notEnhanced|enhanced|Depth|Normal|Scribble|lastFrame)_v(?:[0-9]+|N)"
)



class GenerateShotID:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "project_name": ("STRING", {"default": "NONE", "multiline": False}),
                "shot_no": ("INT", {"default": 0, "min": 0, "max": 10000, "step": 1}),
                "pipeline_step": (["firstFrame", "notEnhanced", "enhanced", "Depth", "Normal", "Scribble", "lastFrame"],),
                "version": ("INT", {"default": -1, "min": -1, "max": 10000, "step": 1}),
                "shot_no_zero_padding": ("INT", {"default": 3, "min": 0, "max": 15, "step": 1}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("generated ID",)
    FUNCTION = "generate"
    CATEGORY = "KASKI/ID-Tools"

    def generate(
        self,
        project_name: str,
        shot_no: int,
        pipeline_step: str,
        version: int,
        shot_no_zero_padding: int
    ):
        project_name = project_name.strip()

        if "_" in project_name:
            raise ValueError(f"KASKI-Nodes: project_name must not contain underscores: {project_name}")

        if " " in project_name:
            raise ValueError(f"KASKI-Nodes: project_name must not contain spaces: {project_name}")

        if version != -1:
            version_string = f"v{version}"
        else:
            version_string = "vN"

        shot_no_padded = f"{shot_no:0{shot_no_zero_padding}d}"

        if project_name == "" or project_name == "NONE":
            out = f"sh{shot_no_padded}_{pipeline_step}_{version_string}"
        else:
            out = f"{project_name}_sh{shot_no_padded}_{pipeline_step}_{version_string}"

        return (out,)
        


class ModifyShotID:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "idx": ("STRING", {"multiline": False}),
                "project_name": ("STRING", {"default": "KEEP", "multiline": False}),
                "shot_no": ("INT", {"default": -1, "min": -1, "max": 10000, "step": 1}),
                "pipeline_step": (["KEEP", "firstFrame", "notEnhanced", "enhanced", "Depth", "Normal", "Scribble", "lastFrame"],),
                "version": ("INT", {"default": -1, "min": -1, "max": 10000, "step": 1}),
                "shot_no_zero_padding": ("INT", {"default": 3, "min": 0, "max": 15, "step": 1}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("modified ID",)
    FUNCTION = "modify"
    CATEGORY = "KASKI/ID-Tools"

    def modify(
        self,
        idx: str,
        project_name: str,
        shot_no: int,
        pipeline_step: str,
        version: int,
        shot_no_zero_padding: int,
    ):
        
        
        if not REGEX_ID.fullmatch(idx):
            return (idx,)
            
        project_name = project_name.strip()

        if "_" in project_name:
            raise ValueError(f"KASKI-Nodes: project_name must not contain underscores: {project_name}")

        if " " in project_name:
            raise ValueError(f"KASKI-Nodes: project_name must not contain spaces: {project_name}")

        parts = idx.split("_")

        # Possible structures:
        # [sh###, pipeline_step, v#]
        # [project_name, sh###, pipeline_step, v#]

        if len(parts) == 3:
            old_project_name = ""
            old_shot, old_pipeline_step, old_version = parts

        elif len(parts) == 4:
            old_project_name, old_shot, old_pipeline_step, old_version = parts

        else:
            raise ValueError(f"KASKI-Nodes: Malformed ID (split failed): {idx}")

        # --- PROJECT NAME ---
        if project_name != "KEEP":
            if project_name == "" or project_name == "NONE":
                old_project_name = ""
            else:
                old_project_name = project_name

        # --- SHOT NO ---
        if shot_no != -1:
            shot_no_padded = f"{shot_no:0{shot_no_zero_padding}d}"
            old_shot = f"sh{shot_no_padded}"

        # --- PIPELINE STEP ---
        if pipeline_step != "KEEP":
            old_pipeline_step = pipeline_step

        # --- VERSION ---
        if version != -1:
            old_version = f"v{version}"

        # --- REBUILD ---
        if old_project_name == "":
            out = f"{old_shot}_{old_pipeline_step}_{old_version}"
        else:
            out = f"{old_project_name}_{old_shot}_{old_pipeline_step}_{old_version}"

        return (out,)


class ExtractShotID:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": False}),
                "fail_if_not_found": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("extracted ID",)
    FUNCTION = "extract"
    CATEGORY = "KASKI/ID-Tools"

    def extract(self, text: str, fail_if_not_found: bool):
        match = REGEX_ID.search(text)

        if not match:
            if fail_if_not_found:
                raise ValueError(f"KASKI-Nodes: Couldn't extract ID from: {text}")
            else:
                return (text,)

        return (match.group(0),)
        
        


# MAPPING-DICTS

ID_TOOLS_NODE_CLASS_MAPPINGS = {   
    "GenerateReferenceID_KASKI": GenerateReferenceID,
    "ExtractReferenceID_KASKI": ExtractReferenceID,
    
    "ExtractShotID_KASKI": ExtractShotID,
    "GenerateShotID_KASKI": GenerateShotID,
    "ModifyShotID_KASKI": ModifyShotID,
}
    
ID_TOOLS_NODE_DISPLAY_NAME_MAPPINGS = {   
    "GenerateReferenceID_KASKI": "Generate Reference ID",
    "ExtractReferenceID_KASKI": "Extract Reference ID",
    
    "ExtractShotID_KASKI": "Extract Shot ID",
    "GenerateShotID_KASKI": "Generate Shot ID",
    "ModifyShotID_KASKI": "Modify Shot ID",
}