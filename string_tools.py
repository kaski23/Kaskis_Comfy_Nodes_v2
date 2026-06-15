import re

### JSON-String-Tools

class JsonStringTool:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "key": ("STRING", {"default": "", "multiline": False}),
                "value": ("STRING", {"default": "", "multiline": True}),
                "nested": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("JSON-String",)
    FUNCTION = "create_json_string"
    CATEGORY = "KASKI/stringtools"

    def create_json_string(self, key, value, nested):
        key = key.strip().strip('"').rstrip(":").strip()
        value = value.strip()

        if not key:
            return ("",)

        # Value absichern
        if not value:
            value = '""'
        else:
            starts_structured = value[0] in ['"', '{', '[']
            ends_structured = value[-1] in ['"', '}', ']']

            if not starts_structured:
                value = '"' + value

            if not ends_structured:
                value = value + '"'

        # Nested wrapping
        if nested:
            if not value.startswith("{"):
                value = "{\n" + value + "\n}"

        json_string = f'"{key}": {value},\n'

        return (json_string,)



### GENERAL STRING TOOLS


class StringSplitAtSymbol:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": False}),
                "delimiter": ("STRING", {"default": "_"}),
                "index": ("INT", {"default": 0, "min": 0, "max": 1000, "step": 1}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("selected string",)
    FUNCTION = "split_and_select"
    CATEGORY = "KASKI/stringtools"

    def split_and_select(self, text: str, delimiter: str, index: int):
        if not delimiter:
            raise ValueError(f"KASKI-Nodes: no delimiter specified")   # if no delimiter is specified, kill the whole thing

        parts = text.split(delimiter)
        if 0 <= index < len(parts):
            return (parts[index],)
        else:
            return ("",)

           
class JoinStrings:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "string_a": ("STRING",),
                "string_b": ("STRING",),
                "delimiter": ("STRING", {"default": "_", "multiline": False}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("joined string",)
    FUNCTION = "join"
    CATEGORY = "KASKI/stringtools"

    def join(self, string_a: str, string_b: str, delimiter: str):
        out = f"{string_a}{delimiter}{string_b}"
        return (out,)

class NumberToString:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "number_int": ("INT",),
                "number_float": ("FLOAT",),
                "mode": (["INT", "FLOAT"],),
                "zero_padding": ("INT", {"default": 0, "min": 0, "max": 15, "step": 1}),
                "decimal_places": ("INT", {"default": 2, "min": 0, "max": 10, "step": 1}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("string",)
    FUNCTION = "convert"
    CATEGORY = "KASKI/stringtools"

    def convert(
        self,
        number_int: int,
        number_float: float,
        mode: str,
        zero_padding: int,
        decimal_places: int,
    ):
        if mode == "INT":
            out = f"{number_int:0{zero_padding}d}"
        else:
            out = f"{number_float:0{zero_padding}.{decimal_places}f}"

        return (out,)



# MAPPING-DICTS

STRING_TOOLS_NODE_CLASS_MAPPINGS = {
    "JsonStringTool_KASKI": JsonStringTool,

    "StringSplitAtSymbol_KASKI": StringSplitAtSymbol,
    "JoinStrings_KASKI": JoinStrings,
    "NumberToString_KASKI": NumberToString,
}
    
STRING_TOOLS_NODE_DISPLAY_NAME_MAPPINGS = {
    "JsonStringTool_KASKI": "JSON Key-Value String",

    "StringSplitAtSymbol_KASKI": "String Split at Symbol",
    "JoinStrings_KASKI": "Join Strings",
    "NumberToString_KASKI": "Number to String",
}