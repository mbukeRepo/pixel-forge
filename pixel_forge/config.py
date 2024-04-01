from dataclasses import dataclass
import os
import torch
from typing import Optional

# available caption models
CAPTION_MODELS = {
    'blip-base': 'Salesforce/blip-image-captioning-base',  # 990MB
    'blip-large': 'Salesforce/blip-image-captioning-large',  # 1.9GB
    'blip2-2.7b': 'Salesforce/blip2-opt-2.7b',  # 15.5GB
    'blip2-flan-t5-xl': 'Salesforce/blip2-flan-t5-xl',  # 15.77GB
}


@dataclass
class Config:
    caption_model = None
    caption_processor = None
    clip_model = None
    clip_preprocessor = None

    # blip settings
    caption_max_length: int = 32
    caption_model_name: Optional[str] = 'blip-base'  # use a key from CAPTION_MODELS or None
    caption_offload: bool = False

    # clip settings
    clip_model_name: str = 'ViT-L-14/openai'
    clip_model_path: Optional[str] = None
    clip_offload: bool = False

    # pixel-forge settings
    cache_path = "cache"  # where to store the text embeddings
    chunk_size = 2048  # batch size for CLIP! use smaller for low RAM consumption
    prompt_helpers_path = os.path.join(os.path.dirname(__file__), 'prompt_helpers')  # prompt helpers
    device = "cuda" if torch.cuda.is_available() else "cpu"
    flavor_intermediate_count: int = 2048

# TODO: apply low vram
