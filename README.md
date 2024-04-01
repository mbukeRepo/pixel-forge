# pixel-forge

Looking for a good question to create new images similar to an existing one? Use PixelForge. 

### Usage

Install all dependencies: 
```
   pip install -r requirements.txt
```

You can now use it in your script. 
```python
from PIL import Image
from pixel_forge import PixelForge, Config

clip_model_name = "ViT-L-14/openai"
image_path = "your image path"
image = Image.open(image_path).convert("RGB")
px = PixelForge(Config(clip_model_name=clip_model_name))
print(px.interrogate(image)) # best image prompt 
```

### Configuration

The `Config` object lets you configure CLIP Interrogator's processing. 
* `clip_model_name`: which of the OpenCLIP pretrained CLIP models to use
* `cache_path`: path where to save precomputed text embeddings
* `download_cache`: when True will download the precomputed embeddings from huggingface
* `chunk_size`: batch size for CLIP, use smaller for lower VRAM
* `quiet`: when True no progress bars or text output will be displayed



