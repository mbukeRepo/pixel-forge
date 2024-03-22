import torch
from config import Config, CAPTION_MODELS
import logging
import open_clip
import time

logging.basicConfig(
	level=logging.DEBUG,
	format='%(name)s - %(levelname)s - %(message)s'
)

class PixelForge():
	def __init__(self, config: Config):
		self.config = config
		self.dtype = torch.float16 if self.device == 'cuda' else torch.float32 # TODO: revise more on the appropriate dtype

		# loading caption and clip models
		self.load_caption_model()
		self.load_clip_model()

	def load_caption_model(self):
		logging.info(f"Loading caption model {self.config.caption_model_name}...")
		model_path = CAPTION_MODELS[self.config.caption_model_name]

		if self.config.caption_model_name.startswith('blip2-'):
			caption_model = Blip2ForConditionalGeneration.from_pretrained(model_path, torch_dtype=self.dtype)
		else:
			caption_model = BlipForConditionalGeneration.from_pretrained(model_path, torch_dtype=self.dtype)

		self.caption_processor = AutoProcessor.from_pretrained(model_path)
		caption_model.eval()
		self.caption_model = caption_model

	def load_clip_model(self):
		logging.info(f"Loading CLIP model {config.clip_model_name}...")
		clip_model_name, clip_model_pretrained_name = config.clip_model_name.split('/', 2)
		start_time = time.time()

		self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
                	clip_model_name,
                	pretrained = clip_model_pretrained_name,
                	precision = 'fp16' if self.config.device == 'cuda' else 'fp32',
                	device = self.config.device,
                	jit = False,
                	cache_dir = self.config.clip_model_path
            	)
            	self.clip_model.eval()
		# clip tokenizer
		self.tokenize = open_clip.get_tokenizer(clip_model_name)
		# TODO: load prompt helpers.

		end_time = time.time()
		logging.info(f"Loaded CLIP model and data in {end_time-start_time:.2f} seconds.")
