import torch
from config import Config, CAPTION_MODELS
import logging
import open_clip
import time
from typing import List

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
		start_time = time.time()
		model_path = CAPTION_MODELS[self.config.caption_model_name]

		if self.config.caption_model_name.startswith('blip2-'):
			caption_model = Blip2ForConditionalGeneration.from_pretrained(model_path, torch_dtype=self.dtype)
		else:
			caption_model = BlipForConditionalGeneration.from_pretrained(model_path, torch_dtype=self.dtype)

		self.caption_processor = AutoProcessor.from_pretrained(model_path)
		caption_model.eval()
		self.caption_model = caption_model
		end_time = time.time()
		logging.info(f"Loaded caption model in {end_time-start_time:.2f} seconds.")

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

		# load popular image sharing platforms
		platforms = [
			'dribbble', 'flickr', 'instagram', 'pexels', 'pinterest', 'pixabay', 'pixiv', 'polycount', 'reddit', 'shutterstock', 'tumblr', 'unsplash', 'zbrush central'
		]

		trending_list = [platform for platform in platforms]
		trending_list.extend([f"trending on {site}" for site in platforms])
		trending_list.extend([f"featured on {site}" for site in platforms])

		# loading artists
		raw_artists = load_list(config.data_path, "artists.txt")
		artists = [f"by {artist}" for artist in raw_artists]
		artists.extend([f"inspired by {artist}" for artist in raw_artists])

		# preparing clip model
		self.clip_model = self.clip_model.to(self.device)

		self.artists = []
		self.flavors = []
		self.mediums = []
		self.movements = []
		self.trendings = []
		self.negative = []

		end_time = time.time()
		logging.info(f"Loaded CLIP model and prompt helpers in {end_time-start_time:.2f} seconds.")


class LabelTable():
	def __init__(self, labels: List[str], interrogator: PixelForge):
		clip_model, config = interrogator.clip_model, interrogator.config
		self.chunk_size = config.chunk_size
		self.config = config
		self.device = config.device
		self.embeddings = []
		self.labels = labels
		self.tokenize = interrogator.tokenize

		hash = hashlib.sha256(",".join(labels).encode()).hexdigest()
		sanitized_name = self.config.clip_model_name.replace('/', '_').replace('@', '_')

	def _load_cached(self, desc: str, hash: str, interrogator: PixelForge):
		pass

	def _rank(self, image_features: torch.Tensor, text_embeds: torch.Tensor, top_count: int = 1, reverse: bool = False):
		pass

	def rank(self, image_features: torch.Tensor, top_count: int = 1, reverse: bool = False):
		if len(self.labels) <= self.chunk_size:
			tops = self._rank(
				image_features,
				self.embeddings,
				top_count = top_count,
				reverse = reverse
			)


