import hashlib
import logging
import math
import os
import time
from typing import List, Optional

import numpy as np
import open_clip
import torch
from PIL import Image
from safetensors.numpy import load_file
from tqdm import tqdm
from transformers import AutoProcessor, Blip2ForConditionalGeneration, BlipForConditionalGeneration

from config import Config, CAPTION_MODELS
from utils import download_file, load_list, prompt_at_max_len

logging.basicConfig(
    level=logging.DEBUG,
    format='%(name)s - %(levelname)s - %(message)s'
)

CACHE_BASE_URL = 'https://huggingface.co/pharmapsychotic/ci-preprocess/resolve/main/'


class PixelForge:
    def __init__(self, config: Config):
        self.config = config
        self.dtype = torch.float16 if self.config.device == 'cuda' else torch.float32  # TODO: revise more on the

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
        logging.info(f"Loaded caption model in {end_time - start_time:.2f} seconds.")

    def load_clip_model(self):
        logging.info(f"Loading CLIP model {self.config.clip_model_name}...")
        clip_model_name, clip_model_pretrained_name = self.config.clip_model_name.split('/', 2)
        start_time = time.time()

        self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
            clip_model_name,
            pretrained=clip_model_pretrained_name,
            precision='fp16' if self.config.device == 'cuda' else 'fp32',
            device=self.config.device,
            jit=False,
            cache_dir=self.config.clip_model_path
        )
        self.clip_model.eval()
        # clip tokenizer
        self.tokenize = open_clip.get_tokenizer(clip_model_name)

        # load popular image sharing platforms
        platforms = [
            'Artstation', 'behance', 'cg society', 'cgsociety', 'deviantart', 'dribbble',
            'flickr', 'instagram', 'pexels', 'pinterest', 'pixabay', 'pixiv', 'polycount',
            'reddit', 'shutterstock', 'tumblr', 'unsplash', 'zbrush central'
        ]

        trending_list = [platform for platform in platforms]
        trending_list.extend([f"trending on {site}" for site in platforms])
        trending_list.extend([f"featured on {site}" for site in platforms])

        # loading artists
        raw_artists = load_list(self.config.prompt_helpers_path, "artists.txt")
        artists = [f"by {artist}" for artist in raw_artists]
        artists.extend([f"inspired by {artist}" for artist in raw_artists])

        # preparing clip model
        self.clip_model = self.clip_model.to(self.config.device)

        self.artists = LabelTable(artists, "artists", self)
        self.flavors = LabelTable(load_list(self.config.prompt_helpers_path, 'flavors.txt'), "flavors", self)
        self.mediums = LabelTable(load_list(self.config.prompt_helpers_path, 'mediums.txt'), "mediums", self)
        self.movements = LabelTable(load_list(self.config.prompt_helpers_path, 'movements.txt'), "movements", self)
        self.trendings = LabelTable(trending_list, "trendings", self)
        self.negative = LabelTable(load_list(self.config.prompt_helpers_path, 'negative.txt'), "negative", self)

        end_time = time.time()
        logging.info(f"Loaded CLIP model and prompt helpers in {end_time - start_time:.2f} seconds.")

    def generate_caption(self, image: Image):
        self.caption_model = self.caption_model.to(self.config.device)

        inputs = self.caption_processor(images=image, return_tensors="pt").to(self.config.device)
        tokens = self.caption_model.generate(**inputs, max_new_tokens=self.config.caption_max_length)

        return self.caption_processor.batch_decode(tokens, skip_special_tokens=True)[0].strip()

    def image_to_features(self, image: Image):
        self.clip_model = self.clip_model.to(self.config.device)
        images = self.clip_preprocess(image).unsqueeze(0).to(self.config.device)

        with torch.no_grad(), torch.cuda.amp.autocast():
            image_features = self.clip_model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)

        return image_features

    def chain(
            self,
            image_features: torch.Tensor,
            phrases: List[str],
            best_prompt: str = "",
            best_sim: float = 0,
            min_count: int = 8,
            max_count: int = 32,
            desc="Chaining",
            reverse: bool = False
    ) -> str:
        self.clip_model = self.clip_model.to(self.config.device)
        phrases = set(phrases)
        if not best_prompt:
            best_prompt = self.rank_top(image_features, [f for f in phrases], reverse=reverse)
            best_sim = self.similarity(image_features, best_prompt)
            phrases.remove(best_prompt)
        curr_prompt, curr_sim = best_prompt, best_sim

        def check(addition: str, idx: int) -> bool:
            nonlocal best_prompt, best_sim, curr_prompt, curr_sim
            prompt = curr_prompt + ", " + addition
            sim = self.similarity(image_features, prompt)
            if reverse:
                sim = -sim

            if sim > best_sim:
                best_prompt, best_sim = prompt, sim
            if sim > curr_sim or idx < min_count:
                curr_prompt, curr_sim = prompt, sim
                return True
            return False

        for idx in tqdm(range(max_count), desc=desc, disable=self.config.quiet):
            best = self.rank_top(image_features, [f"{curr_prompt}, {f}" for f in phrases], reverse=reverse)
            flave = best[len(curr_prompt) + 2:]
            if not check(flave, idx):
                break
            if prompt_at_max_len(curr_prompt, self.tokenize):
                break
            phrases.remove(flave)

        return best_prompt

    def interrogate(self, image: Image, min_flavors: int = 8, max_flavors: int = 32,
                    caption: Optional[str] = None) -> str:
        if caption is None:
            caption = self.generate_caption(image)

        image_features = self.image_to_features(image)
        # merge label tables
        new_table = LabelTable([], None, self)
        for table in [self.artists, self.flavors, self.mediums, self.movements, self.trendings]:
            new_table.labels.extend(table.labels)
            new_table.embeddings.extend(table.embeddings)
        phrases = new_table.rank(image_features, top_count=self.config.flavor_intermediate_count)

        best_prompt, best_sim = caption, self.similarity(image_features, caption)
        best_prompt = self.chain(
            image_features,
            phrases,
            best_prompt,
            best_sim,
            min_flavors,
            max_flavors,
            desc="Flavor chain"
        )
        # warn: less readable interrogate TODO: improve with more variations
        candidates = [caption, best_prompt]
        return candidates[np.argmax(self.similarities(image_features, candidates))]

    def rank_top(self, image_features: torch.Tensor, text_array: List[str], reverse: bool = False) -> str:
        self.clip_model = self.clip_model.to(self.config.device)
        text_tokens = self.tokenize([text for text in text_array]).to(self.config.device)
        with torch.no_grad(), torch.cuda.amp.autocast():
            text_features = self.clip_model.encode_text(text_tokens)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            similarity = text_features @ image_features.T
            if reverse:
                similarity = -similarity
        return text_array[similarity.argmax().item()]

    def similarity(self, image_features: torch.Tensor, text: str) -> float:
        self.clip_model = self.clip_model.to(self.config.device)
        text_tokens = self.tokenize([text]).to(self.config.device)
        with torch.no_grad(), torch.cuda.amp.autocast():
            text_features = self.clip_model.encode_text(text_tokens)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            similarity = text_features @ image_features.T
        return similarity[0][0].item()

    def similarities(self, image_features: torch.Tensor, text_array: List[str]) -> List[float]:
        self.clip_model = self.clip_model.to(self.config.device)
        text_tokens = self.tokenize([text for text in text_array]).to(self.config.device)
        with torch.no_grad(), torch.cuda.amp.autocast():
            text_features = self.clip_model.encode_text(text_tokens)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            similarity = text_features @ image_features.T
        return similarity.T[0].tolist()


class LabelTable:
    def __init__(self, labels: List[str], desc: Optional[str], interrogator: PixelForge):
        clip_model, config = interrogator.clip_model, interrogator.config
        self.chunk_size = config.chunk_size
        self.config = config
        self.device = config.device
        self.embeddings = []
        self.labels = labels
        self.tokenize = interrogator.tokenize

        hash_str = hashlib.sha256(",".join(labels).encode()).hexdigest()
        self.sanitized_name = self.config.clip_model_name.replace('/', '_').replace('@', '_')
        self.load_checkpoints(desc=desc, hash_str=hash_str)

    def load_checkpoints(self, desc: str, hash_str: str):
        """
          Loads the checkpoints and initialize the embedding table,
          :param desc: description of the checkpoints
          :param hash_str: hash of the label keywords
        """
        cached_safetensors = os.path.join(self.config.cache_path, f"{self.sanitized_name}_{desc}.safetensors")

        if not os.path.exists(cached_safetensors):
            download_url = CACHE_BASE_URL + cached_safetensors
            try:
                os.makedirs(os.path.dirname(cached_safetensors), exist_ok=True)
                download_file(download_url, download_url)
            except Exception as e:
                logging.error(f'failed to download {download_url}')
                logging.error(f'error message: {e}')
        try:
            tensors = load_file(cached_safetensors)
            if 'hash' in tensors and 'embeds' in tensors:
                if np.array_equal(tensors['hash'], np.array([ord(c) for c in hash_str], dtype=np.int8)):
                    self.embeddings = tensors['embeds']
                    if len(self.embeddings.shape) == 2:
                        self.embeddings = [self.embeddings[i] for i in range(self.embeddings.shape[0])]
                    return True
        except Exception as e:
            logging.error(f'error loading {cached_safetensors}')
            logging.error(f'error message: {e}')

    def _rank(self, image_features: torch.Tensor, text_embeds: torch.Tensor, top_count: int=1, reverse: bool=False) -> List[str]:
        top_count = min(top_count, len(text_embeds))
        text_embeds = torch.stack([torch.from_numpy(t) for t in text_embeds]).to(self.device)
        with torch.cuda.amp.autocast():
            similarity = image_features @ text_embeds.T
            if reverse:
                similarity = -similarity
        _, top_labels = similarity.float().cpu().topk(top_count, dim=-1)
        return [top_labels[0][i].numpy() for i in range(top_count)]

    def rank(self, image_features: torch.Tensor, top_count: int = 1, reverse: bool = False) -> List[str]:
        if len(self.labels) <= self.chunk_size:
            tops = self._rank(image_features, self.embeddings, top_count=top_count, reverse=reverse)
            return [self.labels[i] for i in tops]

        num_chunks = int(math.ceil(len(self.labels)/self.chunk_size))
        keep_per_chunk = int(self.chunk_size / num_chunks)

        top_labels, top_embeds = [], []
        for chunk_idx in tqdm(range(num_chunks)):
            start = chunk_idx*self.chunk_size
            stop = min(start+self.chunk_size, len(self.embeddings))
            tops = self._rank(image_features, self.embeddings[start:stop], top_count=keep_per_chunk, reverse=reverse)
            top_labels.extend([self.labels[start + i] for i in tops])
            top_embeds.extend([self.embeddings[start + i] for i in tops])

        tops = self._rank(image_features, top_embeds, top_count=top_count)
        return [top_labels[i] for i in tops]
