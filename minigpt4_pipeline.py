from typing import List
import torch
from PIL import Image

from extensions.multimodal.abstract_pipeline import AbstractMultimodalPipeline
from modules import shared
from modules.text_generation import encode
from huggingface_hub import hf_hub_download
from .minigpt4.processor import Blip2ImageEvalProcessor
from .minigpt4.mini_gpt4 import MiniGPT4

class MiniGPT4_13b_Pipeline(AbstractMultimodalPipeline):
    def __init__(self, params: dict) -> None:
        super().__init__()
        self.image_processor = Blip2ImageEvalProcessor()
        ckpt_path = hf_hub_download("Vision-CAIR/MiniGPT-4", "pretrained_minigpt4.pth")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        # eva-clip-g should be loaded on params["vision_device"]
        # linear layer should be on params["projector_device"]
        # QFormer on idk, one of them
        self.vision_tower = MiniGPT4()
        self.vision_tower.load_state_dict(ckpt['model'], strict=False)

    @staticmethod
    def name() -> str:
        return "minigpt4-13b"

    @staticmethod
    def placeholder_token_id() -> int:
        return 1

    @staticmethod
    def image_start() -> str:
        return "<Img>"

    @staticmethod
    def image_end() -> str:
        return "</Img>"

    @staticmethod
    def num_image_embeds() -> int:
        return 32

    @staticmethod
    def embed_tokens(input_ids: torch.Tensor) -> torch.Tensor:
        return shared.model.model.embed_tokens(input_ids).to(shared.model.device, dtype=shared.model.dtype)

    @staticmethod
    def placeholder_embeddings() -> torch.Tensor:
        placeholders = torch.ones(MiniGPT4_13b_Pipeline.num_image_embeds()) * MiniGPT4_13b_Pipeline.placeholder_token_id()
        return MiniGPT4_13b_Pipeline.embed_tokens(placeholders.to(shared.model.device, dtype=shared.model.dtype))

    def embed_images(self, images: List[Image.Image]) -> torch.Tensor:
        im = torch.stack([self.image_processor(image) for image in images])
        image_emb, _ = self.vision_tower.encode_img(im)
        return image_emb.to(shared.model.device, dtype=shared.model.dtype)
