from typing import List
import torch
from PIL import Image

from extensions.multimodal.abstract_pipeline import AbstractMultimodalPipeline
from modules import shared
from modules.text_generation import encode
from huggingface_hub import hf_hub_download
from .minigpt4.processor import Blip2ImageEvalProcessor
from .minigpt4.mini_gpt4 import MiniGPT4

class MiniGPT4_Pipeline(AbstractMultimodalPipeline):
    def __init__(self, params: dict) -> None:
        super().__init__()
        self.image_processor = Blip2ImageEvalProcessor()

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
        placeholders = encode("<ImgContent>", add_bos_token=False, add_special_tokens=False)[0]#torch.ones(MiniGPT4_Pipeline.num_image_embeds()) * MiniGPT4_Pipeline.placeholder_token_id()
        return MiniGPT4_Pipeline.embed_tokens(placeholders.to(shared.model.device, dtype=torch.int64)).to(dtype=shared.model.dtype)

    def embed_images(self, images: List[Image.Image]) -> torch.Tensor:
        im = torch.stack([self.image_processor(image) for image in images])
        image_emb = self.vision_tower.encode_img(im)
        return image_emb.to(shared.model.device, dtype=shared.model.dtype)


class MiniGPT4_13b_Pipeline(MiniGPT4_Pipeline):
    def __init__(self, params: dict) -> None:
        super().__init__(params)
        ckpt_path = hf_hub_download("Vision-CAIR/MiniGPT-4", "pretrained_minigpt4.pth")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        self.vision_tower = MiniGPT4(llama_hidden_size=5120,
                                     vision_dtype=self._get_dtype("vision_bits", params),
                                     vision_device=self._get_device("vision_device", params),
                                     projector_device=self._get_device("projector_device", params),
                                     projector_dtype=self._get_dtype("projector_bits", params))
        self.vision_tower.load_state_dict(ckpt['model'], strict=False)

    @staticmethod
    def name() -> str:
        return "minigpt4-13b"


class MiniGPT4_7b_Pipeline(MiniGPT4_Pipeline):
    def __init__(self, params: dict) -> None:
        super().__init__(params)
        ckpt_path = hf_hub_download("ckpt/minigpt4-7B", "prerained_minigpt4_7b.pth")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        self.vision_tower = MiniGPT4(llama_hidden_size=4096,
                                     vision_dtype=self._get_dtype("vision_bits", params),
                                     vision_device=self._get_device("vision_device", params),
                                     projector_device=self._get_device("projector_device", params),
                                     projector_dtype=self._get_dtype("projector_bits", params))
        self.vision_tower.load_state_dict(ckpt['model'], strict=False)

    @staticmethod
    def name() -> str:
        return "minigpt4-7b"
