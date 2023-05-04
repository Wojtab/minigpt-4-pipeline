from typing import Optional
from extensions.multimodal.abstract_pipeline import AbstractMultimodalPipeline


available_pipelines = ['minigpt4-13b', 'minigpt4-7b']


def get_pipeline(name: str, params: dict) -> Optional[AbstractMultimodalPipeline]:
    if name == 'minigpt4-13b':
        from .minigpt4_pipeline import MiniGPT4_13b_Pipeline
        return MiniGPT4_13b_Pipeline(params)
    if name == 'minigpt4-7b':
        from .minigpt4_pipeline import MiniGPT4_7b_Pipeline
        return MiniGPT4_7b_Pipeline(params)
    return None
