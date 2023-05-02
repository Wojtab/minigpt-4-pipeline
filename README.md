# MiniGPT-4-Pipeline

Unfinished, example pipeline for generic multimodality support in text-generation-webui.

It looks to be working, with:
- `9cc054f` commit of text-generation-webui (multimodality draft PR, from `Wojtab/text-generation-webui`, `generalize-multimodality` branch)
- anon Vicuna 13B 4bit 128g as the LLM
- GPTQ - ooba's old CUDA
- Ubuntu 20.04, WSL2, Cuda 11.3 (for some reason I don't have 11.7 in this conda)
- RTX 3090

In short: I copied enough code from MiniGPT-4, so that inference works(but only for image embeds), then I added a pipeline descriptor.

It probably can be cleaned up, etc. but the point of this repo for now is to give an example new pipeline for the generic multimodality proposal, and not to be the actual, finished implementation.

To use, clone it into `extensions/multimodal/pipelines`, and install `requirements.txt`. Select Vicuna-v0-13B as the model, and add `--extensions multimodal --multimodal-pipeline minigpt4-13b` to CLI. If it crashes - oh well


DISCLAIMER: this is a hack on hack on hack on copy-pasted code, if you are dumb enough to trust it I take no liability whatsoever, and don't provide any warranty, nor support
