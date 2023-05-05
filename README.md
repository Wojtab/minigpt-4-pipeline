# MiniGPT-4-Pipeline

A pipeline adding [MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4) support to [text-generation-webui](https://github.com/oobabooga/text-generation-webui).

### Installation
Clone this repo into `extensions/multimodal/pipelines` directory in text-generation-webui, and install `requirements.txt`.

### Pipelines

This module provides 2 pipelines:
- `minigpt4-13b` - for use with Vicuna-v0-13B LLM
- `minigpt4-7b` - for use with Vicuna-v0-7B LLM

To use it in webui, select the appropriate LLM and run `server.py` with `--extensions multimodal --multimodal-pipeline minigpt4-13b` (or `minigpt4-7b`)

The supported parameter combinations for both the vision model, and the projector are: CUDA/32bit, CUDA/16bit, CPU/32bit

### Credits
Almost all the code in minigpt4 directory is taken from [the original MiniGPT4 repo](https://github.com/Vision-CAIR/MiniGPT-4), it was then cleaned up to leave mostly the parts, which are needed for inference. The only modifications are to `minigpt4/mini_gpt4.py`, but again, they are mostly removing not-needed parts of the code.

In short: I copied enough code from MiniGPT-4, so that inference works(but only for image embeds), then I added a pipeline descriptor.


### DISCLAIMER
This is not production-ready code, I take no liability whatsoever, and don't provide any warranty, nor support. Use it only for fun/research
