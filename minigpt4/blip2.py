"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE_Lavis file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import logging
import torch
import torch.nn as nn

from .dist_utils import download_cached_file
from .Qformer import BertConfig, BertLMHeadModel
from .eva_vit import create_eva_vit_g
from transformers import BertTokenizer


class Blip2Base(nn.Module):
    @classmethod
    def init_tokenizer(cls):
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        tokenizer.add_special_tokens({"bos_token": "[DEC]"})
        return tokenizer

    @property
    def device(self):
        return list(self.parameters())[0].device

    @classmethod
    def init_Qformer(cls, num_query_token, vision_width, cross_attention_freq=2):
        encoder_config = BertConfig.from_pretrained("bert-base-uncased")
        encoder_config.encoder_width = vision_width
        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = cross_attention_freq
        encoder_config.query_length = num_query_token
        encoder_config.is_decoder = True
        Qformer = BertLMHeadModel(config=encoder_config)
        query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size)
        )
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
        return Qformer, query_tokens

    @classmethod
    def init_vision_encoder(
        cls, model_name, img_size, drop_path_rate, use_grad_checkpoint, precision
    ):
        assert model_name == "eva_clip_g", "vit model must be eva_clip_g for current version of MiniGPT-4"
        visual_encoder = create_eva_vit_g(
            img_size, drop_path_rate, use_grad_checkpoint, precision
        )

        ln_vision = LayerNorm(visual_encoder.num_features)
        return visual_encoder, ln_vision

    def load_from_pretrained(self, url_or_filename):
        cached_file = download_cached_file(
            url_or_filename, check_hash=False, progress=True
        )
        checkpoint = torch.load(cached_file, map_location="cpu")

        state_dict = checkpoint["model"]

        msg = self.load_state_dict(state_dict, strict=False)

        # logging.info("Missing keys {}".format(msg.missing_keys))
        logging.info("load checkpoint from %s" % url_or_filename)

        return msg


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)

