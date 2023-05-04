import torch
import torch.nn as nn

from .blip2 import Blip2Base


class MiniGPT4(Blip2Base):
    """
    BLIP2 GPT-LLAMA model.
    """

    def __init__(
        self,
        llama_hidden_size=5120,
        vision_dtype=torch.float32,
        vision_device=torch.device("cpu"),
        projector_dtype=torch.float32,
        projector_device=torch.device("cpu"),
        vit_model="eva_clip_g",
        q_former_model="https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth",
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp32",
        num_query_token=32,
        max_txt_len=32,
        end_sym='\n'
    ):
        super().__init__()

        self.vision_dtype = vision_dtype
        self.vision_device = vision_device
        self.projector_dtype = projector_dtype
        self.projector_device = projector_device

        self.tokenizer = self.init_tokenizer()

        print('Loading VIT')
        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision
        )
        self.visual_encoder = self.visual_encoder.eval().to(self.vision_device, dtype=self.vision_dtype)
        self.ln_vision = self.ln_vision.eval().to(self.vision_device, dtype=self.vision_dtype)
        print('Loading VIT Done')

        print('Loading Q-Former')
        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, self.visual_encoder.num_features
        )
        self.Qformer.cls = None
        self.Qformer.bert.embeddings.word_embeddings = None
        self.Qformer.bert.embeddings.position_embeddings = None
        for layer in self.Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None
        self.load_from_pretrained(url_or_filename=q_former_model)

        self.Qformer = self.Qformer.eval().to(self.projector_device, dtype=self.projector_dtype)
        print('Loading Q-Former Done')

        self.llama_proj = nn.Linear(
            self.Qformer.config.hidden_size, llama_hidden_size
        ).to(self.projector_device, dtype=self.projector_dtype)
        self.max_txt_len = max_txt_len
        self.end_sym = end_sym


    def encode_img(self, image):
        image = image.to(self.vision_device, dtype=self.vision_dtype)

        with torch.no_grad():
            image_embeds = self.ln_vision(self.visual_encoder(image)).to(self.projector_device, dtype=self.projector_dtype)
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(self.projector_device)

            query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1).to(self.projector_device)
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

            inputs_llama = self.llama_proj(query_output.last_hidden_state)
        return inputs_llama
