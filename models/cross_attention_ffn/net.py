# coding=utf-8
# Copyright 2022 The SimREC Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
import math
from models.language_encoder import language_encoder
from models.visual_encoder import visual_encoder
from layers.fusion_layer import MultiScaleFusion,SimpleFusion,GaranAttention
from models.segment_anything.modeling.image_encoder_cross_attention_FFN import ImageEncoderViT
from models.segment_anything.modeling.prompt_encoder import PromptEncoder
from models.segment_anything.modeling.mask_decoder_v2 import MaskDecoder
from models.segment_anything.modeling.transformer import TwoWayTransformer
from functools import partial
import torch.nn.functional as F
from typing import Any, Dict, List, Tuple
from operator import mul
from functools import reduce

torch.backends.cudnn.enabled=False

'''_build_sam(
        encoder_embed_dim=1280,
        encoder_depth=32,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[7, 15, 23, 31],
        checkpoint=checkpoint,
    )'''
class Net(nn.Module):
    def __init__(self, __C, pretrained_emb, token_size):
        super(Net, self).__init__()
        prompt_embed_dim = 256
        image_size = 1024
        vit_patch_size = 16
        image_embedding_size = image_size // vit_patch_size
        self.visual_encoder=ImageEncoderViT(depth=12,
                                              embed_dim=768,
                                              img_size=1024,
                                              mlp_ratio=4,
                                              norm_layer=partial(nn.LayerNorm, eps=1e-6),
                                              num_heads=12,
                                              patch_size=16,
                                              qkv_bias=True,
                                              use_rel_pos=True,
                                              global_attn_indexes=[2, 5, 8, 11],
                                              window_size=14,
                                              out_chans=256)
        self.prompt_encoder=PromptEncoder(
                                        embed_dim=prompt_embed_dim,
                                        image_embedding_size=(image_embedding_size, image_embedding_size),
                                        input_image_size=(image_size, image_size),
                                        mask_in_chans=16,
                                        )
        self.mask_decoder=MaskDecoder(
                            num_multimask_outputs=3,
                            transformer=TwoWayTransformer(
                                depth=2,
                                embedding_dim=prompt_embed_dim,
                                mlp_dim=2048,
                                num_heads=8,
                            ),
                            transformer_dim=prompt_embed_dim,
                            iou_head_depth=3,
                            iou_head_hidden_dim=256,
                        )
        self.query_embeddings=nn.Parameter(
            torch.zeros(1, 4, 768)
        )
        self.query_embeddings.data.normal_(mean=0.0, std=.01)
        
        '''self.visual_encoder=ImageEncoderViT(depth=32,
                                              embed_dim=1280,
                                              img_size=1024,
                                              mlp_ratio=4,
                                              norm_layer=partial(nn.LayerNorm, eps=1e-6),
                                              num_heads=16,
                                              patch_size=16,
                                              qkv_bias=True,
                                              use_rel_pos=True,
                                              global_attn_indexes=[7, 15, 23, 31],
                                              window_size=14,
                                              out_chans=256)''' #VIT-H
        if __C.LANG_ENC=='lstm':
            self.lang_encoder=language_encoder(__C,pretrained_emb,token_size)
        else:
            self.lang_encoder=language_encoder(__C,'bert-base-uncased',True)
        self.text_proj = nn.Linear(self.lang_encoder.num_channels, 256)
        self.self_attention = nn.MultiheadAttention(768, 12)
        self.up_sample=nn.UpsamplingBilinear2d(scale_factor=4)
        total = sum([param.nelement() for param in self.lang_encoder.parameters()])
        state_dict = torch.load('./checkpoints/sam_vit_b_01ec64.pth')
        keys = state_dict.keys()
        new_state_dict = {}
        for key in keys:
            if key.startswith('image_encoder.'):
                new_key = key.replace('image_encoder.', 'visual_encoder.')
                new_state_dict[new_key] = state_dict[key]
            else:
                new_state_dict[key] = state_dict[key]
        self.load_state_dict(new_state_dict, strict=False)
        print('  + Number of lang enc params: %.2fM' % (total / 1e6))  # 每一百万为一个单位
        if __C.VIS_FREEZE:
            if __C.VIS_ENC=='vgg' or __C.VIS_ENC=='darknet':
                self.frozen(self.visual_encoder.module_list[:-2])
            elif __C.VIS_ENC=='cspdarknet':
                self.frozen(self.visual_encoder.model[:-2])
            else:
                self.frozen(self.visual_encoder)
        if __C.LANG_FREEZE:
            self.frozen(self.lang_encoder)
    def frozen(self,module):
        if getattr(module,'module',False):
            for child in module.module():
                for param in child.parameters():
                    param.requires_grad = False
        else:
            for param in module.parameters():
                param.requires_grad = False
    def forward(self,x,y, det_label=None,seg_label=None):
        query_emb = self.query_embeddings.repeat(x.shape[0], 1, 1)
        b, h, w = x.shape[0], x.shape[2], x.shape[3]
        y,cls_token=self.lang_encoder(y)
        cls_token = cls_token.unsqueeze(1)
        text_src, text_mask = y.decompose()
        '''text_mask = text_mask.repeat(1, 1, text_src.shape[-1])   
        text_src = text_src.masked_fill(text_mask, 0)'''
        #text_src = text_src.permute(1, 0, 2)
        text_mask = text_mask.permute(1, 0)
        #text_src = self.self_attention(text_src, text_src, text_src, key_padding_mask=text_mask)[0].permute(1, 0, 2) #B,L,C
        x=self.visual_encoder(x,text_src,query_emb)
        #x = self.visual_encoder(x, text_src)
        text_src = self.text_proj(text_src)
        sparse_embeddings, dense_embeddings = self.prompt_encoder(points=None,
                                                                  boxes=None,
                                                                  masks=None,
                                                                  text_embeds=text_src)
        sparse_embeddings = sparse_embeddings.to(text_src.dtype)
        low_res_masks, iou_predictions = self.mask_decoder(image_embeddings=x,
                                                           image_pe=self.prompt_encoder.get_dense_pe(h),
                                                           sparse_prompt_embeddings=sparse_embeddings,
                                                           dense_prompt_embeddings=dense_embeddings,
                                                           multimask_output=False
        )
        mask = self.up_sample(low_res_masks)
        if not self.training:
            mask=(mask.squeeze(1).sigmoid()>0.35).float()
            return mask
        else:
            loss = F.binary_cross_entropy_with_logits(mask, seg_label)
            return loss
        

def print_weights(model):
    for name, param in model.named_parameters():
        if 'weight' in name:
            print(f"{name} : {param.detach().cpu().numpy()}")
def postprocess_masks(
        self,
        masks: torch.Tensor,
        input_size: Tuple[int, ...],
        original_size: Tuple[int, ...],
    ) -> torch.Tensor:
        """
        Remove padding and upscale masks to the original image size.

        Arguments:
          masks (torch.Tensor): Batched masks from the mask_decoder,
            in BxCxHxW format.
          input_size (tuple(int, int)): The size of the image input to the
            model, in (H, W) format. Used to remove padding.
          original_size (tuple(int, int)): The original size of the image
            before resizing for input to the model, in (H, W) format.

        Returns:
          (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
            is given by original_size.
        """

        dtype = masks.dtype

        masks = F.interpolate(
            masks.float(),
            (self.image_encoder.img_size, self.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )
        # masks = masks.to(dtype)
        masks = masks[..., : input_size[0], : input_size[1]]
        masks = F.interpolate(
            masks, original_size, mode="bilinear", align_corners=False
        )
        return masks

if __name__ == '__main__':
    class Cfg():
        def __init__(self):
            super(Cfg, self).__init__()
            self.USE_GLOVE = False
            self.WORD_EMBED_SIZE = 300
            self.HIDDEN_SIZE = 512
            self.N_SA = 0
            self.FLAT_GLIMPSES = 8
            self.DROPOUT_R = 0.1
            self.LANG_ENC = 'lstm'
            self.VIS_ENC = 'darknet'
            self.VIS_PRETRAIN = True
            self.PRETTRAIN_WEIGHT = './darknet.weights'
            self.ANCHORS = [[116, 90], [156, 198], [373, 326]]
            self.ANCH_MASK = [[0, 1, 2]]
            self.N_CLASSES = 0
            self.VIS_FREEZE = True
    cfg=Cfg()
    model=Net(cfg,torch.zeros(1),100)
    # model.train()
    img=torch.zeros(2,3,224,224)
    lang=torch.randint(10,(2,14))
    seg, det=model(img,lang)
    print(seg.size(),det.size())
