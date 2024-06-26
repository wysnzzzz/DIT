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
import  torch.nn as nn

from layers.sa_layer import SA,AttFlat
from utils.utils import make_mask
from utils.misc import NestedTensor, is_main_process
# from .position_encoding import build_position_encoding

from pytorch_pretrained_bert.modeling import BertModel

class BERT(nn.Module):
    def __init__(self, __C,train_bert=False, enc_num=12, **kwargs):
        super().__init__()
        self.num_channels = 768
        self.enc_num = enc_num

        self.bert = BertModel.from_pretrained('bert-base-uncased')
        if not train_bert:
            for parameter in self.bert.parameters():
                parameter.requires_grad_(False)

    def forward(self, tensor_list: NestedTensor):

        if self.enc_num > 0:
            all_encoder_layers, cls_token = self.bert(tensor_list.tensors, token_type_ids=None, attention_mask=tensor_list.mask)
            # use the output of the X-th transformer encoder layers
            xs = all_encoder_layers[self.enc_num - 1]
            # use the output of all transformer encoder layers
            #xs = all_encoder_layers
        else:
            xs = self.bert.embeddings.word_embeddings(tensor_list.tensors)
        mask = tensor_list.mask.to(torch.bool)
        mask = ~mask
        out = NestedTensor(xs, mask)

        return out

class LSTM_SA(nn.Module):
    def __init__(self, __C, pretrained_emb, token_size):
        super(LSTM_SA, self).__init__()
        self.embedding = nn.Embedding(
            num_embeddings=token_size,
            embedding_dim=__C.WORD_EMBED_SIZE
        )

        # Loading the GloVe embedding weights
        if __C.USE_GLOVE:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_emb))

        self.lstm = nn.GRU(
            input_size=__C.WORD_EMBED_SIZE,
            hidden_size=__C.HIDDEN_SIZE,
            num_layers=1,
            batch_first=True,
            dropout=__C.DROPOUT_R,
            bidirectional=False
        )

        self.sa_list = nn.ModuleList([SA(__C) for _ in range(__C.N_SA)])
        self.att_flat=AttFlat(__C)
        if __C.EMBED_FREEZE:
            self.frozen(self.embedding)
    def frozen(self, module):
        if getattr(module, 'module', False):
            for child in module.module():
                for param in child.parameters():
                    param.requires_grad = False
        else:
            for param in module.parameters():
                param.requires_grad = False
    def forward(self, ques_ix):

        # Pre-process Language Feature
        lang_feat_mask = make_mask(ques_ix.unsqueeze(2))
        lang_feat = self.embedding(ques_ix)
        lang_feat, _ = self.lstm(lang_feat)


        for sa in self.sa_list:
            lang_feat = sa(lang_feat, lang_feat_mask)

        flat_lang_feat = self.att_flat(lang_feat, lang_feat_mask)
        return  {
            'flat_lang_feat':flat_lang_feat,
            'lang_feat':lang_feat,
            'lang_feat_mask':lang_feat_mask
        }


backbone_dict={
    'lstm':LSTM_SA,
    'bert':BERT
}

def language_encoder(__C, pretrained_emb, token_size):
    lang_enc=backbone_dict[__C.LANG_ENC](__C, pretrained_emb, token_size)
    return lang_enc