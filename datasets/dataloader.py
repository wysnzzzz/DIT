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

import os
import cv2
import json, re, en_vectors_web_lg, random

import albumentations as A
import numpy as np

import torch
import torch.utils.data as Data
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import datasets.transforms as T
from datasets.randaug import RandAugment
from utils.utils import label2yolobox
from utils.box_ops import box_xywh_to_xyxy,box_cxcywh_to_xyxy
import transformers
from transformers import BertTokenizer
from PIL import Image

def read_examples(input_line, unique_id):
    """Read a list of `InputExample`s from an input file."""
    examples = []
    # unique_id = 0
    line = input_line #reader.readline()
    # if not line:
    #     break
    line = line.strip()
    text_a = None
    text_b = None
    m = re.match(r"^(.*) \|\|\| (.*)$", line)
    if m is None:
        text_a = line
    else:
        text_a = m.group(1)
        text_b = m.group(2)
    examples.append(
        InputExample(unique_id=unique_id, text_a=text_a, text_b=text_b))
    # unique_id += 1
    return examples

## Bert text encoding
class InputExample(object):
    def __init__(self, unique_id, text_a, text_b):
        self.unique_id = unique_id
        self.text_a = text_a
        self.text_b = text_b

class InputFeatures(object):
    """A single set of features of data."""
    def __init__(self, unique_id, tokens, input_ids, input_mask, input_type_ids):
        self.unique_id = unique_id
        self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.input_type_ids = input_type_ids

def convert_examples_to_features(examples, seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""
    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)

        if tokens_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > seq_length - 2:
                tokens_a = tokens_a[0:(seq_length - 2)]
        tokens = []
        input_type_ids = []
        tokens.append("[CLS]")
        input_type_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            input_type_ids.append(0)
        tokens.append("[SEP]")
        input_type_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                input_type_ids.append(1)
            tokens.append("[SEP]")
            input_type_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < seq_length:
            input_ids.append(0)
            input_mask.append(0)
            input_type_ids.append(0)

        assert len(input_ids) == seq_length
        assert len(input_mask) == seq_length
        assert len(input_type_ids) == seq_length
        features.append(
            InputFeatures(
                unique_id=example.unique_id,
                tokens=tokens,
                input_ids=input_ids,
                input_mask=input_mask,
                input_type_ids=input_type_ids))
    return features

def make_transforms(__C, image_set ):
    # if is_onestage:
    #     normalize = Compose([
    #         ToTensor(),
    #         Normalize(__C.MEAN, __C.STD)
    #     ])
    #     return normalize

    imsize = __C.INPUT_SHAPE[0]

    if image_set == 'train':
        scales = []
        if __C.AUG_SCALE:
            # scales=[256, 272, 288, 304, 320, 336, 352, 368, 384, 400, 416, 432, 448, 464, 480, 496, 512, 528, 544, 560, 576, 592, 608]
            for i in range(7):
                scales.append(imsize - 32 * i)
        else:
            scales = [imsize]

        if __C.AUG_CROP:
            crop_prob = 0.5
        else:
            crop_prob = 0.

        return T.Compose([
            T.RandomSelect(
                T.RandomResize(scales),
                T.Compose([
                    T.RandomResize([400, 500, 600], with_long_side=False),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales),
                ]),
                p=crop_prob
            ),
            T.ColorJitter(0.4, 0.4, 0.4),
            T.GaussianBlur(aug_blur=__C.AUG_BLUR),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.NormalizeAndPad(mean=__C.MEAN,std=__C.STD,size=imsize, aug_translate=__C.AUG_TRANSLATE)
        ])

    if image_set in ['val', 'test', 'testA', 'testB']:
        return T.Compose([
            T.RandomResize([imsize]),
            T.ToTensor(),
            T.NormalizeAndPad(mean=__C.MEAN,std=__C.STD,size=imsize),
        ])

    raise ValueError(f'unknown {image_set}')

class RefCOCODataSet(Data.Dataset):
    def __init__(self, __C,split):
        super(RefCOCODataSet, self).__init__()
        self.__C = __C
        self.split=split
        assert  __C.DATASET in ['refcoco', 'refcoco+', 'refcocog','referit','vg','merge']
        # --------------------------
        # ---- Raw data loading ---
        # --------------------------
        stat_refs_list=json.load(open(__C.ANN_PATH[__C.DATASET], 'r'))
        total_refs_list=[]
        '''if __C.DATASET in ['vg','merge']:
            total_refs_list = json.load(open(__C.ANN_PATH['merge'], 'r'))+json.load(open(__C.ANN_PATH['refcoco+'], 'r'))+json.load(open(__C.ANN_PATH['refcocog'], 'r'))+json.load(open(__C.ANN_PATH['refcoco'], 'r'))'''
        self.lang_enc = __C.LANG_ENC
        self.ques_list = []
        splits=split.split('+')
        self.refs_anno=[]
        for split_ in splits:
            self.refs_anno+= stat_refs_list[split_]
        if self.lang_enc == 'bert':
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)


        refs=[]

        for split in stat_refs_list:
            for ann in stat_refs_list[split]:
                for ref in ann['refs']:
                    refs.append(ref)
        for split in total_refs_list:
            for ann in total_refs_list[split]:
                for ref in ann['refs']:
                    refs.append(ref)



        self.image_path=__C.IMAGE_PATH[__C.DATASET]
        self.mask_path=__C.MASK_PATH[__C.DATASET]
        self.input_shape=__C.INPUT_SHAPE
        self.flip_lr=__C.FLIP_LR if split=='train' else False
        # Define run data size
        self.data_size = len(self.refs_anno)

        print(' ========== Dataset size:', self.data_size)
        # ------------------------
        # ---- Data statistic ----
        # ------------------------
        # Tokenize
        self.token_to_ix,self.ix_to_token, self.pretrained_emb, max_token = self.tokenize(stat_refs_list, __C.USE_GLOVE)
        self.token_size = self.token_to_ix.__len__()
        print(' ========== Question token vocab size:', self.token_size)
        #keys = list(self.token_to_ix.keys())
        #print('keys:', len(keys))
        #self.tokenizer.add_tokens(keys)
        #self.tokenizer.save_pretrained('/data/huangxiaorui/SAM_research/SimREC_Reseach-TMM_version/vocab/bert_vocab_ori')
        self.max_token = __C.MAX_TOKEN
        if self.max_token == -1:
            self.max_token = max_token
        print('Trimmed to:', self.max_token)
        print('Finished!')
        print('')

        self.candidate_transforms ={}
        if  self.split == 'train':
            if 'RandAugment' in self.__C.DATA_AUGMENTATION:
                self.candidate_transforms['RandAugment']=RandAugment(2,9)
            if 'ElasticTransform' in self.__C.DATA_AUGMENTATION:
                self.candidate_transforms['ElasticTransform']=A.ElasticTransform(p=0.5)
            if 'GridDistortion' in self.__C.DATA_AUGMENTATION:
                self.candidate_transforms['GridDistortion']=A.GridDistortion(p=0.5)
            if 'RandomErasing' in self.__C.DATA_AUGMENTATION:
                self.candidate_transforms['RandomErasing']=transforms.RandomErasing(p=0.3, scale=(0.02, 0.2), ratio=(0.05, 8),
                                                                              value="random")
        self.transforms=make_transforms(__C,self.split)
        #self.transforms=transforms.Compose([transforms.ToTensor(), transforms.Normalize(__C.MEAN, __C.STD)])



    def tokenize(self, stat_refs_list, use_glove):
        token_to_ix = {
            'PAD': 0,
            'UNK': 1,
            'CLS': 2,
        }

        spacy_tool = None
        pretrained_emb = []
        if use_glove:
            spacy_tool = en_vectors_web_lg.load()
            pretrained_emb.append(spacy_tool('PAD').vector)
            pretrained_emb.append(spacy_tool('UNK').vector)
            pretrained_emb.append(spacy_tool('CLS').vector)

        max_token = 0
        for split in stat_refs_list:
            for ann in stat_refs_list[split]:
                for ref in ann['refs']:
                    words = re.sub(
                        r"([.,'!?\"()*#:;])",
                        '',
                        ref.lower()
                    ).replace('-', ' ').replace('/', ' ').split()

                    if len(words) > max_token:
                        max_token = len(words)

                    for word in words:
                        if word not in token_to_ix:
                            token_to_ix[word] = len(token_to_ix)
                            if use_glove:
                                pretrained_emb.append(spacy_tool(word).vector)

        pretrained_emb = np.array(pretrained_emb)
        ix_to_token={}
        for item in token_to_ix:
            ix_to_token[token_to_ix[item]]=item

        return token_to_ix, ix_to_token,pretrained_emb, max_token


    def proc_ref(self, ref, token_to_ix, max_token):
        ques_ix = np.zeros(max_token, np.int64)

        words = re.sub(
            r"([.,'!?\"()*#:;])",
            '',
            ref.lower()
        ).replace('-', ' ').replace('/', ' ').split()

        for ix, word in enumerate(words):
            if word in token_to_ix:
                ques_ix[ix] = token_to_ix[word]
            else:
                ques_ix[ix] = token_to_ix['UNK']

            if ix + 1 == max_token:
                break

        return ques_ix

    # ----------------------------------------------
    # ---- Real-Time Processing Implementations ----
    # ----------------------------------------------

    def load_refs(self, idx):
        refs = self.refs_anno[idx]['refs']
        ref=refs[np.random.choice(len(refs))]
        
        return ref

    def preprocess_info(self,img,mask,box,iid,lr_flip=False):
        h, w, _ = img.shape
        # img = img[:, :, ::-1]
        imgsize=self.input_shape[0]
        new_ar = w / h
        if new_ar < 1:
            nh = imgsize
            nw = nh * new_ar
        else:
            nw = imgsize
            nh = nw / new_ar
        nw, nh = int(nw), int(nh)


        dx = (imgsize - nw) // 2
        dy = (imgsize - nh) // 2

        img = cv2.resize(img, (nw, nh))
        sized = np.ones((imgsize, imgsize, 3), dtype=np.uint8) * 127
        sized[dy:dy + nh, dx:dx + nw, :] = img
        info_img = (h, w, nh, nw, dx, dy,iid)

        mask=np.expand_dims(mask,-1).astype(np.float32)
        mask=cv2.resize(mask, (nw, nh))
        mask=np.expand_dims(mask,-1).astype(np.float32)
        sized_mask = np.zeros((imgsize, imgsize, 1), dtype=np.float32)
        sized_mask[dy:dy + nh, dx:dx + nw, :]=mask
        sized_mask=np.transpose(sized_mask, (2, 0, 1))
        sized_box=label2yolobox(box,info_img,self.input_shape[0],lrflip=lr_flip)
        return sized,sized_mask,sized_box, info_img

    def load_img_feats(self, idx):
        img_path=None
        if self.__C.DATASET in ['refcoco','refcoco+','refcocog']:
            img_path=os.path.join(self.image_path,'COCO_train2014_%012d.jpg'%self.refs_anno[idx]['iid'])
        elif self.__C.DATASET=='referit':
            img_path = os.path.join(self.image_path, '%d.jpg' % self.refs_anno[idx]['iid'])
        elif self.__C.DATASET=='vg':
            img_path = os.path.join(self.image_path, self.refs_anno[idx]['url'])
        elif self.__C.DATASET == 'merge':
            if self.refs_anno[idx]['data_source']=='coco':
                iid='COCO_train2014_%012d.jpg'%int(self.refs_anno[idx]['iid'].split('.')[0])
            else:
                iid=self.refs_anno[idx]['iid']
            img_path = os.path.join(self.image_path,self.refs_anno[idx]['data_source'], iid)
        else:
            assert NotImplementedError

        #image= cv2.imread(img_path)
        image= Image.open(img_path).convert('RGB')
        if self.__C.DATASET in ['refcoco','refcoco+','refcocog','referit']:
            mask=np.load(os.path.join(self.mask_path,'%d.npy'%self.refs_anno[idx]['mask_id']))
        else:
            mask=np.zeros([image.shape[0],image.shape[1],1],dtype=np.float)

        box=np.array([self.refs_anno[idx]['bbox']])
        mask = Image.fromarray(mask * 255)
        return image,mask,box,self.refs_anno[idx]['mask_id'],self.refs_anno[idx]['iid']

    def __getitem__(self, idx):
        ref_iter = self.load_refs(idx)
        image_iter,mask_iter,gt_box_iter,mask_id,iid= self.load_img_feats(idx)
        w, h = image_iter.size
        input_dict = {'img': image_iter,
                      'box': box_xywh_to_xyxy(torch.from_numpy(gt_box_iter[0]).float()),
                      'mask': mask_iter,
                      'text': ref_iter}
        input_dict = self.transforms(input_dict)
        examples = read_examples(input_dict['text'], idx)
        features = convert_examples_to_features(
            examples=examples, seq_length=self.max_token, tokenizer=self.tokenizer)
        ref_iter = features[0].input_ids
        ref_mask = features[0].input_mask
        ref_iter = np.array(ref_iter)
        ref_mask_iter = np.array(ref_mask)
        info_iter = [h, w, *input_dict['info_img'], iid]
        #image_iter, mask_iter, box_iter,info_iter=self.preprocess_info(image_iter,mask_iter,gt_box_iter.copy(),iid,flip_box)
        return \
            torch.from_numpy(ref_iter).long(), \
            input_dict['img'], \
            input_dict['mask'], \
            input_dict['box'], \
            torch.from_numpy(gt_box_iter).float(), \
            mask_id,\
            np.array(info_iter),\
            ref_mask_iter

    def __len__(self):
        return self.data_size

    def shuffle_list(self, list):
        random.shuffle(list)

def loader(__C,dataset: torch.utils.data.Dataset, rank: int, shuffle,drop_last=False):
    if __C.MULTIPROCESSING_DISTRIBUTED:
        assert __C.BATCH_SIZE % len(__C.GPU) == 0
        assert __C.NUM_WORKER % len(__C.GPU) == 0
        assert dist.is_initialized()

        dist_sampler = DistributedSampler(dataset,
                                          num_replicas=__C.WORLD_SIZE,
                                          rank=rank)

        data_loader = DataLoader(dataset,
                                 batch_size=__C.BATCH_SIZE // len(__C.GPU),
                                 shuffle=shuffle,
                                 sampler=dist_sampler,
                                 num_workers=__C.NUM_WORKER //len(__C.GPU),
                                 pin_memory=True,
                                 drop_last=drop_last)  # ,
                                # prefetch_factor=_C['PREFETCH_FACTOR'])  only works in PyTorch 1.7.0
    else:
        data_loader = DataLoader(dataset,
                                 batch_size=__C.BATCH_SIZE,
                                 shuffle=shuffle,
                                 num_workers=__C.NUM_WORKER,
                                 pin_memory=True,
                                 drop_last=drop_last)
    return data_loader

if __name__ == '__main__':

    class Cfg():
        def __init__(self):
            super(Cfg, self).__init__()
            self.ANN_PATH= {
                'refcoco': './data/anns/refcoco.json',
                'refcoco+': './data/anns/refcoco+.json',
                'refcocog': './data/anns/refcocog.json',
                'vg': './data/anns/vg.json',
            }

            self.IMAGE_PATH={
                'refcoco': './data/images/train2014',
                'refcoco+': './data/images/train2014',
                'refcocog': './data/images/train2014',
                'vg': './data/images/VG'
            }

            self.MASK_PATH={
                'refcoco': './data/masks/refcoco',
                'refcoco+': './data/masks/refcoco+',
                'refcocog': './data/masks/refcocog',
                'vg': './data/masks/vg'}
            self.INPUT_SHAPE = (416, 416)
            self.USE_GLOVE = True
            self.DATASET = 'vg'
            self.MAX_TOKEN = 15
            self.MEAN = [0., 0., 0.]
            self.STD = [1., 1., 1.]
    cfg=Cfg()
    dataset=RefCOCODataSet(cfg,'val')
    data_loader = DataLoader(dataset,
                             batch_size=10,
                             shuffle=False,
                             pin_memory=True)
    for _,ref_iter,image_iter, mask_iter, box_iter,words,info_iter in data_loader:
        print(ref_iter)
        # print(image_iter.size())
        # print(mask_iter.size())
        # print(box_iter.size())
        # print(ref_iter.size())
        # # cv2.imwrite('./test.jpg', image_iter.numpy()[0].transpose((1, 2, 0))*255)
        # # cv2.imwrite('./mask.jpg', mask_iter.numpy()[0].transpose((1, 2, 0))*255)
        # print(info_iter.size())
        # print(info_iter)





