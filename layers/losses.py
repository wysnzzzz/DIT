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
import torch.nn.functional as F
from torch.autograd import Variable


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)  # prevents nans when probability 0
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        
    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        
        return 1 - dice

class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, gamma=2.0, smooth=1.):
        super(CombinedLoss, self).__init__()
        self.focal_loss = FocalLoss(gamma=gamma)
        self.dice_loss = DiceLoss(smooth=smooth)
        self.alpha = alpha
        self.beta = beta
        
    def forward(self, inputs, targets):
        loss_focal = self.focal_loss(inputs, targets)
        loss_dice = self.dice_loss(inputs, targets)
        return self.alpha * loss_focal + self.beta * loss_dice

class IOUWH_loss(nn.Module): #used for anchor guiding
    def __init__(self, reduction='none'):
        super(IOUWH_loss, self).__init__()
        self.reduction = reduction

    def forward(self, pred, target):
        orig_shape = pred.shape
        pred = pred.view(-1,4)
        target = target.view(-1,4)
        target[:,:2] = 0
        tl = torch.max((target[:, :2]-pred[:,2:]/2),
                      (target[:, :2] - target[:, 2:]/2))

        br = torch.min((target[:, :2]+pred[:,2:]/2),
                      (target[:, :2] + target[:, 2:]/2))

        area_p = torch.prod(pred[:,2:], 1)
        area_g = torch.prod(target[:,2:], 1)

        en = (tl< br).type(tl.type()).prod(dim=1)
        area_i = torch.prod(br-tl, 1) * en
        U = area_p+area_g-area_i+ 1e-16
        iou= area_i / U

        loss = 1-iou**2
        if self.reduction =='mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()

        return loss

class IOUloss(nn.Module):
    def __init__(self, reduction='none'):
        super(IOUloss, self).__init__()
        self.reduction = reduction

    def forward(self, pred, target):
        orig_shape = pred.shape
        pred = pred.view(-1,4)
        target = target.view(-1,4)
        tl = torch.max((pred[:, :2]-pred[:,2:]/2),
                      (target[:, :2] - target[:, 2:]/2))
        br = torch.min((pred[:, :2]+pred[:,2:]/2),
                      (target[:, :2] + target[:, 2:]/2))

        area_p = torch.prod(pred[:,2:], 1)
        area_g = torch.prod(target[:,2:], 1)

        en = (tl< br).type(tl.type()).prod(dim=1)
        area_i = torch.prod(br-tl, 1) * en
        iou= (area_i) / (area_p+area_g-area_i+ 1e-16)

        loss = 1-iou**2
        if self.reduction =='mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()

        return loss