import torch
import torch.nn as nn

import os
import numpy as np
import os.path as osp
import matplotlib.pyplot as plt

# from .base_model import BaseModel
from .modules import SetBlockWrapper, HorizontalPoolingPyramid, PackSequenceWrapper, SeparateFCs, SeparateBNNecks, conv1x1, conv3x3, BasicBlock2D, BasicBlockP3D, BasicBlock3D

from einops import rearrange
class DirectionEstimation(nn.Module):

    def build_network(self, model_cfg):
        channels    = model_cfg['Backbone']['channels']
        
        self.FCs = SeparateFCs(16, channels[3], channels[2])
        self.BNNecks = SeparateBNNecks(16, channels[2], class_num=model_cfg['SeparateBNNecks']['class_num'])

        self.TP = PackSequenceWrapper(torch.max)
        self.HPP = HorizontalPoolingPyramid(bin_num=[16])

    def init_parameters(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.Conv2d, nn.Conv1d)):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif isinstance(m, (nn.BatchNorm3d, nn.BatchNorm2d, nn.BatchNorm1d)):
                if m.affine:
                    nn.init.normal_(m.weight.data, 1.0, 0.02)
                    nn.init.constant_(m.bias.data, 0.0)

    def forward(self, inputs):
        # ipts, labs, typs, vies, seqL = inputs
        
        # if len(ipts[0].size()) == 4:
        #     sils = ipts[0].unsqueeze(1)
        # else:
        #     sils = ipts[0]
        #     sils = sils.transpose(1, 2).contiguous()
        # assert sils.size(-1) in [44, 88]

        # del ipts
        # out0 = self.layer0(sils)
        # out1 = self.layer1(out0)
        # out2 = self.layer2(out1)
        # out3 = self.layer3(out2)
        # out4 = self.layer4(out3) # [n, c, s, h, w]
        
        out4, labs, typs, vies, seqL = inputs
        # Temporal Pooling, TP
        outs = self.TP(out4, seqL, options={"dim": 2})[0]  # [n, c, h, w]

        # Horizontal Pooling Matching, HPM
        feat = self.HPP(outs)  # [n, c, p]

        embed_1 = self.FCs(feat)  # [n, c, p]
        embed_2, logits = self.BNNecks(embed_1)  # [n, c, p]

        if self.inference_use_emb2:
                embed = embed_2
        else:
                embed = embed_1

        retval = {
            'training_feat': {
                'triplet': {'embeddings': embed_1, 'labels': vies},
                'softmax': {'logits': logits, 'labels': vies}
            },
            # 'visual_summary': {
            #     'image/sils': rearrange(sils, 'n c s h w -> (n s) c h w'),
            # },
            'inference_feat': {
                'embeddings': embed
            }
        }

        return retval