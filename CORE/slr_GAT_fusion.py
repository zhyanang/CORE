# coding=utf-8
import pdb
import copy
import utils
import torch
import types
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from modules.criterions import SeqKD
from modules import BiLSTMLayer, TemporalConv,GAT,GAT_Euli,GAT_cos,GAT_multiEuli,SpGAT
import math
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import pickle

# path = '/hdd1/zy/code/GCN_cue1/weight/'

def update_lgt(self, lgt):
    feat_len = copy.deepcopy(lgt)
    for ks in self.kernel_size:
        if ks[0] == 'P':
            feat_len //= 2
        else:
            feat_len -= int(ks[1]) - 1
    return feat_len

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class GCN(nn.Module):
    # def __init__(self,nfeat, nhid, hidden_size, dropout,num_classes,gloss_dict=None):
    def __init__(self,num_classes,conv_type,use_bn,nfeat, nhid, hidden_size1, dropout,hidden_size=1024,gloss_dict=None,loss_weights=None):
        super(GCN, self).__init__()
        self.decoder = None
        self.loss = dict()
        self.criterion_init()
        self.num_classes = num_classes
        self.loss_weights = loss_weights
        self.blank_id = 1295
        # self.ctc_loss = nn.CTCLoss(blank=self.blank_id, zero_infinity=True, reduction='none')
        # self.numLevels = 1
        self.resnet18 = models.resnet18(pretrained=True)
        self.resnet = nn.Sequential(*list(self.resnet18.children())[:-2])
        self.resnet2 = self.resnet18
        self.resnet2.fc = Identity()
        self.alpha = nn.Parameter(torch.ones(3), requires_grad=True)
        # self.conv2d = getattr(models, 'resnet18')(pretrained=True)
        # self.conv2d.fc = Identity()

        self.conv1d = TemporalConv(input_size=768,
                                   hidden_size=hidden_size,
                                   conv_type=conv_type,
                                   use_bn=use_bn,
                                   num_classes=num_classes)
        self.gat = GAT_multiEuli(nfeat=nfeat, nhid=nhid, nclass=768, dropout=0.5, alpha=0.01, nheads=1)
        self.gat2 = SpGAT(nfeat=nfeat, nhid=nhid, nclass=768, dropout=0.5, alpha=0.01, nheads=1)
        # self.mapping_res = nn.Linear(512, 512)
        self.mapping1 = nn.Linear(3072, 2048)
        self.mapping2 = nn.Linear(2304, 2048)
        # self.mapping1 = nn.Linear(2304, 2048)
        # self.mapping2 = nn.Linear(1536, 2048)
        self.mapping3 = nn.Linear(768, 2048)
        self.dropout = dropout
        self.decoder = utils.Decode(gloss_dict, num_classes, 'beam')
        self.temporal_model = BiLSTMLayer(rnn_type='LSTM', input_size=2048, hidden_size=hidden_size,
                                              num_layers=2, bidirectional=True)
        self.classifier = nn.Linear(2816, self.num_classes)
        self.register_backward_hook(self.backward_hook)

    def backward_hook(self, module, grad_input, grad_output):
        for g in grad_input:
            g[g != g] = 0

    def normalize(self,x, axis=-1):
        x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
        return x

    # def masked_bn(self, inputs, len_x):
    # #     def pad(tensor, length):
    # #         return torch.cat([tensor, tensor.new(length - tensor.size(0), *tensor.size()[1:]).zero_()])
    # #
    # #     x = torch.cat([inputs[len_x[0] * idx:len_x[0] * idx + lgt] for idx, lgt in enumerate(len_x)])
    #     x = self.resnet(inputs)
    #     # x = torch.cat([pad(x[sum(len_x[:idx]):sum(len_x[:idx + 1])], len_x[0])
    #     #                for idx, lgt in enumerate(len_x)])
    #     return x

# # 1.GCN  2.1d卷积   3.LSTM  4.分类
    def forward(self, x, len_x, label=None, label_lgt=None,Face_keypoint=None,Rhand_keypoint=None,Lhand_keypoint=None):
        # face_Rhand_value =0
        # face_Lhand_value = 0
        # Rhand_Lhand_value = 0
        # face_hand_value = 0
        full_fram_feature = []
        batch, temp, channel, height, width = x.shape
        inputs = x.reshape(batch * temp, channel, height, width)
        pose_fea = self.resnet2(inputs)
        pose_fea1 = pose_fea.reshape(batch,temp,-1)
        # full_fram = inputs.transpose(0,1)
        adj_full_fram = torch.zeros(temp,temp)
        for i in range(temp):
            adj_full_fram[i, max(0, i - 2):min(temp, i + 3)] = 1
            adj_full_fram[i, i] = 0

        for ii in range(batch):
            full_fram_fea = self.gat2(pose_fea1[ii],adj_full_fram.cuda())
            # full_fram_fea_final = torch.mean(full_fram_fea,dim = 0)
            full_fram_feature.append(full_fram_fea)

        Face0 = Face_keypoint.reshape(batch * temp, -1)
        Lhand0 = Lhand_keypoint.reshape(batch * temp, -1)
        Rhand0 = Rhand_keypoint.reshape(batch * temp, -1)
        pose = self.resnet(inputs)
        # pose_fea = self.resnet2(inputs)
        final = []
        zzz = 0

        for i in range(0,batch*temp):
            Face = pose[i, :, int(Face0[i][0]*7/256)-1, int(Face0[i][1]*7/256)-1]
            # Face = self.mapping_res(Face)
            if Lhand0[i][0]==0 and Lhand0[i][1]==0:
                Lhand = torch.zeros(512).cuda()
            else:

                Lhand = pose[i, :, int(Lhand0[i][0]*7/256)-1, int(Lhand0[i][1]*7/256)-1]
                # Lhand = self.mapping_res(Lhand)
            if Rhand0[i][0]==0 and Rhand0[i][1]==0:
                Rhand = torch.zeros(512).cuda()
            else:
                Rhand = pose[i, :, int(Rhand0[i][0]*7/256)-1, int(Rhand0[i][1]*7/256)-1]
                # Rhand = self.mapping_res(Rhand)

            node_fea = torch.cat((self.normalize(pose_fea[i]).unsqueeze(0),self.normalize(Face).unsqueeze(0),self.normalize(Rhand).unsqueeze(0),self.normalize(Lhand).unsqueeze(0)),dim=0)
            nonzero_idx = torch.nonzero(torch.sum(node_fea, dim=1))
            node_fea = node_fea[nonzero_idx.squeeze()]
            # adj = torch.Tensor([[0, 1, 1], [1, 0, 1], [1, 1, 0]]).cuda()
            # adj = adj[:len(node_fea)-1, :len(node_fea)-1]
            x1 = self.gat(node_fea,self.alpha[0])

            # file = open(path + 'weight'+'.pkl', 'wb')
            # pickle.dump([torch.abs(x1['face_Rhand']).item(), torch.abs(x1['face_Lhand']).item(), torch.abs(x1['Rhand_Lhand']).item()], file)
            # file.close()

            if len(node_fea)==4:
                # face_Rhand_value += torch.abs(x1['face_Rhand'])
                # face_Lhand_value += torch.abs(x1['face_Lhand'])
                # Rhand_Lhand_value += torch.abs(x1['Rhand_Lhand'])

                xxx1 = torch.cat((x1['output'][0],x1['output'][1],x1['output'][2],x1['output'][3]),dim=0)
                # xxx1 = self.mapping1(xxx1)
                # xxx1 = torch.mean(torch.stack(( x1['output'][1], x1['output'][2], x1['output'][3])),dim=0)
                xxx1 = self.mapping1(xxx1)

                final.append(xxx1)
                # zzz+=1
            else:
                xxx1 = torch.cat((x1['output'][0],x1['output'][1],x1['output'][2]),dim=0)
                xxx1 = self.mapping2(xxx1)
                # xxx1 = torch.mean(torch.stack((x1['output'][1], x1['output'][2])), dim=0)
                # xxx1 = self.mapping2(xxx1)
                # zzz += 1

                final.append(xxx1)

        # face_Rhand_value = face_Rhand_value/zzz
        # face_Lhand_value = face_Lhand_value/zzz
        # Rhand_Lhand_value = Rhand_Lhand_value/zzz

        final = torch.stack(final)

        full_fram_feature =torch.stack(full_fram_feature).reshape(batch*temp,-1)

        # full_fram_feature = self.mapping3(full_fram_feature)
        final = torch.cat((final,full_fram_feature),dim=1)

        # final = (final*0.3+full_fram_feature*0.7)

        final = final.reshape(batch,temp,-1).permute(1,0,2)#[temp,batch,-1]
        lgt = len_x.cpu()
        # tm_outputs = self.temporal_model(final, lgt)#[temp,batch,-1]
        outputs = self.classifier(final)
        pred = None if self.training \
            else self.decoder.decode(outputs, lgt, batch_first=False, probs=False)

        return {
            # "face_hand":face_hand_value,
            # "face_Rhand":face_Rhand_value,
            # "face_Lhand":face_Lhand_value,
            # "Rhand_Lhand":Rhand_Lhand_value,
            "visual_features": x,
            "feat_len": lgt,
            "sequence_logits": outputs,
            "recognized_sents": pred,}

    def criterion_calculation(self, ret_dict, label, label_lgt):
        loss = 0
        for k, weight in self.loss_weights.items():
            if k == 'SeqCTC':
                loss += self.loss['CTCLoss'](ret_dict["sequence_logits"].log_softmax(-1),
                                                      label.cpu().int(), ret_dict["feat_len"].cpu().int(),
                                                      label_lgt.cpu().int()).mean()
        return loss

    def criterion_init(self):
        self.loss['CTCLoss'] = torch.nn.CTCLoss(reduction='none', zero_infinity=False)
        self.loss['distillation'] = SeqKD(T=8)
        return self.loss