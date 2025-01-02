import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import GraphAttentionLayer, SpGraphAttentionLayer,GraphAttentionLayer_Euli,GraphAttentionLayer_cos,GraphAttentionLayer_multiEuli
#
class GAT_Euli(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        super(GAT_Euli, self).__init__()
        self.dropout = dropout
        self.attentions = GraphAttentionLayer_Euli(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True)
        self.out_att = GraphAttentionLayer_Euli(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        if len(x)==4:
            x = F.dropout(x, self.dropout, training=self.training)
            x = self.attentions(x, adj)
            x = F.dropout(x['h_prime'], self.dropout, training=self.training)
            x = self.out_att(x, adj)
            output = F.elu(x['h_prime'])
            return {'output':output, 'face_Rhand':x['face_Rhand'],'face_Lhand': x['face_Lhand'],'Rhand_Lhand': x['Rhand_Lhand']}
        else:
            x = F.dropout(x, self.dropout, training=self.training)
            x = self.attentions(x, adj)
            x = F.dropout(x['h_prime'], self.dropout, training=self.training)
            x = self.out_att(x, adj)
            output = F.elu(x['h_prime'])
            return {'output':output, 'face_hand':x['face_hand']}

class GAT_cos(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        super(GAT_cos, self).__init__()
        self.dropout = dropout
        self.attentions = GraphAttentionLayer_cos(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True)
        self.out_att = GraphAttentionLayer_cos(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        if len(x)==4:
            x = F.dropout(x, self.dropout, training=self.training)
            x = self.attentions(x, adj)
            x = F.dropout(x['h_prime'], self.dropout, training=self.training)
            x = self.out_att(x, adj)
            output = F.elu(x['h_prime'])
            return {'output':output, 'face_Rhand':x['face_Rhand'],'face_Lhand': x['face_Lhand'],'Rhand_Lhand': x['Rhand_Lhand']}
        else:
            x = F.dropout(x, self.dropout, training=self.training)
            x = self.attentions(x, adj)
            x = F.dropout(x['h_prime'], self.dropout, training=self.training)
            x = self.out_att(x, adj)
            output = F.elu(x['h_prime'])
            return {'output':output, 'face_hand':x['face_hand']}


class GAT_multiEuli(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        super(GAT_multiEuli, self).__init__()
        self.dropout = dropout
        self.attentions = GraphAttentionLayer_multiEuli(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True)
        self.out_att = GraphAttentionLayer_multiEuli(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        if len(x)==4:
            x = F.dropout(x, self.dropout, training=self.training)
            x = self.attentions(x, adj)
            x = F.dropout(x['h_prime'], self.dropout, training=self.training)
            x = self.out_att(x, adj)
            output = F.elu(x['h_prime'])
            return {'output':output, 'face_Rhand':x['face_Rhand'],'face_Lhand': x['face_Lhand'],'Rhand_Lhand': x['Rhand_Lhand']}
        elif len(x)== 3:
            x = F.dropout(x, self.dropout, training=self.training)
            x = self.attentions(x, adj)
            x = F.dropout(x['h_prime'], self.dropout, training=self.training)
            x = self.out_att(x, adj)
            output = F.elu(x['h_prime'])
            return {'output':output, 'face_hand':x['face_hand']}

        else:
            x = F.dropout(x, self.dropout, training=self.training)
            x = self.attentions(x, adj)
            x = F.dropout(x['h_prime'], self.dropout, training=self.training)
            x = self.out_att(x, adj)
            output = F.elu(x['h_prime'])
            return {'output':output}
# class GAT_Euli(nn.Module):
#     def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
#         super(GAT_Euli, self).__init__()
#         self.dropout = dropout
#         self.attentions = [GraphAttentionLayer_Euli(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
#         for i, attention in enumerate(self.attentions):
#             self.add_module('attention_{}'.format(i), attention)
#         self.out_att = GraphAttentionLayer_Euli(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)
#
#     def forward(self, x, adj):
#         x = F.dropout(x, self.dropout, training=self.training)
#         x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
#         x = F.dropout(x, self.dropout, training=self.training)
#         x = F.elu(self.out_att(x, adj))
#         return x
#
# class GAT_cos(nn.Module):
#     def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
#         super(GAT_cos, self).__init__()
#         self.dropout = dropout
#         self.attentions = [GraphAttentionLayer_cos(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
#         for i, attention in enumerate(self.attentions):
#             self.add_module('attention_{}'.format(i), attention)
#         self.out_att = GraphAttentionLayer_cos(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)
#
#     def forward(self, x, adj):
#         x = F.dropout(x, self.dropout, training=self.training)
#         x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
#         x = F.dropout(x, self.dropout, training=self.training)
#         x = F.elu(self.out_att(x, adj))
#         return x

class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        super(GAT, self).__init__()
        self.dropout = dropout
        self.attentions = GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True)
        # self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        # for i, attention in enumerate(self.attentions):
        #     self.add_module('attention_{}'.format(i), attention)
        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    # def forward(self, x, adj):
    #     x = F.dropout(x, self.dropout, training=self.training)
    #     x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
    #     x = F.dropout(x, self.dropout, training=self.training)
    #     x = F.elu(self.out_att(x, adj))
    #     return x

    def forward(self, x, adj):
        if len(x)==4:
            x = F.dropout(x, self.dropout, training=self.training)
            x = self.attentions(x, adj)
            x = F.dropout(x['h_prime'], self.dropout, training=self.training)
            x = self.out_att(x, adj)
            output = F.elu(x['h_prime'])
            return {'output':output, 'face_Rhand':x['face_Rhand'],'face_Lhand': x['face_Lhand'],'Rhand_Lhand': x['Rhand_Lhand']}
        else:
            x = F.dropout(x, self.dropout, training=self.training)
            x = self.attentions(x, adj)
            x = F.dropout(x['h_prime'], self.dropout, training=self.training)
            x = self.out_att(x, adj)
            output = F.elu(x['h_prime'])
            return {'output':output, 'face_hand':x['face_hand']}



class SpGAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        super(SpGAT, self).__init__()
        self.dropout = dropout

        self.attentions = [SpGraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = SpGraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return x

        # self.dropout = dropout
    #     self.attentions = [SpGraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
    #     for i, attention in enumerate(self.attentions):
    #         self.add_module('attention_{}'.format(i), attention)
    #     self.out_att = SpGraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)
    #
    # def forward(self, x, adj):
    #     x = F.dropout(x, self.dropout, training=self.training)
    #     x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
    #     x = F.dropout(x, self.dropout, training=self.training)
    #     x = F.elu(self.out_att(x, adj))
    #     return F.log_softmax(x, dim=1)