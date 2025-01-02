import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAttentionLayer_cos(nn.Module):
    def __init__(self, in_dim, out_dim,dropout, alpha, concat=True):
        super(GraphAttentionLayer_cos, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.W = nn.Parameter(torch.zeros(size=(in_dim, out_dim)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.a = nn.Parameter(torch.zeros(size=(2 * out_dim, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

    def forward(self, input, adj):
        h = torch.matmul(input, self.W)
        N = h.size()[0]

        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_dim)
        e = F.leaky_relu(torch.matmul(a_input, self.a).squeeze(2))

        face_Rhand = e[1][2]
        face_Lhand = e[1][3]
        Rhand_Lhand = e[3][2]

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        alpha = F.softmax(attention, dim=1)
        h_prime = torch.matmul(alpha, h)
        # return h_prime
        return h_prime,face_Rhand,face_Lhand,Rhand_Lhand

class GraphAttentionLayer_Euli(nn.Module):
    def __init__(self, in_features, out_features,dropout, alpha, concat=True):
        super(GraphAttentionLayer_Euli, self).__init__()
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.alpha = alpha
        self.dropout = dropout
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.concat = concat

    def forward(self, h, adj):
        h = torch.matmul(h, self.W)
        # Compute pairwise euclidean distances between nodes
        dist = torch.cdist(h, h)
        e = self.leakyrelu(dist)

        if len(h)==4:
            face_Rhand = e[1][2]
            face_Lhand = e[1][3]
            Rhand_Lhand = e[3][2]

            zero_vec = -9e15 * torch.ones_like(e)
            attention = torch.where(adj > 0, e, zero_vec)

            attention = torch.softmax(attention, dim=0)
            attention = F.dropout(attention, self.dropout, training=self.training)
            h_prime = torch.matmul(attention, h)

            if self.concat:
                return {'h_prime':h_prime, 'face_Rhand':face_Rhand,'face_Lhand': face_Lhand,'Rhand_Lhand': Rhand_Lhand}
            else:
                return {'h_prime':h_prime, 'face_Rhand':face_Rhand,'face_Lhand': face_Lhand,'Rhand_Lhand': Rhand_Lhand}
        else:
            face_hand = e[1][2]

            zero_vec = -9e15 * torch.ones_like(e)
            attention = torch.where(adj > 0, e, zero_vec)

            attention = F.softmax(attention, dim=1)
            attention = F.dropout(attention, self.dropout, training=self.training)
            h_prime = torch.matmul(attention, h)

            if self.concat:
                return {'h_prime':h_prime,'face_hand':face_hand}
            else:
                return {'h_prime':h_prime,'face_hand':face_hand}


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        Wh = torch.mm(h, self.W)  # h.shape: (N, in_features), Wh.shape: (N, out_features)
        e = self._prepare_attentional_mechanism_input(Wh)
        if len(h)==4:
            face_Rhand = e[1][2]
            face_Lhand = e[1][3]
            Rhand_Lhand = e[3][2]

            zero_vec = -9e15 * torch.ones_like(e)
            attention = torch.where(adj > 0, e, zero_vec)

            attention = F.softmax(attention, dim=1)
            attention = F.dropout(attention, self.dropout, training=self.training)
            h_prime = torch.matmul(attention, Wh)

            if self.concat:
                return {'h_prime':h_prime, 'face_Rhand':face_Rhand,'face_Lhand': face_Lhand,'Rhand_Lhand': Rhand_Lhand}
            else:
                return {'h_prime':h_prime, 'face_Rhand':face_Rhand,'face_Lhand': face_Lhand,'Rhand_Lhand': Rhand_Lhand}
        else:
            face_hand = e[1][2]

            zero_vec = -9e15 * torch.ones_like(e)
            attention = torch.where(adj > 0, e, zero_vec)

            attention = F.softmax(attention, dim=1)
            attention = F.dropout(attention, self.dropout, training=self.training)
            h_prime = torch.matmul(attention, Wh)

            if self.concat:
                return {'h_prime':h_prime,'face_hand':face_hand}
            else:
                return {'h_prime':h_prime,'face_hand':face_hand}


    def _prepare_attentional_mechanism_input(self, Wh):

        # Wh.shape (N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (N, 1)
        # e.shape (N, N)
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        # broadcast add
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)


class SpGraphAttentionLayer(nn.Module):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(SpGraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        Wh = torch.mm(h, self.W)  # h.shape: (N, in_features), Wh.shape: (N, out_features)
        e = self._prepare_attentional_mechanism_input(Wh)

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        # Wh.shape (N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (N, 1)
        # e.shape (N, N)
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        # broadcast add
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

# class SpecialSpmmFunction(torch.autograd.Function):
#     """Special function for only sparse region backpropataion layer."""
#
#     @staticmethod
#     def forward(ctx, indices, values, shape, b):
#         assert indices.requires_grad == False
#         a = torch.sparse_coo_tensor(indices, values, shape)
#         ctx.save_for_backward(a, b)
#         ctx.N = shape[0]
#         return torch.matmul(a, b)
#
#     @staticmethod
#     def backward(ctx, grad_output):
#         a, b = ctx.saved_tensors
#         grad_values = grad_b = None
#         if ctx.needs_input_grad[1]:
#             grad_a_dense = grad_output.matmul(b.t())
#             edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
#             grad_values = grad_a_dense.view(-1)[edge_idx]
#         if ctx.needs_input_grad[3]:
#             grad_b = a.t().matmul(grad_output)
#         return None, grad_values, None, grad_b
#
#
# class SpecialSpmm(nn.Module):
#     def forward(self, indices, values, shape, b):
#         return SpecialSpmmFunction.apply(indices, values, shape, b)
#
#
# class SpGraphAttentionLayer(nn.Module):
#     """
#     Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
#     """
#
#     def __init__(self, in_features, out_features, dropout, alpha, concat=True):
#         super(SpGraphAttentionLayer, self).__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.alpha = alpha
#         self.concat = concat
#
#         self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
#         nn.init.xavier_normal_(self.W.data, gain=1.414)
#
#         self.a = nn.Parameter(torch.zeros(size=(1, 2 * out_features)))
#         nn.init.xavier_normal_(self.a.data, gain=1.414)
#
#         self.dropout = nn.Dropout(dropout)
#         self.leakyrelu = nn.LeakyReLU(self.alpha)
#         self.special_spmm = SpecialSpmm()
#
#     def forward(self, input, adj):
#         dv = 'cuda' if input.is_cuda else 'cpu'
#
#         N = input.size()[0]
#         edge = adj.nonzero().t()
#
#         h = torch.mm(input, self.W)
#         # h: N x out
#         assert not torch.isnan(h).any()
#
#         # Self-attention on the nodes - Shared attention mechanism
#         edge_h = torch.cat((h[edge[0, :], :], h[edge[1, :], :]), dim=1).t()
#         # edge: 2*D x E
#
#         edge_e = torch.exp(-self.leakyrelu(self.a.mm(edge_h).squeeze()))
#         assert not torch.isnan(edge_e).any()
#         # edge_e: E
#
#         e_rowsum = self.special_spmm(edge, edge_e, torch.Size([N, N]), torch.ones(size=(N, 1), device='cuda'))
#         # e_rowsum: N x 1
#
#         edge_e = self.dropout(edge_e)
#         # edge_e: E
#
#         h_prime = self.special_spmm(edge, edge_e, torch.Size([N, N]), h)
#         assert not torch.isnan(h_prime).any()
#         # h_prime: N x out
#
#         h_prime = h_prime.div(e_rowsum)
#         # h_prime: N x out
#         assert not torch.isnan(h_prime).any()
#
#         if self.concat:
#             # if this layer is not last layer,
#             return F.elu(h_prime)
#         else:
#             # if this layer is last layer,
#             return h_prime
#
#     def __repr__(self):
#         return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

class GraphAttentionLayer_multiEuli(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer_multiEuli, self).__init__()
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.alpha = alpha
        self.dropout = dropout
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.concat = concat

    def forward(self, h, adj):
        h = torch.matmul(h, self.W)
        # Compute pairwise euclidean distances between nodes
        dist2 = torch.cdist(h, h)
        dist = torch.cdist(h[1:],h[1:])

        e2 = self.leakyrelu(dist2)
        e = self.leakyrelu(dist)

        if len(h) == 4:
            # e[1][2] = e[2][1] = e[1][2] + (e[0][1] + e[0][2]) * 0.5
            # e[1][3] = e[3][1] = e[1][3] + (e[0][1] + e[0][3]) * 0.5
            # e[3][2] = e[2][3] = e[2][3] + (e[0][2] + e[0][3]) * 0.5
            #
            # face_Rhand = e[1][2]
            # face_Lhand = e[1][3]
            # Rhand_Lhand = e[3][2]

            e[0][1] = e[1][0] = e2[1][2] + (e2[0][1] + e2[0][2]) * 0.5
            e[0][2] = e[2][0] = e2[1][3] + (e2[0][1] + e2[0][3]) * 0.5
            e[1][2] = e[2][1] = e2[2][3] + (e2[0][2] + e2[0][3]) * 0.5

            face_Rhand = e[0][1]
            face_Lhand = e[0][2]
            Rhand_Lhand = e[1][2]

            zero_vec = -9e15 * torch.ones_like(e)
            attention = torch.where(adj > 0, e, zero_vec)

            attention = torch.softmax(attention, dim=0)
            attention = F.dropout(attention, self.dropout, training=self.training)
            h_prime = torch.matmul(attention, h[1:])
            h_prime = torch.cat([h[0:1],h_prime],dim=0)

            if self.concat:
                return {'h_prime': h_prime, 'face_Rhand': face_Rhand, 'face_Lhand': face_Lhand, 'Rhand_Lhand': Rhand_Lhand}
            else:
                return {'h_prime': h_prime, 'face_Rhand': face_Rhand, 'face_Lhand': face_Lhand, 'Rhand_Lhand': Rhand_Lhand}
        elif len(h)==3:

            e[0][1] = e[1][0] = e2[1][2] + (e2[0][1] + e2[0][2]) * 0.5

            # e[1][2] = e[2][1] = e[1][2] + (e[0][1] + e[0][2]) * 0.5
            # face_hand = e[1][2]
            face_hand = e[0][1]

            zero_vec = -9e15 * torch.ones_like(e)
            attention = torch.where(adj > 0, e, zero_vec)

            attention = F.softmax(attention, dim=1)
            attention = F.dropout(attention, self.dropout, training=self.training)
            h_prime = torch.matmul(attention, h[1:])
            h_prime = torch.cat([h[0:1], h_prime], dim=0)

            if self.concat:
                return {'h_prime': h_prime, 'face_hand': face_hand}
            else:
                return {'h_prime': h_prime, 'face_hand': face_hand}

        else:

            # e[0][1] = e[1][0] = e2[1][2] + (e2[0][1] + e2[0][2]) * 0.5
            #
            # # e[1][2] = e[2][1] = e[1][2] + (e[0][1] + e[0][2]) * 0.5
            # # face_hand = e[1][2]
            # face_hand = e[0][1]
            #
            # zero_vec = -9e15 * torch.ones_like(e)
            # attention = torch.where(adj > 0, e, zero_vec)
            #
            # attention = F.softmax(attention, dim=1)
            # attention = F.dropout(attention, self.dropout, training=self.training)
            # h_prime = torch.matmul(attention, h[1:])
            # h_prime = torch.cat([h[0:1], h_prime], dim=0)
            #
            # if self.concat:
            return {'h_prime': h}
            # else:
            #     return {'h_prime': h_prime, 'face_hand': face_hand}