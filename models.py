import pdb
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from dgl import function as fn
import dgl.ops as ops
from dgl.nn.pytorch.conv import GraphConv, GATConv
from dgl.nn.functional import edge_softmax
import numpy as np
import math
from utils import sample_vec, KLDivergence, WDistance, pair_norm
import random
import pdb


class FermiDiracDecoder(nn.Module):
    def __init__(self, r, t):
        super(FermiDiracDecoder, self).__init__()
        self.r = r
        self.t = t


    def forward(self, dist):
        probs = 1. / (torch.exp((dist - self.r) / self.t) + 1.0)
        return probs



class MLP(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):   
        super(MLP, self).__init__()

        self.linear_or_not = True 
        self.num_layers = num_layers

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.batch_norms = torch.nn.ModuleList()
        
            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

            for layer in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d((hidden_dim)))
    

    def reset_parameters(self):
        if self.linear_or_not:
            nn.init.xavier_normal_(self.linear.weight)
        else:
            for i in range(self.num_layers):
                nn.init.xavier_normal_(self.linears[i].weight)


    def forward(self, x):
        if self.linear_or_not:
            return self.linear(x)
        else:
            h = x
            for layer in range(self.num_layers - 1):
                h = self.linears[layer](h)
                h = self.batch_norms[layer](h)
                h = F.relu(h)
            return self.linears[self.num_layers - 1](h)



# Convolution Block
class ConvLayer(nn.Module):
    def __init__(self, h_dim, eta, dropout = 0.0, num_head = 1):
        super(ConvLayer, self).__init__()

        self.eta = eta
        self.num_head = num_head
        self.wh = nn.Linear(h_dim * 2, num_head, bias = False)
        self.W = nn.Linear(h_dim, h_dim)

        self.reset_parameters()
    

    def reset_parameters(self):
        nn.init.xavier_normal_(self.wh.weight)
        nn.init.xavier_normal_(self.W.weight)
            

    def edge_applying_fea(self, edges):
        h_src = edges.src['h']
        h_dst = edges.dst['h']
        weight = self.wh(torch.cat((h_src, h_dst), dim=1))
        return {'wf': F.leaky_relu(weight)}
    

    def edge_applying_tax(self, edges):
        h_src = edges.src['tax']
        h_dst = edges.dst['tax']
        weight = torch.einsum('ij,ij->i', h_src, h_dst).reshape(-1,1)
        return {'wt': weight}


    def forward(self, g):
        g.apply_edges(self.edge_applying_fea)
        alpha_f = edge_softmax(g, g.edata['wf'])

        g.apply_edges(self.edge_applying_tax)
        alpha_t = edge_softmax(g, g.edata['wt'])

        g.edata['alpha'] = self.eta * alpha_f + (1 - self.eta) * alpha_t
        g.update_all(fn.u_mul_e('h', 'alpha', '_'), fn.sum('_', 'z'))
        
        h = g.ndata['z']
        h = self.W(h)

        return h



class TaxoGNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout, num_layer, 
                 taxo_p2c, taxo_c2p_prob, num_taxo, leaf_taxos,
                 inner_taxos, eta, use_random, args):
        super(TaxoGNN, self).__init__()

        self.num_layer = num_layer
        self.dropout = dropout
        self.taxo_layers = len(taxo_p2c)
        self.taxo_dim = out_dim // (self.taxo_layers + 1)
        self.item_dim = out_dim - self.taxo_layers * self.taxo_dim
        self.inner_num = len(inner_taxos)
        self.taxo_p2c = taxo_p2c
        self.taxo_c2p_prob = taxo_c2p_prob
        self.num_taxo = num_taxo
        self.taxo_id = torch.LongTensor([i for i in range(num_taxo)])
        self.hidden_dim = hidden_dim

        # self.activation = torch.tanh
        self.activation = nn.PReLU()
        # self.activation = F.elu
        # self.activation = torch.relu

        self.W = nn.Linear(in_dim, hidden_dim)

        self.taxo_mean = nn.Embedding(num_taxo, hidden_dim)
        self.taxo_std_log = nn.Embedding(num_taxo, hidden_dim)

        self.conv_layers = nn.ModuleList([ConvLayer(hidden_dim, eta, dropout=dropout)] * self.num_layer)
        # self.conv_layers = nn.ModuleList([GraphConv(hidden_dim, hidden_dim, norm='both', weight=True, bias=False, activation=None)] * self.num_layer)
        self.psi = nn.ModuleList([nn.Bilinear(hidden_dim, hidden_dim, 1)] * self.taxo_layers)

        self.mlps = nn.ModuleList()
        for _ in range(self.taxo_layers):
            self.mlps.append(MLP(1, hidden_dim, hidden_dim, self.taxo_dim))
        self.mlps.append(MLP(1, hidden_dim, hidden_dim, self.item_dim))

        self.emb = None
        self.idx = None
        if use_random >= 0:
            self.emb = nn.Embedding(use_random, 64)
            self.idx = torch.LongTensor([i for i in range(use_random)])

        self.xent = nn.BCEWithLogitsLoss()

        self.reset_parameters(args)


    def reset_parameters(self, args):
        for i in range(self.taxo_layers):
            nn.init.xavier_normal_(self.psi[i].weight)
        nn.init.uniform_(self.taxo_mean.weight, args.min_mean, args.max_mean)
        nn.init.uniform_(self.taxo_std_log.weight, args.min_std, args.max_std)
        if self.emb is not None:
            nn.init.normal_(self.emb.weight)


    def forward(self, g, h, taxo_cats, taxo2nodes, idx_adj, idx_adj_norm, device, tp_mask = None):
        if self.emb is not None:
            h = self.emb(self.idx)
        h = self.W(h)
        h = self.activation(h)
        h = F.dropout(h, self.dropout, training = self.training)

        g.ndata['h'] = h

        # for each node, sample tax-aware embedding and then save to dgl
        tax_mean_all = []
        tax_std_all = []
        for i in range(self.taxo_layers):
            taxo_index = taxo_cats[:,i]
            means = self.taxo_mean(taxo_index)
            stds = torch.exp(self.taxo_std_log(taxo_index))
            tax_mean_all.append(means)
            tax_std_all.append(stds)
        tax_mean_all = torch.stack(tax_mean_all)
        tax_std_all = torch.stack(tax_std_all)

        tax_mean_concat = tax_mean_all.transpose(1, 0) # (node, layer, dim)
        tax_mean_concat = tax_mean_concat.reshape(tax_mean_concat.shape[0], -1)
        tax_mean_concat = torch.spmm(idx_adj_norm, tax_mean_concat)
        g.ndata['tax'] = tax_mean_concat

        # calculate attention & convolution
        for i in range(self.num_layer):
            h = self.conv_layers[i](g)
            if i < self.num_layer - 1:
                h = self.activation(h)
            g.ndata['h'] = h

        raw = h

        emb_h = self.mlps[-1](h)
        emb_h = F.normalize(emb_h, dim=1)

        # layer-aware embedding
        emb_list = []
        for i in range(self.taxo_layers):
            h_rev = torch.spmm(idx_adj, h)
            stpl = self.psi[i](h_rev, tax_mean_all[i])
            stpl = torch.sigmoid(stpl).reshape(-1, 1)
            tau = stpl * torch.exp(-tax_std_all[i])
            # tau = 0.5
            if tp_mask is not None:
                tau = tp_mask * tau
            cur_emb = (1 - tau) * h_rev + tau * tax_mean_all[i]
            cur_emb = torch.spmm(idx_adj_norm, cur_emb)
            emb = self.mlps[i](cur_emb)
            emb = F.normalize(emb, dim=1)
            emb_list.append(emb)
        emb = torch.stack(emb_list, dim=2)
        emb = emb.reshape(emb.shape[0], -1)

        emb = torch.cat((emb_h, emb), dim=1)

        return raw, emb


    def loss_rec(self, embeds1, embeds2, neg_sap):
        dis = torch.sum((embeds1 - embeds2) ** 2, dim=1)
        pos = -torch.mean(F.logsigmoid(-dis))
        neg = 0.0
        for j in range(0,len(neg_sap)):
            dis = torch.sum((embeds1 - neg_sap[j]) ** 2, dim=1)
            tmp = -torch.mean(F.logsigmoid(dis))
            neg += tmp
        return pos + neg
    

    # pos_node_emb: (#leaf * #samples, d)
    def cal_loss_leaf(self, leafs, raw, pos_node_samples, neg_node_samples, k, ispdb = False):
        taxo_m = self.taxo_mean(self.taxo_id)[leafs]
        taxo_std = torch.exp(self.taxo_std_log(self.taxo_id))[leafs]
        taxo_m = torch.repeat_interleave(taxo_m, k, dim=0)
        taxo_std = torch.repeat_interleave(taxo_std, k, dim=0)

        taxo_samp = sample_vec(taxo_m, taxo_std) # (#leaf * #samples, d)

        pos_node_emb = raw[pos_node_samples]
        neg_node_emb = raw[neg_node_samples]

        pred_pos = (taxo_samp * pos_node_emb).sum(dim = 1)
        info_pos = self.xent(pred_pos, torch.ones_like(pred_pos))
        pred_neg = (taxo_samp * neg_node_emb).sum(dim = 1)
        info_neg = self.xent(pred_neg, torch.zeros_like(pred_neg))

        # if ispdb:
        #     pdb.set_trace()
        
        loss_leaf = info_pos + info_neg
        return loss_leaf


    def cal_loss_inner(self, sample_num = 20):
        taxo_m = self.taxo_mean(self.taxo_id)
        taxo_std = torch.exp(self.taxo_std_log(self.taxo_id))

        loss_inner = 0.0

        for i in range(self.taxo_layers-1, -1, -1):
            cur_layer = self.taxo_p2c[str(i)]
            m_pred = []
            std_pred = []
            mix_true = []
            for parent, children in cur_layer.items():
                m_true = taxo_m[children]
                std_true = taxo_std[children]
                for _ in range(sample_num):
                    m_pred.append(taxo_m[int(parent)])
                    std_pred.append(taxo_std[int(parent)])

                    samp_true = sample_vec(m_true, std_true)
                    mix_true.append((self.taxo_c2p_prob[children].reshape(-1,1) * samp_true).sum(dim=0))

            m_pred = torch.stack(m_pred)
            std_pred = torch.stack(std_pred)
            mix_true = torch.stack(mix_true)

            samp_pred = sample_vec(m_pred, std_pred)
            false_idx = [i for i in range(samp_pred.shape[0])]
            random.shuffle(false_idx)
            mix_false = mix_true[false_idx]

            pred_pos = (samp_pred * mix_true).sum(dim=1)
            info_pos = self.xent(pred_pos, torch.ones_like(pred_pos))
            loss_inner += info_pos * (self.taxo_layers - i)
            if i > 0:
                pred_neg = (samp_pred * mix_false).sum(dim=1)
                info_neg = self.xent(pred_neg, torch.zeros_like(pred_neg))
                loss_inner += info_neg * (self.taxo_layers - i)      

        return loss_inner

    
    def cal_loss_sim(self, tag1, tag2, neg, m):
        mean_1 = self.taxo_mean(tag1)
        mean_2 = self.taxo_mean(tag2)
        mean_n = self.taxo_mean(neg)
        var_1 = torch.exp(self.taxo_std_log(tag1)) ** 2
        var_2 = torch.exp(self.taxo_std_log(tag2)) ** 2
        var_n = torch.exp(self.taxo_std_log(neg)) ** 2
        
        # dis_pos = KLDivergence(mean_1, mean_2, var_1, var_2, self.hidden_dim)
        # dis_neg = KLDivergence(mean_1, mean_n, var_1, var_n, self.hidden_dim)
        dis_pos = WDistance(mean_1, mean_2, var_1, var_2)
        dis_neg = WDistance(mean_1, mean_n, var_1, var_n)
        loss = torch.max(torch.zeros_like(dis_pos), 
                    m * torch.ones_like(dis_pos) + dis_pos - dis_neg)
        return loss.mean(dim=0)


    def clip_std(self, min_std, max_std):
        U = self.taxo_std_log.weight.data
        U.copy_(torch.clamp(U, min_std, max_std))


    # def cal_loss_tree(self, sample_node, raw, min_std=0.05):
    #     taxo_m = self.taxo_mean(self.taxo_id)
    #     taxo_v = torch.exp(self.taxo_std_log(self.taxo_id)) ** 2

    #     mean_list_true = []
    #     var_list_true = []
    #     mean_list_pred = []
    #     var_list_pred = []
    #     for k, n in sample_node.items():
    #         mean_list_true.append(raw[n].mean(dim=0))
    #         mean_list_pred.append(taxo_m[k])
    #         std = raw[n].std(dim=0, unbiased = True)
    #         std = torch.max(std, min_std * torch.ones_like(std))
    #         var_list_true.append(std ** 2)
    #         var_list_pred.append(taxo_v[k])
    #     mean_list_true = torch.stack(mean_list_true)
    #     mean_list_pred = torch.stack(mean_list_pred)
    #     var_list_true = torch.stack(var_list_true)
    #     var_list_pred = torch.stack(var_list_pred)
    #     loss_leaf = WDistance(mean_list_true, mean_list_pred, var_list_true, var_list_pred).mean()            

    #     mean_list_true = []
    #     var_list_true = []
    #     mean_list_pred = []
    #     var_list_pred = []
    #     for i in range(self.taxo_layers-1, 0, -1):
    #         cur_layer = self.taxo_p2c[str(i)]
    #         for parent, children in cur_layer.items():
    #             mean_emb = (self.taxo_c2p_prob[children].reshape(-1,1) * taxo_m[children]).sum(dim=0)
    #             mean_list_true.append(mean_emb)
    #             mean_list_pred.append(taxo_m[int(parent)])
    #             var_emb = (self.taxo_c2p_prob[children].reshape(-1,1) * 
    #                         ((taxo_m[children] ** 2) + taxo_v[children])).sum(dim=0) - mean_emb ** 2
    #             var_list_true.append(var_emb)
    #             var_list_pred.append(taxo_v[int(parent)])
    #     mean_list_true = torch.stack(mean_list_true)
    #     mean_list_pred = torch.stack(mean_list_pred)
    #     var_list_true = torch.stack(var_list_true)
    #     var_list_pred = torch.stack(var_list_pred)
    #     loss_inner = WDistance(mean_list_true, mean_list_pred, var_list_true, var_list_pred).mean()

    #     return loss_leaf + loss_inner, loss_leaf, loss_leaf, loss_inner, loss_inner    