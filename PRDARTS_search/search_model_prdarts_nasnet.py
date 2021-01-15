import torch
import torch.nn as nn
from copy import deepcopy
from search_cells import NASNetSearchCell as SearchCell
import numpy as np
import pdb


# The macro structure is based on NASNet
class NASNetworkPRDARTS(nn.Module):

    def __init__(self, C, N, steps, multiplier, stem_multiplier, num_classes, search_space, affine,
                 track_running_stats):
        super(NASNetworkPRDARTS, self).__init__()
        self._C = C
        self._layerN = N
        self._steps = steps
        self._multiplier = multiplier
        self.stem = nn.Sequential(
            nn.Conv2d(3, C * stem_multiplier, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(C * stem_multiplier))

        # config for each layer
        layer_channels = [C] * N + [C * 2] + [C * 2] * (N - 1) + [C * 4] + [C * 4] * (N - 1)
        layer_reductions = [False] * N + [True] + [False] * (N - 1) + [True] + [False] * (N - 1)


        num_edge, edge2index = None, None
        C_prev_prev, C_prev, C_curr, reduction_prev = C * stem_multiplier, C * stem_multiplier, C, False

        self.cells = nn.ModuleList()
        for index, (C_curr, reduction) in enumerate(zip(layer_channels, layer_reductions)):
            cell = SearchCell(search_space, steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev,
                              affine, track_running_stats)
            print(index)
            print(search_space, steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev, affine,
                  track_running_stats)
            # pdb.set_trace()
            if num_edge is None:
                num_edge, edge2index = cell.num_edges, cell.edge2index
            else:
                assert num_edge == cell.num_edges and edge2index == cell.edge2index, 'invalid {:} vs. {:}.'.format(
                    num_edge, cell.num_edges)
            self.cells.append(cell)
            C_prev_prev, C_prev, reduction_prev = C_prev, multiplier * C_curr, reduction

        self.op_names = deepcopy(search_space)
        self._Layer = len(self.cells)
        self.edge2index = edge2index
        self.lastact = nn.Sequential(nn.BatchNorm2d(C_prev), nn.ReLU(inplace=True))
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)

        self.arch_normal_parameters = nn.Parameter(1e-3 * torch.randn(num_edge, len(search_space)))
        self.arch_reduce_parameters = nn.Parameter(1e-3 * torch.randn(num_edge, len(search_space)))

        self.num_edge = num_edge
        self.tau = 10
        self.limit_a = -0.1
        self.limit_b = 1.1
        self.k = 2
        self.dropout_pro = 0.2
        self.regular_lambda1=1.0
        self.regular_lambda2=0.5
        self.regular_lambda3=0.5
        self.threshold = 0.2

    def set_threshold(self, threshold):
        self.threshold = threshold

    def set_dropout(self, dropout_pro):
        self.dropout_pro = dropout_pro

    def get_weights(self):
        xlist = list(self.stem.parameters()) + list(self.cells.parameters())
        xlist += list(self.lastact.parameters()) + list(self.global_pooling.parameters())
        xlist += list(self.classifier.parameters())
        return xlist

    def set_tau(self, tau):
        self.tau = tau

    def set_k(self, k):
        self.k = k

    def get_tau(self):
        return self.tau

    def set_sparsity(self, sparsity):
        self.sparsity = sparsity

    def set_ab(self, a, b):
        self.limit_a = a
        self.limit_b = b

    def set_weights(self, a, b):
        self.visit_weight = a
        self.loss_weight = b

    def get_alphas(self):
        return [self.arch_normal_parameters, self.arch_reduce_parameters]

    def get_a(self):
        return self.limit_a

    def get_b(self):
        return self.limit_b

    def get_sparsity(self):
        return self.sparsity

    def sampling_hard_gate(self, xins):
        # probs = torch.sigmoid(xins - self.tau * np.log(-self.limit_a/self.limit_b))

        logits = (xins + 0.5) / self.tau
        probs = torch.sigmoid(logits) * (self.limit_b - self.limit_a) + self.limit_a
        probs = probs.clamp_(min=0.0, max=1.0)
        return probs

    def show_alphas(self):
        with torch.no_grad():
            probs_A = self.sampling_hard_gate(self.arch_normal_parameters)
            probs_B = self.sampling_hard_gate(self.arch_reduce_parameters)

            A = 'arch-normal-parameters :\n{:}'.format(probs_A.cpu())
            B = 'arch-reduce-parameters :\n{:}'.format(probs_B.cpu())
        return '{:}\n{:}'.format( A, B)

    def get_message(self):
        string = self.extra_repr()
        for i, cell in enumerate(self.cells):
            string += '\n {:02d}/{:02d} :: {:}'.format(i, len(self.cells), cell.extra_repr())
        return string

    def extra_repr(self):
        return ('{name}(C={_C}, N={_layerN}, steps={_steps}, multiplier={_multiplier}, L={_Layer})'.format(
            name=self.__class__.__name__, **self.__dict__))

    def genotype(self):
        def _parse(weights):
            gene = []
            for i in range(self._steps):
                edges = []
                for j in range(2 + i):
                    node_str = '{:}<-{:}'.format(i, j)
                    ws = weights[self.edge2index[node_str]]
                    for k, op_name in enumerate(self.op_names):
                        if op_name == 'none': continue
                        edges.append((op_name, j, ws[k]))
                edges = sorted(edges, key=lambda x: -x[-1])
                selected_edges = edges[:2]
                gene.append(tuple(selected_edges))
            return gene

        with torch.no_grad():
            gene_normal = _parse(torch.softmax(self.arch_normal_parameters, dim=-1).cpu().numpy())
            gene_reduce = _parse(torch.softmax(self.arch_reduce_parameters, dim=-1).cpu().numpy())
        return {'normal': gene_normal,
                'normal_concat': list(range(2 + self._steps - self._multiplier, self._steps + 2)),
                'reduce': gene_reduce,
                'reduce_concat': list(range(2 + self._steps - self._multiplier, self._steps + 2))}


    def set_regular_lambda(self, regular_lambda1,  regular_lambda2,  regular_lambda3):
        self.regular_lambda1, self.regular_lambda2, self.regular_lambda3=regular_lambda1,  regular_lambda2,  regular_lambda3

    def get_probability(self):
        para = self.tau * np.log(-self.limit_a / self.limit_b)
        normal_probability = (self.arch_normal_parameters - para).sigmoid().mean().cpu().item()
        # pdb.set_trace()
        # normal_probability =normal_probability.cpu().item()
        reduction_probability = (self.arch_reduce_parameters - para).sigmoid().mean().cpu().item()
        logits = (self.arch_normal_parameters + 0.5) / self.tau
        probs = torch.sigmoid(logits) * (self.limit_b - self.limit_a) + self.limit_a
        normal_probability2 = probs.clamp_(min=0.0, max=1.0).mean().cpu().item()
        logits = (self.arch_reduce_parameters + 0.5) / self.tau
        probs = torch.sigmoid(logits) * (self.limit_b - self.limit_a) + self.limit_a
        reduction_probability2 = probs.clamp_(min=0.0, max=1.0).mean().cpu().item()
        probabilitys = [normal_probability, reduction_probability, \
                        normal_probability2, reduction_probability2]

        return probabilitys

    def get_regularization(self,normal_hardwts_idx,reduce_hardwts_idx):
        para = self.tau * np.log(-self.limit_a / self.limit_b)

        ## expectation activation probability
        normal_cell_activation = (self.arch_normal_parameters - para).sigmoid().mul(normal_hardwts_idx)
        reduction_cell_activation = (self.arch_reduce_parameters - para).sigmoid().mul(reduce_hardwts_idx)

        ##  expectation activation probability of skip group, idx 1 refers to skip connection
        skip_group_activation = 0.5 * normal_cell_activation[:,1].sum()/normal_hardwts_idx[:,1].sum() if normal_hardwts_idx[:,1].sum()>0 else 0.0
        skip_group_activation += (0.5 * reduction_cell_activation[:,1].sum()/reduce_hardwts_idx[:,1].sum() if reduce_hardwts_idx[:,1].sum()>0 else 0.0)

        # ##  expectation activation probability of skip group, idx 1 refers to skip connection
        # edge_number_normal = normal_hardwts_idx.sum() - normal_hardwts_idx[:,1].sum()
        # edge_number_reduction = reduce_hardwts_idx.sum() - reduce_hardwts_idx[:,1].sum()
        # nonskip_group_activation = (0.5/edge_number_normal) * (normal_cell_activation[:,0].sum() + normal_cell_activation[:,2:].sum()) if edge_number_normal else 0.0
        # nonskip_group_activation += ((0.5/edge_number_reduction) *(reduction_cell_activation[:,0].sum() + reduction_cell_activation[:,2:].sum()) if edge_number_reduction else 0.0)

        ## pooling
        edge_number_normal_pooling = normal_hardwts_idx[:, 6:8].sum()
        edge_number_reduction_pooling = reduce_hardwts_idx[:, 6:8].sum()
        pooling_group_activation = (0.5 / edge_number_normal_pooling) * normal_cell_activation[:, 6:8].sum() \
                                         if edge_number_normal_pooling else 0.0
        pooling_group_activation += ((0.5 / edge_number_reduction_pooling) * reduction_cell_activation[:, 6:8].sum() \
                                         if edge_number_reduction_pooling else 0.0)

        ##  expectation activation probability of skip group, idx 1 refers to skip connection
        edge_number_normal = normal_hardwts_idx.sum() - normal_hardwts_idx[:,1].sum() - edge_number_normal_pooling
        edge_number_reduction = reduce_hardwts_idx.sum() - reduce_hardwts_idx[:,1].sum() - edge_number_reduction_pooling
        nonskip_group_activation = (0.5/edge_number_normal) * (normal_cell_activation[:,0].sum() + normal_cell_activation[:,2:6].sum()) if edge_number_normal else 0.0
        nonskip_group_activation += ((0.5/edge_number_reduction) *(reduction_cell_activation[:,0].sum() + reduction_cell_activation[:,2:6].sum()) if edge_number_reduction else 0.0)


        ## expectation of path regularization
        # if idx.is_cuda:
        #     rand_idx = rand_idx.cuda()
        # pdb.set_trace()
        path_idx = torch.tensor([self.edge2index['0<-1'],self.edge2index['1<-2'],self.edge2index['2<-3'],self.edge2index['3<-4']])
        # pdb.set_trace()
        select_idx = path_idx[normal_hardwts_idx[path_idx,2:6].sum(dim=1).nonzero().squeeze()]
        # select_idx = path_idx[AA[path_idx, 2:6].sum(dim=1).nonzero().squeeze()]
        path_normal_activation = 0.0
        if select_idx.numel()==1:
            # pdb.set_trace()
            path_normal_activation = normal_cell_activation[select_idx,2:6].sum().mul(1.0/normal_hardwts_idx[select_idx,2:6].sum())
        if select_idx.numel() > 1:
            path_normal_activation = normal_cell_activation[select_idx, 2:6].sum(dim=1).mul(
                1.0 / normal_hardwts_idx[select_idx, 2:6].sum(dim=1)).prod()

        path_reducation_activation = 0.0
        select_idx = path_idx[reduce_hardwts_idx[path_idx,2:6].sum(dim=1).nonzero().squeeze()]
        if select_idx.numel()==1:
            path_reducation_activation = reduction_cell_activation[select_idx,2:6].sum().mul(1.0/reduce_hardwts_idx[select_idx,2:6].sum())
        if select_idx.numel() > 1:
            path_reducation_activation = reduction_cell_activation[select_idx, 2:6].sum(dim=1).mul(
                1.0 / reduce_hardwts_idx[select_idx, 2:6].sum(dim=1)).prod()

        path_activation = 0.5*(path_normal_activation + path_reducation_activation)



        ## combine together
        skip_group_activation = skip_group_activation - self.threshold if skip_group_activation > self.threshold else 0.0
        nonskip_group_activation = nonskip_group_activation - self.threshold if nonskip_group_activation > self.threshold else 0.0
        pooling_group_activation = pooling_group_activation - self.threshold if pooling_group_activation > self.threshold else 0.0
        regularizer = self.regular_lambda1 * skip_group_activation \
                      + self.regular_lambda2 * (nonskip_group_activation + pooling_group_activation) \
                      - self.regular_lambda3 * path_activation

        # regularizer = self.regular_lambda1 * skip_group_activation \
        #               + self.regular_lambda2 * (nonskip_group_activation + pooling_group_activation) \
        #               - self.regular_lambda3 * path_activation
        return regularizer
        #skip_group_activation, nonskip_group_activation, path_activation


    def forward(self, inputs):
        def get_gumbel_prob(xins):
            while True:
                # pdb.set_trace()
                gumbels = torch.rand_like(xins)  # .to('cuda')
                gumbels = torch.log(gumbels) - torch.log(1 - gumbels)
                # pdb.set_trace()
                # logits = (xins.sigmoid().log() + gumbels) / self.tau
                logits = (xins + gumbels) / self.tau

                probs = torch.sigmoid(logits) * (self.limit_b - self.limit_a) + self.limit_a
                idx = probs.clamp_(min=0.0, max=1.0)
                # pdb.set_trace()
                if self.dropout_pro <= 0.0:
                    if idx.sum():
                        one_num = idx.ge(1.0).int().sum(dim=1)## number of operation > 1.0, we need to randomly select
                        if one_num.ge(self.k).sum():  ## number of operation > 1.0 is larger than k, we need to randomly selectk ops
                            rand_idx = torch.randperm(idx.shape[1])
                            if idx.is_cuda:
                                rand_idx = rand_idx.cuda()
                            idx = idx[:, rand_idx]
                            _, top_idx = torch.topk(idx, dim=1, k=self.k)
                            rand_idx = rand_idx.repeat(idx.shape[0], 1)
                            top_idx = rand_idx.gather(dim=1, index=top_idx)
                        else:
                            _, top_idx = torch.topk(idx, dim=1, k=self.k)
                        hardwts_idx = torch.zeros_like(idx).scatter_(-1, top_idx, 1.0)
                        # without gradient:
                        #     hardwts_idx = torch.zeros_like(idx).scatter_(-1, top_idx, 1.0)
                        hardwts = hardwts_idx - probs.detach() + probs
                        break
                else:  # dropout
                    ber_drop = - torch.bernoulli(torch.zeros_like(idx), self.dropout_pro)  ## add -1 to the death nodes to avoid selected
                    idx = idx + ber_drop
                    rand_idx = torch.randperm(idx.shape[1])
                    if idx.is_cuda:
                        rand_idx = rand_idx.cuda()
                    idx = idx[:, rand_idx]
                    _, top_idx = torch.topk(idx, dim=1, k=self.k)
                    rand_idx = rand_idx.repeat(idx.shape[0], 1)
                    top_idx = rand_idx.gather(dim=1, index=top_idx)
                    hardwts_idx = torch.zeros_like(idx).scatter_(-1, top_idx, 1.0)
                    hardwts = hardwts_idx - probs.detach() + probs
                    break
            return hardwts_idx, hardwts

        normal_hardwts_idx, normal_hardwts = get_gumbel_prob(self.arch_normal_parameters)
        reduce_hardwts_idx, reduce_hardwts = get_gumbel_prob(self.arch_reduce_parameters)

        s0 = s1 = self.stem(inputs)
        for i, cell in enumerate(self.cells):
            if cell.reduction:
                hardwts = reduce_hardwts
            else:
                hardwts = normal_hardwts
            s0, s1 = s1, cell.forward_prdarts(s0, s1, hardwts)

        out = self.lastact(s1)
        out = self.global_pooling(out)
        out = out.view(out.size(0), -1)
        logits = self.classifier(out)

        return out, logits, self.get_regularization(normal_hardwts_idx, reduce_hardwts_idx)