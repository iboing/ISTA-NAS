import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from ..operations import *
from ..genotypes import Genotype, PRIMITIVES
from ...utils import drop_path


__all__ = ["MixedOp", "Cell", "NetWork"]


class AuxiliaryHeadCIFAR(nn.Module):

    def __init__(self, C, num_classes):
        """assuming input size 8x8"""
        super(AuxiliaryHeadCIFAR, self).__init__()
        self.features = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.AvgPool2d(5, stride=3, padding=0, count_include_pad=False), # image size = 2 x 2
            nn.Conv2d(C, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 768, 2, bias=False),
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True)
        )
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x.view(x.size(0),-1))
        return x


class MixedOp(nn.Module):

    def __init__(self, C, stride):
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()
        for primitive in PRIMITIVES:
            op = OPS[primitive](C, stride, True)
            if 'pool' in primitive:
                op = nn.Sequential(op, nn.BatchNorm2d(C, affine=True))
            if 'skip' in primitive and isinstance(op, Identity):
                op = nn.Sequential(op, nn.BatchNorm2d(C, affine=True))
            self._ops.append(op)
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
                m.weight.requires_grad = False
                m.bias.requires_grad = False

    def forward(self, x, weights, drop_prob):
        if weights.sum() == 0:
            return 0
        feats = []
        for w, op in zip(weights, self._ops):
            if w == 0:
                continue
            feat = w * op(x)
            if self.training and drop_prob > 0:
                feat = drop_path(feat, drop_prob)
            feats.append(feat)
        return sum(feats)


class Cell(nn.Module):

    def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev):
        super(Cell, self).__init__()
        self.reduction = reduction

        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=True)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=True)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=True)
        self._steps = steps
        self._multiplier = multiplier

        self._ops = nn.ModuleList()
        for i in range(self._steps):
            for j in range(2+i):
                stride = 2 if reduction and j < 2 else 1
                op = MixedOp(C, stride)
                self._ops.append(op)

    def forward(self, s0, s1, weights, drop_prob):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        offset = 0
        for i in range(self._steps):
            s = sum(self._ops[offset+j](h, weights[offset+j], drop_prob) for j, h in enumerate(states))
            offset += len(states)
            states.append(s)

        return torch.cat(states[-self._multiplier:], dim=1)


class NetWork(nn.Module):

    def __init__(self, C, num_classes, layers,
                 proj_dims=2, steps=4, multiplier=4, stem_multiplier=3, auxiliary=False):
        super(NetWork, self).__init__()
        self._C = C
        self._num_classes = num_classes
        self._layers = layers
        self._steps = steps
        self._multiplier = multiplier
        self.num_edges = sum(1 for i in range(self._steps) for n in range(2 + i))
        self.proj_dims = proj_dims
        self.auxiliary = auxiliary

        C_curr = stem_multiplier * C
        self.stem = nn.Sequential(
            nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
            nn.BatchNorm2d(C_curr))

        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        reduction_prev = False
        for i in range(layers):
            if i in [layers//3, 2*layers//3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
            reduction_prev = reduction
            self.cells.append(cell)
            C_prev_prev, C_prev = C_prev, multiplier * C_curr

            if i == 2*layers//3:
                C_to_auxiliary = C_prev

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)
        if self.auxiliary:
            self.auxiliary_head = AuxiliaryHeadCIFAR(C_to_auxiliary, num_classes)

        self._initialize_alphas()

    def new(self):
        model_new = NetWork(self._C, self._num_classes, self._layers).cuda()
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)
        return model_new

    def forward(self, input, all_freeze):
        logits_aux = None
        s0 = s1 = self.stem(input)
        if not all_freeze:
            self.proj_alphas(self.A_normals, self.A_reduces)
        for i, cell in enumerate(self.cells):
            if cell.reduction:
                weights = self.alphas_reduce
            else:
                weights = self.alphas_normal
            s0, s1 = s1, cell(s0, s1, weights, self.drop_path_prob)
            if i== 2 * self._layers//3:
                if self.auxiliary and self.training:
                    logits_aux = self.auxiliary_head(s1)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0),-1))

        return logits, logits_aux

    def _loss(self, input, target):
        logits = self(input)
        return F.cross_entropy(logits, target)

    def _initialize_alphas(self):
        self.alphas_normal_ = []
        self.alphas_reduce_ = []
        for i in range(self._steps):
            self.alphas_normal_.append(nn.Parameter(1e-3 * torch.randn(self.proj_dims, device='cuda')))
            self.alphas_reduce_.append(nn.Parameter(1e-3 * torch.randn(self.proj_dims, device='cuda')))
        self._arch_parameters = self.alphas_normal_ + self.alphas_reduce_

    def freeze_alpha(self, normal_freeze_alpha, reduce_freeze_alpha):
        offset = 0
        for i, (flag, alpha) in enumerate(zip(normal_freeze_alpha, self.alphas_normal_)):
            if flag and alpha.requires_grad:
                alpha.requires_grad = False
                for cell in self.cells:
                    if cell.reduction:
                        continue
                    for j in range(offset, offset+i+2):
                        op = cell._ops[j]
                        for m in op.modules():
                            if isinstance(m, nn.BatchNorm2d):
                                #print(m)
                                m.weight.requires_grad = True
                                m.bias.requires_grad = True

            offset += i + 2  # 0, 2, 5, 9,

        offset = 0
        for i, (flag, alpha) in enumerate(zip(reduce_freeze_alpha, self.alphas_reduce_)):
            if flag and alpha.requires_grad:
                alpha.requires_grad = False
                for cell in self.cells:
                    if not cell.reduction:
                        continue
                    for j in range(offset, offset+i+2):
                        op = cell._ops[j]
                        for m in op.modules():
                            if isinstance(m, nn.BatchNorm2d):
                                #print('weight shape:', m.weight.size(), 'value', m.weight[0])
                                #print('bias shape:', m.bias.size(), 'valud', m.bias[0])
                                m.weight.requires_grad = True
                                m.bias.requires_grad = True

            offset += i + 2

    def init_proj_mat(self, A_normals, A_reduces):
        self.A_normals = A_normals
        self.A_reduces = A_reduces

    def init_bias(self, normal_bias, reduce_bias):
        self.normal_bias = normal_bias
        self.reduce_bias = reduce_bias

    def proj_alphas(self, A_normals, A_reduces):
        assert len(A_normals) == len(A_reduces) == self._steps
        alphas_normal = []
        alphas_reduce = []
        alphas_normal_ =  torch.stack(self.alphas_normal_) #F.softmax(torch.stack(self.alphas_normal_), dim=-1)  # torch.stack(self.alphas_normal_)
        alphas_reduce_ =  torch.stack(self.alphas_reduce_) #F.softmax(torch.stack(self.alphas_reduce_), dim=-1)  # torch.stack(self.alphas_reduce_)
        for i in range(self._steps):
            A_normal = A_normals[i].to(alphas_normal_.device).requires_grad_(False)
            t_alpha = alphas_normal_[[i]].mm(A_normal).reshape(-1, len(PRIMITIVES))
            alphas_normal.append(t_alpha)
            A_reduce = A_reduces[i].to(alphas_reduce_.device).requires_grad_(False)
            t_alpha = alphas_reduce_[[i]].mm(A_reduce).reshape(-1, len(PRIMITIVES))
            alphas_reduce.append(t_alpha)

        self.alphas_normal = torch.cat(alphas_normal) - self.normal_bias.to(
             alphas_normal_.device)
        self.alphas_reduce = torch.cat(alphas_reduce) - self.reduce_bias.to(
             alphas_reduce_.device)
        #self.alphas_normal = torch.cat(alphas_normal)
        #self.alphas_reduce = torch.cat(alphas_reduce)

    def arch_parameters(self):
        return self._arch_parameters

    def genotype(self):

        def _parse(weights):
            gene = []
            n = 2
            start = 0
            for i in range(self._steps):
                end = start + n
                W = weights[start:end].copy()
                edges = sorted(range(i + 2),
                    key=lambda x: -max(W[x][k] for k in range(len(W[x]))
                                        if k != PRIMITIVES.index('none')))[:2]
                for j in edges:
                    k_best = None
                    for k in range(len(W[j])):
                        if k != PRIMITIVES.index('none'):
                            if k_best is None or W[j][k] > W[j][k_best]:
                                k_best = k
                    gene.append((PRIMITIVES[k_best], j))
                start = end
                n += 1
            return gene

        gene_normal = _parse(F.softmax(self.alphas_normal, dim=-1).data.cpu().numpy())
        gene_reduce = _parse(F.softmax(self.alphas_reduce, dim=-1).data.cpu().numpy())

        concat = range(2+self._steps-self._multiplier, self._steps+2)
        genotype = Genotype(
            normal=gene_normal, normal_concat=concat,
            reduce=gene_reduce, reduce_concat=concat
        )
        return genotype
