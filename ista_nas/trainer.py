import sys
import logging

import torch
import numpy as np
import torchvision.datasets as datasets
import torch.utils.data as data

from .search import *
from .recovery import *
from .utils import *


__all__ = ["Trainer"]


log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')


class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg

        self.num_ops      = len(PRIMITIVES)
        self.proj_dims    = cfg.proj_dims
        self.sparseness   = cfg.sparseness
        self.steps        = cfg.steps

        self.search_trainer = InnerTrainer(cfg)
        self.num_edges = self.search_trainer.model.num_edges
        self.train_queue, self.valid_queue = self.set_dataloader()

    def set_dataloader(self):
        train_transform, valid_transform = cifar10_transforms(self.cfg)
        train_data = datasets.CIFAR10(
            root=self.cfg.data, train=True, download=True, transform=train_transform)

        num_train = len(train_data)
        indices = list(range(num_train))
        split = int(np.floor(self.cfg.train_portion * num_train))

        train_queue = data.DataLoader(
            train_data, batch_size=self.cfg.batch_size,
            sampler=data.sampler.SubsetRandomSampler(indices[:split]),
            pin_memory=True, num_workers=2)

        valid_queue = data.DataLoader(
            train_data, batch_size=self.cfg.batch_size,
            sampler=data.sampler.SubsetRandomSampler(indices[split:]),
            pin_memory=True, num_workers=2)

        return train_queue, valid_queue

    def do_recovery(self, As, alpha):
        xs = []
        for i in range(self.steps):
            lasso = LASSO(As[i].cpu().numpy().copy())
            b = alpha[i]
            x = lasso.solve(b)
            xs.append(x)

        return xs

    def do_search(self, A_normal, normal_biases,
                       A_reduce, reduce_biases, epoch):
        self.search_trainer.model.init_proj_mat(A_normal, A_reduce)
        self.search_trainer.model.init_bias(normal_biases, reduce_biases)
        # train
        train_acc, train_obj = self.search_trainer.train_epoch(
            self.train_queue, self.valid_queue, epoch)
        logging.info("train_acc {:.4f}".format(train_acc))
        # valid
        valid_acc, valid_obj = self.search_trainer.validate(self.valid_queue)
        logging.info("valid_acc {:.4f}".format(valid_acc))

        alpha_normal, alpha_reduce = self.search_trainer.model.arch_parameters()
        alpha_normal = alpha_normal.detach().cpu().numpy()
        alpha_reduce = alpha_reduce.detach().cpu().numpy()

        return alpha_normal, alpha_reduce

    def sample_and_proj(self, base_As, xs):
        As= []
        biases = []
        for i in range(self.steps):
            A = base_As[i].numpy().copy()
            E = A.T.dot(A) - np.eye(A.shape[1])
            x = xs[i].copy()
            zero_idx = np.abs(x).argsort()[:-self.sparseness]
            x[zero_idx] = 0.
            A[:, zero_idx] = 0.
            As.append(torch.from_numpy(A).float())
            E[:, zero_idx] = 0.
            bias = E.T.dot(x).reshape(-1, self.num_ops)
            biases.append(torch.from_numpy(bias).float())

        biases = torch.cat(biases)

        return As, biases

    def show_selected(self, epoch, x_normals, x_reduces):
        print("[Epoch {}]".format(epoch if epoch > 0 else 'initial'))

        print("normal cell:")
        gene_normal = []
        for i, x in enumerate(x_normals):
            id1, id2 = np.abs(x).argsort()[-2:]
            print("Step {}: edge{} op{}, edge{} op{}".format(
                i + 1, id1 // self.num_ops + 1,
                       id1 % self.num_ops + 1,
                       id2 // self.num_ops + 1,
                       id2 % self.num_ops + 1))
            gene_normal.append((PRIMITIVES[id1 % self.num_ops], id1 // self.num_ops))
            gene_normal.append((PRIMITIVES[id2 % self.num_ops], id2 // self.num_ops))

        print("reduction cell:")
        gene_reduce = []
        for i, x in enumerate(x_reduces):
            id1, id2 = np.abs(x).argsort()[-2:]
            print("Step {}: edge{} op{}, edge{} op{}".format(
                i + 1, id1 // self.num_ops + 1,
                       id1 % self.num_ops + 1,
                       id2 // self.num_ops + 1,
                       id2 % self.num_ops + 1))
            gene_reduce.append((PRIMITIVES[id1 % self.num_ops], id1 // self.num_ops))
            gene_reduce.append((PRIMITIVES[id2 % self.num_ops], id2 // self.num_ops))

        concat = range(2, 2 + len(x_normals))
        genotype = Genotype(
            normal = gene_normal, normal_concat = concat,
            reduce = gene_reduce, reduce_concat = concat)
        print(genotype)

    def train(self):
        #
        base_A_normals = []
        base_A_reduces = []

        for i in range(self.steps):
            base_A_normals.append(
                torch.from_numpy(np.random.rand(self.proj_dims, (i+2) * self.num_ops)))
            base_A_reduces.append(
                torch.from_numpy(np.random.rand(self.proj_dims, (i+2) * self.num_ops)))

        alpha_normal = self.search_trainer.model.alphas_normal_.detach().cpu().numpy()
        alpha_reduce = self.search_trainer.model.alphas_reduce_.detach().cpu().numpy()
        x_normals = self.do_recovery(base_A_normals, alpha_normal)
        x_reduces = self.do_recovery(base_A_reduces, alpha_reduce)

        self.show_selected(0, x_normals, x_reduces)

        for i in range(self.cfg.epochs):
            A_normals, normal_biases = self.sample_and_proj(
                base_A_normals, x_normals)
            A_reduces, reduce_biases = self.sample_and_proj(
                base_A_reduces, x_reduces)
            print("Doing Search ...")
            alpha_normal, alpha_reduce = self.do_search(A_normals, normal_biases,
                                                       A_reduces, reduce_biases, i+1)
            print("Doing Recovery ...")
            x_normals = self.do_recovery(base_A_normals, alpha_normal)
            x_reduces = self.do_recovery(base_A_reduces, alpha_reduce)
            self.show_selected(i+1, x_normals, x_reduces)
