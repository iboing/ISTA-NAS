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
        self.train_queue, self.test_queue = self.set_dataloader()

    def set_dataloader(self):
        kwargs = {"num_workers": 2, "pin_memory": True}

        train_transform, valid_transform = cifar10_transforms(self.cfg)
        train_data = datasets.CIFAR10(
            root=self.cfg.data, train=True, download=True, transform=train_transform)
        test_data = datasets.CIFAR10(
            root=self.cfg.data, train=False, download=True, transform=valid_transform)

        train_queue = data.DataLoader(
            train_data, batch_size=self.cfg.batch_size,
            # sampler=data.sampler.SubsetRandomSampler(indices[:split]),
            shuffle=True, **kwargs)

        test_queue = data.DataLoader(
            test_data, batch_size=self.cfg.batch_size,
            shuffle=False, **kwargs)

        return train_queue, test_queue

    def do_recovery(self, As, alpha, x_last=None, freeze_flag=None):
        xs = []
        for i in range(self.steps):
            if freeze_flag is not None and freeze_flag[i]:
                xs.append(x_last[i])
                continue
            lasso = LASSO(As[i].cpu().numpy().copy())
            b = alpha[i]
            x = lasso.solve(b)
            xs.append(x)

        return xs

    def do_search(self, A_normal, normal_biases, normal_freeze_flag,
                       A_reduce, reduce_biases, reduce_freeze_flag, epoch, all_freeze):
        if not all_freeze:
            self.search_trainer.model.init_proj_mat(A_normal, A_reduce)
            self.search_trainer.model.freeze_alpha(normal_freeze_flag, reduce_freeze_flag)
            self.search_trainer.model.init_bias(normal_biases, reduce_biases)
        # train
        train_acc, train_obj = self.search_trainer.train_epoch(
            self.train_queue, epoch, all_freeze)
        logging.info("train_acc {:.4f}".format(train_acc))
        #print(self.search_trainer.model.alphas_normal)
        #print(self.search_trainer.model.alphas_reduce)
        # valid
        valid_acc, valid_obj = self.search_trainer.validate(self.test_queue, all_freeze)
        logging.info("valid_acc {:.4f}".format(valid_acc))

        if not all_freeze:
            alphas = self.search_trainer.model.arch_parameters()
            alpha_normal = torch.stack(alphas[:self.steps]).detach().cpu().numpy()
            alpha_reduce = torch.stack(alphas[self.steps:]).detach().cpu().numpy()
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

    def show_selected(self, epoch, x_normals_last, x_reduces_last,
                                   x_normals_new, x_reduces_new):
        print("\n[Epoch {}]".format(epoch if epoch > 0 else 'initial'))
        # print("x_normals:\n", x_normals)
        # print("x_reduces:\n", x_reduces)
        print("x_normals distance:")
        normal_freeze_flag = []
        reduce_freeze_flag = []
        for i, (x_n_b, x_n_a) in enumerate(zip(x_normals_last, x_normals_new)):
            dist = np.linalg.norm(x_n_a - x_n_b, 2)
            normal_freeze_flag.append(False if epoch == 0 else dist <= 1e-3)
            print("Step {}: L2 dist is {}. {}".format(i+1, dist,
                            "freeze!!!" if normal_freeze_flag[-1] else "active"))
        print("x_reduces distance:")
        for i, (x_r_b, x_r_a) in enumerate(zip(x_reduces_last, x_reduces_new)):
            dist = np.linalg.norm(x_r_a - x_r_b, 2)
            reduce_freeze_flag.append(False if epoch == 0 else dist <= 1e-3)
            print("Step {}: L2 dist is {}. {}".format(i+1, dist,
                            "freeze!!!" if reduce_freeze_flag[-1] else "active"))

        print("normal cell:")
        gene_normal = []
        for i, x in enumerate(x_normals_new):
            id1, id2 = np.abs(x).argsort()[-2:]
            print("Step {}: node{} op{}, node{} op{}".format(
                i + 1, id1 // self.num_ops,
                       id1 % self.num_ops,
                       id2 // self.num_ops,
                       id2 % self.num_ops))
            gene_normal.append((PRIMITIVES[id1 % self.num_ops], id1 // self.num_ops))
            gene_normal.append((PRIMITIVES[id2 % self.num_ops], id2 // self.num_ops))

        print("reduction cell:")
        gene_reduce = []
        for i, x in enumerate(x_reduces_new):
            id1, id2 = np.abs(x).argsort()[-2:]
            print("Step {}: node{} op{}, node{} op{}".format(
                i + 1, id1 // self.num_ops,
                       id1 % self.num_ops,
                       id2 // self.num_ops,
                       id2 % self.num_ops))
            gene_reduce.append((PRIMITIVES[id1 % self.num_ops], id1 // self.num_ops))
            gene_reduce.append((PRIMITIVES[id2 % self.num_ops], id2 // self.num_ops))

        concat = range(2, 2 + len(x_normals_new))
        genotype = Genotype(
            normal = gene_normal, normal_concat = concat,
            reduce = gene_reduce, reduce_concat = concat)
        print(genotype)
        model_cifar = NetworkCIFAR(36, 10, 20, True, genotype)
        param_size = count_parameters_in_MB(model_cifar)
        logging.info('param size = %fMB', param_size)

        return normal_freeze_flag, reduce_freeze_flag, param_size

    def train(self):
        base_A_normals = []
        base_A_reduces = []

        for i in range(self.steps):
            base_A_normals.append(
                torch.from_numpy(np.random.rand(self.proj_dims, (i+2) * self.num_ops)))
            base_A_reduces.append(
                torch.from_numpy(np.random.rand(self.proj_dims, (i+2) * self.num_ops)))

        alpha_normal = torch.stack(self.search_trainer.model.alphas_normal_).detach().cpu().numpy()
        alpha_reduce = torch.stack(self.search_trainer.model.alphas_reduce_).detach().cpu().numpy()
        x_normals_new = self.do_recovery(base_A_normals, alpha_normal)
        x_reduces_new = self.do_recovery(base_A_reduces, alpha_reduce)

        x_normals_last = x_normals_new.copy()
        x_reduces_last = x_reduces_new.copy()

        normal_freeze_flag, reduce_freeze_flag, _ = self.show_selected(
            0, x_normals_last, x_reduces_last, x_normals_new, x_reduces_new)


        for i in range(self.cfg.epochs):
            A_normals, normal_biases = self.sample_and_proj(
                base_A_normals, x_normals_last)
            A_reduces, reduce_biases = self.sample_and_proj(
                base_A_reduces, x_reduces_last)
            print("\nDoing Search ...")
            self.search_trainer.model.drop_path_prob = 0 #self.cfg.drop_path_prob * i / self.cfg.epochs
            alpha_normal, alpha_reduce = self.do_search(
                A_normals, normal_biases, normal_freeze_flag,
                A_reduces, reduce_biases, reduce_freeze_flag, i+1, False)
            if False not in normal_freeze_flag and False not in reduce_freeze_flag:
                break
            print("Doing Recovery ...")
            x_normals_new = self.do_recovery(base_A_normals, alpha_normal,
                    x_normals_last, normal_freeze_flag)
            x_reduces_new = self.do_recovery(base_A_reduces, alpha_reduce,
                    x_reduces_last, reduce_freeze_flag)
            ## update freeze flag
            normal_freeze_flag, reduce_freeze_flag, param_size = self.show_selected(
                i+1, x_normals_last, x_reduces_last, x_normals_new, x_reduces_new)
            if param_size >= 3.7: # large model may cause out of memory !!!
                print('-------------> rejected !!!')
                continue
            x_normals_last = x_normals_new
            x_reduces_last = x_reduces_new

        print("\n --- Architecture Fixed, Retrain for {} Epochs --- \n".format(self.cfg.epochs))
        for i in range(self.cfg.epochs):
            self.search_trainer.model.drop_path_prob = self.cfg.drop_path_prob * i / self.cfg.epochs
            self.do_search(
                A_normals, normal_biases, normal_freeze_flag,
                A_reduces, reduce_biases, reduce_freeze_flag, i+1, True)
