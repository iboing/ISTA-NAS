import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from .optimizer import Adam
from .models import NetWork
from ..utils import *


__all__ = ["InnerTrainer"]


class InnerTrainer:
    def __init__(self, cfg):
        self.auxiliary = cfg.auxiliary
        self.auxiliary_weight = cfg.auxiliary_weight

        self.grad_clip = cfg.grad_clip
        self.report_freq = cfg.report_freq
        self.model = NetWork(cfg.init_channels, cfg.num_classes, cfg.layers, proj_dims=cfg.proj_dims, auxiliary=cfg.auxiliary).cuda()
        print("Param size = {}MB".format(count_parameters_in_MB(self.model)))

        weights = []
        for k, p in self.model.named_parameters():
            if 'alpha' not in k:
                weights.append(p)
        self.w_optimizer = optim.SGD(
            weights, cfg.learning_rate,
            momentum=cfg.momentum, weight_decay=cfg.weight_decay
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.w_optimizer, float(cfg.epochs), eta_min=cfg.learning_rate_min
        )
        #self.alpha_optimizer = torch.optim.Adam(self.model.arch_parameters(),
        self.alpha_optimizer = Adam(self.model.arch_parameters(),
            lr=cfg.arch_learning_rate, betas=(0.5, 0.999), weight_decay=cfg.arch_weight_decay)
#        self.criterion = nn.CrossEntropyLoss().cuda()

    def train_epoch(self, train_queue, epoch, all_freeze):
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        lr = self.scheduler.get_lr()
        print('epoch: ', epoch, 'lr:', lr)
        # valid_loader = iter(valid_queue)

        self.model.train()
        for batch_id, (input, target) in enumerate(train_queue):
            # for inner update
            input = input.cuda()
            target = target.cuda()

#            self.model.zero_grad()
            self.alpha_optimizer.zero_grad()
            self.w_optimizer.zero_grad()
            scores, scores_aux = self.model(input, all_freeze)
            loss = F.cross_entropy(scores, target)
            if self.auxiliary:
                loss_aux = F.cross_entropy(scores_aux, target)
                loss += self.auxiliary_weight * loss_aux
            loss.backward() #retain_graph=True
            self.alpha_optimizer.step()
#            self.w_optimizer.zero_grad()
#            scores, scores_aux = self.model(input)
#            loss = F.cross_entropy(scores, target)
#            if self.auxiliary:
#                loss_aux = F.cross_entropy(scores_aux, target)
#                loss += self.auxiliary_weight * loss_aux
#            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            #self.alpha_optimizer.step()
            self.w_optimizer.step()

            n = input.size(0)
            prec1, prec5 = accuracy(scores, target, topk=(1, 5))
            losses.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

            if batch_id % self.report_freq == 0:
                print("Train[{:0>3d}] Loss: {:.4f} Top1: {:.4f} Top5: {:.4f}".format(
                    batch_id, losses.avg, top1.avg, top5.avg
                ))
        if all_freeze:
            self.scheduler.step()
        return top1.avg, losses.avg

    def validate(self, valid_queue, all_freeze):
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        self.model.eval()

        with torch.no_grad():
            for batch_id, (input, target) in enumerate(valid_queue):
                input = input.cuda()
                target = target.cuda()

                scores, _ = self.model(input, all_freeze)
                loss = F.cross_entropy(scores, target)

                n = input.size(0)
                prec1, prec5 = accuracy(scores, target, topk=(1, 5))
                losses.update(loss.item(), n)
                top1.update(prec1.item(), n)
                top5.update(prec5.item(), n)

                if batch_id % self.report_freq == 0:
                    print(" Valid[{:0>3d}] Loss: {:.4f} Top1: {:.4f} Top5: {:.4f}".format(
                        batch_id, losses.avg, top1.avg, top5.avg
                    ))

        return top1.avg, losses.avg
