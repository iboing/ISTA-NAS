import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


from .outer_trainer import OuterTrainer
from .models import NetWork
from ..utils import *


__all__ = ["InnerTrainer"]


class InnerTrainer:
    def __init__(self, cfg):
        self.grad_clip = cfg.grad_clip
        self.report_freq = cfg.report_freq
        self.model = NetWork(cfg.init_channels, cfg.num_classes, cfg.layers, proj_dims=cfg.proj_dims).cuda()
        print("Param size = {}MB".format(count_parameters_in_MB(self.model)))

        weights = []
        for k, p in self.model.named_parameters():
            if 'alpha' not in k:
                weights.append(p)
        self.optimizer = optim.SGD(
            weights, cfg.learning_rate,
            momentum=cfg.momentum, weight_decay=cfg.weight_decay
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, float(cfg.epochs), eta_min=cfg.learning_rate_min
        )
        self.outer_trainer = OuterTrainer(self.model, cfg)

    def train_epoch(self, train_queue, valid_queue, epoch):
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        self.scheduler.step()
        lr = self.scheduler.get_lr()
        #arlr = self.outer_trainer.scheduler.get_lr()
        print('epoch: ', epoch, 'lr:', lr)
        valid_loader = iter(valid_queue)

        self.model.train()
        for batch_id, (input, target) in enumerate(train_queue):
            # for inner update
            input = input.cuda()
            target = target.cuda()
            # for outer update
            try:
                input_search, target_search = next(valid_loader)
            except StopIteration:
                valid_loader = iter(valid_queue)
                input_search, target_search = next(valid_loader)

            input_search = input_search.cuda()
            target_search = target_search.cuda()

            self.outer_trainer.step(input_search, target_search)

            self.optimizer.zero_grad()
            scores = self.model(input)
            loss = F.cross_entropy(scores, target)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()

            n = input.size(0)
            prec1, prec5 = accuracy(scores, target, topk=(1, 5))
            losses.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

            if batch_id % self.report_freq == 0:
                print("Train[{:0>3d}] Loss: {:.4f} Top1: {:.4f} Top5: {:.4f}".format(
                    batch_id, losses.avg, top1.avg, top5.avg
                ))
#        self.scheduler.step()
#        self.outer_trainer.scheduler.step()
        return top1.avg, losses.avg

    def validate(self, valid_queue):
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        self.model.eval()

        with torch.no_grad():
            for batch_id, (input, target) in enumerate(valid_queue):
                input = input.cuda()
                target = target.cuda()

                scores = self.model(input)
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
