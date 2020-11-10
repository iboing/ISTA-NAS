import os
import sys
import argparse
import os.path as osp
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
cudnn.enabled = True
import numpy as np
import torch

sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), '../'))

from ista_nas_single import Trainer


def parse():
    parser = argparse.ArgumentParser("cifar-ISTANAS-single")
    parser.add_argument('--data', type=str, default='~/data', help='location of the data corpus')
    parser.add_argument('--batch_size', type=int, default=96, help='batch size')
    parser.add_argument('--proj_dims', type=int, default=7, help='proj dimensions')
    parser.add_argument('--sparseness', type=int, default=2, help='sparseness')
    parser.add_argument('--num_classes', type=int, default=10, help='num_classes')
    parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
    parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
    parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
    parser.add_argument('--gpu', type=str, default='0', help='gpu device id')
    parser.add_argument('--epochs', type=int, default=600, help='num of training epochs')
    parser.add_argument('--init_channels', type=int, default=36, help='num of init channels')
    parser.add_argument('--layers', type=int, default=20, help='total number of layers')
    parser.add_argument('--steps', type=int, default=4, help='total number of layers')
    parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
    parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
    parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
    parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path probability')
    parser.add_argument('--save', type=str, default='EXP', help='experiment name')
    parser.add_argument('--seed', type=int, default=2, help='random seed')
    parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
    parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
    parser.add_argument('--auxiliary', action='store_true', default=False, help='use one-step unrolled validation loss')
    parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='init learning rate')
    parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
    parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
    args = parser.parse_args()

    return args


def main():
    args = parse()
    #os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    trainer = Trainer(args)
    trainer.train()


if __name__ == "__main__":
    main()
