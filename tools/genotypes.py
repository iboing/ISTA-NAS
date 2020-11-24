from collections import namedtuple


__all__ = ["Genotype", "PRIMITIVES", "ISTA_twostage", "ISTA_onestage", "ISTA_ImageNet"]


Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

PRIMITIVES = [
#    'none',
    'max_pool_3x3',
    'avg_pool_3x3',
    'skip_connect',
    'sep_conv_3x3',
    'sep_conv_5x5',
    'dil_conv_3x3',
    'dil_conv_5x5'
]

ISTA_twostage = Genotype(normal=[('skip_connect', 0), ('dil_conv_3x3', 1), ('sep_conv_5x5', 0), ('dil_conv_3x3', 2), ('dil_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2), ('skip_connect', 0)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('sep_conv_5x5', 1), ('dil_conv_5x5', 2), ('max_pool_3x3', 1), ('dil_conv_5x5', 3), ('sep_conv_3x3', 1), ('dil_conv_5x5', 4), ('sep_conv_5x5', 1)], reduce_concat=range(2, 6))

ISTA_onestage = Genotype(normal=[('skip_connect', 0), ('dil_conv_3x3', 1), ('sep_conv_3x3', 1), ('dil_conv_5x5', 2), ('avg_pool_3x3', 3), ('sep_conv_3x3', 1), ('sep_conv_5x5', 1), ('sep_conv_3x3', 2)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 0), ('max_pool_3x3', 1), ('sep_conv_5x5', 1), ('avg_pool_3x3', 0), ('max_pool_3x3', 0), ('dil_conv_5x5', 2)], reduce_concat=range(2, 6))

ISTA_ImageNet = Genotype(normal=[('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 2), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('dil_conv_5x5', 4)], normal_concat=range(2, 6), reduce=[('skip_connect', 0), ('sep_conv_3x3', 1), ('dil_conv_5x5', 2), ('sep_conv_3x3', 0), ('sep_conv_3x3', 0), ('skip_connect', 2), ('sep_conv_3x3', 2), ('skip_connect', 3)], reduce_concat=range(2, 6))
