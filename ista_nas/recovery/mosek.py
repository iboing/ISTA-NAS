import torch
import cvxpy as cvx
import numpy as np


__all__ = ["LASSO"]


class LASSO:
    def __init__(self, A):
        m, n = A.shape
        self.A = A
        self.m = m
        self.n = n

    def construct_prob(self, b):
        gamma = cvx.Parameter(nonneg=True, value=1e-5)
        x = cvx.Variable(self.n)
        error = cvx.sum_squares(self.A * x - b)
        obj = cvx.Minimize(0.5 * error + gamma * cvx.norm(x, 1))
        prob = cvx.Problem(obj)

        return prob, x

    def solve(self, b):
        prob, x = self.construct_prob(b)
        prob.solve(solver=cvx.MOSEK)
        x_res = np.array(x.value).reshape(-1)
        return x_res
