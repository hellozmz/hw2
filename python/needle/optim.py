"""Optimization module"""
import needle as ndl
import numpy as np


class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay

    def step(self):
        ### BEGIN YOUR SOLUTION
        for param_id, param in enumerate(self.params):
            grad = (self.u.get(param_id, 0) * self.momentum+ (1 - self.momentum) * (param.grad + self.weight_decay *param)).detach()
            grad = ndl.Tensor(grad, dtype=param.dtype, device=param.device)
            self.u[param_id] = grad
            param.data -= self.lr * grad
        ### END YOUR SOLUTION

    def clip_grad_norm(self, max_norm=0.25):
        """
        Clips gradient norm of parameters.
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = {}
        self.v = {}

    def step(self):
        ### BEGIN YOUR SOLUTION
        self.t += 1
        for param_id, param in enumerate(self.params):
            grad = (param.grad + self.weight_decay * param).detach()

            m_curr = (self.beta1 * self.m.get(param_id, 0) + (1 - self.beta1) * grad).detach()     # running avg of grad
            v_curr = (self.beta2 * self.v.get(param_id, 0) + (1 - self.beta2) * grad * grad).detach()   # running avg of grad^2

            self.m[param_id] = m_curr
            self.v[param_id] = v_curr

            m_curr_hat = (m_curr / (1 - (self.beta1 ** self.t))).detach()
            v_curr_hat = (v_curr / (1 - (self.beta2 ** self.t))).detach()

            update = (self.lr * m_curr_hat / (ndl.ops.power_scalar(v_curr_hat, 1/2) + self.eps)).detach()

            param.data -= ndl.Tensor(update, dtype=param.dtype)
        ### END YOUR SOLUTION
