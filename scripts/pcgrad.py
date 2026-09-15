"""
PCGrad: Projecting Conflicting Gradients for Multi-Task Learning
Reference:
    Yu et al., "Gradient Surgery for Multi-Task Learning", NeurIPS 2020.
    https://arxiv.org/abs/2001.06782
"""

import torch
import random
import copy


class PCGrad:
    def __init__(self, optimizer, reduction='mean'):
        self._optim = optimizer
        self._reduction = reduction

    @property
    def optimizer(self):
        return self._optim

    def zero_grad(self):
        return self._optim.zero_grad(set_to_none=True)

    def step(self):
        return self._optim.step()

    def pc_backward(self, objectives):
        """
        Calculates gradient projection for multi-task loss objectives.
        
        Args:
            objectives: list of scalar PyTorch loss tensors, e.g. [loss_bcs, loss_behavior, loss_lameness, loss_id]
        """
        grads, shapes, has_grads = self._pack_grad(objectives)
        pc_grad = self._project_conflicting(grads, has_grads)
        pc_grad = self._unflatten_grad(pc_grad, shapes[0])
        self._set_grad(pc_grad)

    def _project_conflicting(self, grads, has_grads):
        shared = torch.stack(grads)
        num_tasks = len(grads)

        # Shuffle task order to ensure unbiased projection
        task_order = list(range(num_tasks))
        random.shuffle(task_order)

        for i in task_order:
            for j in task_order:
                if i == j:
                    continue
                # Inner product of gradients for task i and task j
                g_i = shared[i]
                g_j = shared[j]
                inner_product = torch.dot(g_i, g_j)

                # If cosine similarity < 0 (conflicting gradients), project g_i onto normal plane of g_j
                if inner_product < 0:
                    shared[i] -= (inner_product / (g_j.norm() ** 2 + 1e-8)) * g_j

        if self._reduction == 'mean':
            return shared.mean(dim=0)
        elif self._reduction == 'sum':
            return shared.sum(dim=0)
        else:
            raise ValueError(f"Unknown reduction: {self._reduction}")

    def _pack_grad(self, objectives):
        grads, shapes, has_grads = [], [], []
        for obj in objectives:
            self._optim.zero_grad(set_to_none=True)
            obj.backward(retain_graph=True)
            grad, shape, has_grad = self._retrieve_grad()
            grads.append(self._flatten_grad(grad, shape))
            has_grads.append(self._flatten_grad(has_grad, shape))
            shapes.append(shape)
        return grads, shapes, has_grads

    def _unflatten_grad(self, grads, shapes):
        unflatten_grad, idx = [], 0
        for shape in shapes:
            length = torch.prod(torch.tensor(shape)).item()
            unflatten_grad.append(grads[idx:idx + length].view(shape).clone())
            idx += length
        return unflatten_grad

    def _flatten_grad(self, grads, shapes):
        flatten_grad = torch.cat([g.flatten() for g in grads])
        return flatten_grad

    def _retrieve_grad(self):
        grad, shape, has_grad = [], [], []
        for group in self._optim.param_groups:
            for p in group['params']:
                if p.grad is None:
                    shape.append(p.shape)
                    grad.append(torch.zeros_like(p, device=p.device))
                    has_grad.append(torch.zeros_like(p, device=p.device))
                    continue
                shape.append(p.grad.shape)
                grad.append(p.grad.clone())
                has_grad.append(torch.ones_like(p, device=p.device))
        return grad, shape, has_grad

    def _set_grad(self, grads):
        idx = 0
        for group in self._optim.param_groups:
            for p in group['params']:
                p.grad = grads[idx]
                idx += 1
