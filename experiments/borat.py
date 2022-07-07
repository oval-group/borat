import torch
import math
import numpy as np
import torch.optim as optim
from torch.optim.optimizer import required

class BORAT(optim.Optimizer):
    def __init__(self, params, model, loss, eta=None, n=1, momentum=0, projection_fn=None, sgd_forward=False, same_batch=False, eps=1e-8, debug=False):
        if eta is not None and eta <= 0.0:
            raise ValueError("Invalid eta: {}".format(eta))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))

        params_list = list(params)
        defaults = dict(eta=eta, momentum=momentum, step_size=None, eps=eps)
        super(BORAT, self).__init__(params_list, defaults)

        self.model = model
        self.obj = loss
        self.projection = projection_fn
        self.print = debug
        self.N = n
        self.zero_plane = True
        self.momentum_forward = False
        self.sgd_forward = sgd_forward
        self.eps = eps
        self.eta_sgd = eta

        if self.sgd_forward:
            self.update = self.sgd_update
        else:
            self.update = self.bundle_update

        if same_batch:
            self.step = self.step_same_batch
        else:
            self.step = self.step_different_batch

        for group in self.param_groups:
            for p in group['params']:
                if group['momentum']:
                    self.state[p]['momentum_buffer'] = torch.zeros_like(p.data, requires_grad=False)
                    if self.momentum_forward:
                        self.state[p]['fast_momentum_buffer'] = torch.zeros_like(p.data, requires_grad=False)

        self.reset_bundle()

        if self.projection is not None:
            self.projection()

        self.bn_stats = {}

    @torch.autograd.no_grad()
    def step_different_batch(self, loss, x, y):
        self.update_bundle(loss())
        if not self.n == self.N:
            if not self.sgd_forward:
                self.solve_bundle()
            self.sample_next_point()
        else:
            self.solve_bundle()
            self.update_parameters()
            if self.projection is not None:
                self.projection()
            self.reset_bundle()

    @torch.autograd.no_grad()
    def step_same_batch(self, loss, x, y):
        # self.save_bn_stats()
        if float(loss()) == 0.0:
            if self.projection is not None:
                self.projection()
            self.reset_bundle()
        else:
            self.update_bundle(loss())
            for _ in range(self.N-1):
                if not self.sgd_forward:
                    self.solve_bundle()
                self.sample_next_point()
                loss = self.forward_backward(x, y)
                self.update_bundle(loss)
            self.solve_bundle()
            self.update_parameters()
            if self.projection is not None:
                self.projection()
            self.reset_bundle()
        # self.load_bn_stats()

    @torch.autograd.no_grad()
    def forward_backward(self, x, y):
        self.model.zero_grad()
        with torch.enable_grad():
            scores = self.model(x)
            loss = self.obj(self.model(x), y)
            loss.backward()
        return float(loss)

    @torch.autograd.no_grad()
    def update_bundle(self, loss):
        self.n += 1
        self.losses.append(float(loss))
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['grads'].append(p.grad.data.detach().clone())
                state['w'].append(p.data.detach().clone())

    @torch.autograd.no_grad()
    def solve_bundle(self):
        self.construct_A_and_b()
        self.generate_combs()
        self.loop_over_combs()

    @torch.autograd.no_grad()
    def reset_bundle(self):
        self.n = 0
        self.losses = []
        for group in self.param_groups:
            for p in group['params']:
                self.state[p]['grads'] = []
                self.state[p]['w'] = []
                if group["momentum"] and self.momentum_forward:
                    self.state[p]['fast_momentum_buffer'] = self.state[p]['momentum_buffer'].clone()

    @torch.autograd.no_grad()
    def sample_next_point(self):
        for group in self.param_groups:
            if group['eta'] > 0.0:
                for p in group['params']:
                    if not self.sgd_forward:
                        p.data.copy_(self.state[p]['w'][0])
                        if group["momentum"]:
                            self.state[p]['fast_momentum_buffer'] = self.state[p]['momentum_buffer'].clone()
                    p.data.add_(self.update(p, group))
                    if group["momentum"] and self.momentum_forward:
                        self.apply_momentum_fast(p, group)

    @torch.autograd.no_grad()
    def construct_A_and_b(self):

        loop_range = len(self.losses)
        self.n = len(self.losses)
        if self.zero_plane:
            self.n += 1

        self.A = torch.ones(self.n + 1, self.n + 1, device='cuda')
        self.A[0:-1,0:-1] = 0
        self.A += self.eps * torch.eye(self.n + 1, device='cuda')
        self.A[-1,-1] = 0

        self.b = torch.zeros(self.n + 1, device='cuda')
        self.b[-1] = 1


        for i in range(loop_range):
            self.b[i] += self.losses[i]

        for group in self.param_groups:
            eta = group['eta']
            for p in group['params']:
                for i in range(loop_range):
                    g_i = self.state[p]['grads'][i]
                    w_s = self.state[p]['w']
                    self.b[i] -= g_i.mul(w_s[i]-w_s[0]).sum()
                    for j in range(loop_range):
                        g_j = self.state[p]['grads'][j]
                        self.A[i, j] += eta * (g_i * g_j).sum()

        '''
        self.A2 = torch.ones(self.n + 1, self.n + 1, device='cuda')
        self.A2[0:-1,0:-1] = 0
        self.A2 += self.eps * torch.eye(self.n + 1, device='cuda')
        self.A2[-1,-1] = 0
        for group in self.param_groups:
            eta = group['eta']
            for p in group['params']:
                for i in range(loop_range):
                    g_i = self.state[p]['grads'][i]
                    w_s = self.state[p]['w']
                    self.b[i] -= g_i.mul(w_s[i]-w_s[0]).sum()
                    for j in range(i,loop_range):
                        g_j = self.state[p]['grads'][j]
                        if (i==j):
                            self.A[i, j] += eta * (g_i * g_j).sum()
                        else:
                            self.A2[j, j] += eta * (g_i * g_j).sum()
                            self.A2[i, j] += eta * (g_i * g_j).sum()
        print('norm', torch.norm(self.A-self.A2))
        input('press any key')
        '''


        self.alig_step = (self.b[0]/(self.A[0,0]+self.eps))
        if self.print:
            print('A')
            print(self.A)
            print('b')
            print(self.b)

    @torch.autograd.no_grad()
    def generate_combs(self):
        # we're creating the binary representation for all numbers from 0 to N-1
        N = 2**self.n
        a = np.arange(N, dtype=int).reshape(1,-1)
        l = int(np.log2(N))
        b = np.arange(l, dtype=int)[::-1,np.newaxis]
        self.combinations = torch.Tensor(np.array(a & 2**b > 0, dtype=int)).bool().cuda()
        self.combinations = self.combinations[:,1:]
        # add line adding columns of combs and removing columns thats sum to one
        self.num_combs = self.combinations.size()[1]
        if self.print:
            print('self.combinations')
            print(self.combinations)

    @torch.autograd.no_grad()
    def loop_over_combs(self):
        self.best_alphas = torch.zeros(self.n, device='cuda')
        self.max_dual_value = -1e9
        for i in range(self.num_combs):
            active_idxs = self.combinations[:,i]
            A, b = self.sub_A_and_b(active_idxs)
            # solve the linear system
            this_alpha = A.inverse().mv(b)
            this_alpha = this_alpha[:-1]

            # check if valid solution 
            if (this_alpha >= 0).all():
                alpha = torch.zeros(self.n, device='cuda')
                alpha[active_idxs] = this_alpha
                this_dual_value = self.dual(alpha)

                if self.print:
                    print('--------------------------------------')
                    print('A')
                    print(A)
                    print('b')
                    print(b)
                    print('solution to linear system')
                    print(this_alpha)
                    print('dual')
                    print(this_dual_value)
                    print('--------------------------------------')

                if this_dual_value > self.max_dual_value:
                    self.max_dual_value = this_dual_value
                    self.best_alpha = alpha

    @torch.autograd.no_grad()
    def update_diagnostics(self):
        alpha = self.best_alpha
        self.step_size = self.alig_step.clamp(min=0,max=self.eta_sgd)
        self.step_0 = alpha[0]
        if len(alpha) > 1:
            self.step_size_unclipped = self.alig_step
            self.step_1 = alpha[1]
        if len(alpha) > 2:
            self.step_2 = alpha[2]
        if len(alpha) > 3:
            self.step_3 = alpha[3]
        if len(alpha) > 4:
            self.step_4 = alpha[4]

        if self.print:
            print('------------------------')
            print('best dual')
            print(self.max_dual_value)
            print('best alpha')
            print(self.best_alpha)
            print('alig step')
            print(self.alig_step)
            print('------------------------')
            input('press any key')

    @torch.autograd.no_grad()
    def sub_A_and_b(self, chosen_idx):
        extra_0 = torch.tensor([1]).bool().cuda()
        idxs = torch.cat((chosen_idx,extra_0),0)
        A_rows = self.A[idxs, :]
        this_A = A_rows[:, idxs]
        this_b = self.b[idxs]
        return this_A, this_b

    @torch.autograd.no_grad()
    def dual(self, alpha):
        A = self.A[:-1,:-1]
        b = self.b[:-1]
        return - 0.5 * alpha.mul(A.mv(alpha)).sum() + b.mul(alpha).sum()

    @torch.autograd.no_grad()
    def update_parameters(self):
        # update parameters of model
        for group in self.param_groups:
            if group['eta'] > 0.0:
                for p in group['params']:
                    p.data.copy_(self.state[p]['w'][0])
                    p.data.add_(self.bundle_update(p, group))
                    if group["momentum"]:
                        self.apply_momentum_nesterov(p, group)
        self.update_diagnostics()

    @torch.autograd.no_grad()
    def bundle_update(self, p, group):
        update = 0
        for i, grad in enumerate(self.state[p]['grads']):
            update += self.best_alpha[i] * grad
        return - group['eta'] * update

    @torch.autograd.no_grad()
    def sgd_update(self, p, group):
        return - self.eta_sgd * p.grad.data

    @torch.autograd.no_grad()
    def apply_momentum_nesterov(self, p, group):
        buffer = self.state[p]['momentum_buffer']
        momentum = group["momentum"]
        buffer.mul_(momentum).add_(self.bundle_update(p, group))
        p.data.add_(momentum, buffer)

    @torch.autograd.no_grad()
    def apply_momentum_fast(self, p, group):
        buffer = self.state[p]['fast_momentum_buffer']
        momentum = group["momentum"]
        buffer.mul_(momentum).add_(self.update(p, group))
        p.data.add_(momentum, buffer)

    @torch.autograd.no_grad()
    def save_bn_stats(self):
        if len(self.bn_stats) == 0:
            for module in self.model.modules():
                if isinstance(module, torch.nn.BatchNorm2d):
                    self.bn_stats[module] = {}
        for module in self.model.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                stats = self.bn_stats[module]
                stats['running_mean'] = module.running_mean.data.clone()
                stats['running_var'] = module.running_var.data.clone()
                stats['num_batches_tracked'] = module.num_batches_tracked.data.clone()

    @torch.autograd.no_grad()
    def load_bn_stats(self):
        for module in self.model.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                stats = self.bn_stats[module]
                module.running_mean.data.copy_(stats['running_mean'])
                module.running_var.data.copy_(stats['running_var'])
                module.num_batches_tracked.data.copy_(stats['num_batches_tracked'])
