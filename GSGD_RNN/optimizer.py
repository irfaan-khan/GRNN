import torch

class GSGD(torch.optim.Optimizer):
    """
    Custom GSGD optimizer.
    """
    def __init__(self, params, lr=1e-3, alpha=0.99, beta=0.999, weight_decay=1e-4):
        """
        Initialize the GSGD optimizer.

        Args:
            params (iterable): Iterable of parameters to optimize or dicts defining parameter groups.
            lr (float): Learning rate. Default is 1e-3.
            alpha (float): Coefficient used for computing running averages of gradient. Default is 0.99.
            beta (float): Coefficient used for computing running averages of squared gradient. Default is 0.999.
            weight_decay (float): Weight decay (L2 penalty). Default is 1e-4.
        """
        defaults = dict(lr=lr, alpha=alpha, beta=beta, weight_decay=weight_decay)
        super(GSGD, self).__init__(params, defaults)

    def step(self, closure=None):
        """
        Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model and returns the loss.
        """
        for group in self.param_groups:
            for param in group['params']:
                if param.grad is None:
                    continue

                grad = param.grad.data
                state = self.state[param]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['grad_avg'] = torch.zeros_like(param.data)
                    state['squared_grad_avg'] = torch.zeros_like(param.data)

                grad_avg = state['grad_avg']
                squared_grad_avg = state['squared_grad_avg']

                # Update step count
                state['step'] += 1

                # Retrieve hyperparameters from the current parameter group
                alpha = group['alpha']
                beta = group['beta']
                lr = group['lr']
                weight_decay = group['weight_decay']

                # Apply weight decay
                if weight_decay != 0:
                    grad.add_(param.data, alpha=weight_decay)  # Updated add_ method

                # Update the running average of the gradient (v_t)
                grad_avg = alpha * grad_avg + (1 - alpha) * grad
                # Update the running average of the squared gradient (g_t)
                squared_grad_avg = beta * squared_grad_avg + (1 - beta) * grad ** 2

                param.data -= lr * grad_avg / (torch.sqrt(squared_grad_avg) + 1e-8)
