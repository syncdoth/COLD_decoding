import torch
from torch.optim.lr_scheduler import LambdaLR

class StepWiseNoiseScheduler:
    def __init__(self,
                 size,
                 step_noise_iters,
                 step_noise_stds,
                 final_std=0.01):
        if isinstance(step_noise_iters, str):
            step_noise_iters = [int(i) for i in step_noise_iters.split(',')]
        if isinstance(step_noise_stds, str):
            step_noise_stds = [float(j) for j in step_noise_stds.split(',')]
        self.step_noise_iters = step_noise_iters
        self.step_noise_stds = step_noise_stds

        self.size = size
        self.current_step = 0
        self.final_std = final_std

    def step(self, arg):
        del arg  # unused
        self.current_step += 1
        if self.current_step > self.step_noise_iters[-1]:
            noise_std = self.final_std
        else:
            for it, std in zip(self.step_noise_iters, self.step_noise_stds):
                if self.current_step <= it:
                    noise_std = std
            return torch.normal(mean=0,
                                std=noise_std,
                                size=self.size,
                                requires_grad=False)

class GeometricNoiseScheduler:
    def __init__(self,
                 size,
                 start=5,
                 end=0.05,
                 total_steps=250,
                 anneal_noise_step=100,
                 schedule='geometric',
                 ):
        self.size = size
        self.start = start
        self.end = end
        self.total_steps = total_steps
        self.anneal_noise_step = anneal_noise_step
        self.schedule = schedule

        self.variance = start
        self.current_step = 0

        self.gamma = self.compute_geometric_gamma(start, end, anneal_noise_step)

    def step(self, step_size):
        self.current_step += 1
        if self.current_step <= self.anneal_noise_step:
            self.variance *= self.gamma

        std = torch.sqrt(2 * step_size * self.variance)
        return torch.normal(mean=0,
                            std=std,
                            size=self.size,
                            requires_grad=False)


    @staticmethod
    def compute_geometric_gamma(start, end, steps):
        return torch.exp(torch.log(torch.tensor(end / start)) / steps)



def get_constant_schedule_with_warmup(optimizer, init_lr, max_lr, num_warmup_steps, num_training_steps, last_epoch=-1):
    """
    Create a schedule with a learning rate that decreases linearly from the initial lr set in the optimizer to 0, after
    a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return 1 - float(max_lr - init_lr) * float(num_warmup_steps - current_step) / float(max(1, num_warmup_steps * max_lr))
        # return 1
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)
