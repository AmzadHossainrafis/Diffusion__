import torch
import numpy as np
from PIL import Image


class Noise_scheduler:
    def __init__(self, max_value, min_value, T, warmup, steps):
        self.max_value = max_value
        self.min_value = min_value
        self.T = T
        self.warmup = warmup
        self.steps = steps

    def step(self, t):
        if t < self.warmup:
            return self.min_value
        elif t > self.T:
            return self.min_value
        else:
            return self.min_value + 0.5 * (self.max_value - self.min_value) * (1 + np.cos(np.pi * (t - self.warmup) / (self.T - self.warmup)))

    def __call__(self, t):
        return self.step(t)
    

class linear_noise_scheduler:
    def __init__(self,time_steps=1000, beta_str=0.1, beta_end=1.0,device='cuda'):
        self.time_steps = time_steps
        self.beta_str = beta_str
        self.beta_end = beta_end
        self.device = device

        self.liner_schedule = self.liner_schedule()
        self.alpha = 1 - (1 / self.time_steps)
        self.alpha_hat = self.alpha ** 0.5

        


    def liner_schedule(self):
        return torch.linspace(self.beta_str, self.beta_end, self.time_steps, device=self.device)
    
    def reandom_time_step(self):
        return np.random.randint(0,self.time_steps)
    
    def noise_image(self, img, time_step):
        noise = torch.randn_like(img, device=self.device)
        return img + self.liner_schedule()[time_step] * noise   
    def __call__(self):
        return self.liner_schedule()
    
class cosine_scheduler:
    def __init__(self, max_value, min_value, T, warmup, steps):
        self.max_value = max_value
        self.min_value = min_value
        self.T = T
        self.warmup = warmup
        self.steps = steps

    def step(self, t):
        if t < self.warmup:
            return self.min_value
        elif t > self.T:
            return self.min_value
        else:
            return self.min_value + 0.5 * (self.max_value - self.min_value) * (1 + np.cos(np.pi * (t - self.warmup) / (self.T - self.warmup)))

    def __call__(self, t):
        return self.step(t)