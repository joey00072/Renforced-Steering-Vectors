
import torch
from contextlib import contextmanager   
import torch.nn.functional as F
import torch.nn as nn

class SteeringVectorModel(nn.Module):
    def __init__(self, model, scale=0.001):
        super().__init__()
        self.model = model
        self.device = model.device
        self.scale = scale
        self.hidden_size = model.config.hidden_size
        self.steering_vector = torch.nn.Parameter(torch.zeros(self.hidden_size).to(self.device))
        self.hook_handles = []
        for params in self.model.parameters():
            params.requires_grad = False
        self.apply_steering_vector()

    def modify_activation(self, module, input,*args,**kwargs):
        return (input[0] - self.steering_vector * self.scale,)  # Ensure tuple return

    def apply_steering_vector(self):
        self.hook_handles = []
        for layer in self.model.model.layers:
            handle = layer.register_forward_pre_hook(self.modify_activation)
            self.hook_handles.append(handle)

    def forward(self, **args):
        return self.model(**args)

    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)

    @contextmanager
    def disable_adapter(self):
        for handle in self.hook_handles:
            handle.remove()
            
        yield
        
        self.apply_steering_vector()  