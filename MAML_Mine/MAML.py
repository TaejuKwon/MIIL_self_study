import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from collections import OrderedDict
from src.tasks import Sine_Task, Sine_Task_Distribution

import numpy as np
import matplotlib.pyplot as plt

class MAMLModel(nn.Module):
    def __init__(self):
        super(MAMLModel, self).__init__()
        self.model = nn.Sequential(OrderedDict([
            ('l1', nn.Linear(1,40)),
            ('relu1', nn.ReLU()),
            ('l2', nn.Linear(40,40)),
            ('relu2', nn.ReLU()),
            ('l3', nn.Linear(40,1))
        ]))
        
    def forward(self, x):
        return self.model(x)
    
    def parameterised(self, x, weights):

        x = nn.functional.linear(x, weights[0], weights[1])                                                                                                                         
        x = nn.functional.relu(x)
        x = nn.functional.linear(x, weights[2], weights[3])
        x = nn.functional.relu(x)
        x = nn.functional.linear(x, weights[4], weights[5])
        return x

class MAML():
    def __init__(self, model, tasks, inner_lr, meta_lr, K = 10, inner_steps = 1, tasks_per_meta_batch = 1000):

        self.tasks = tasks
        self.model = model
        self.weights = list(model.parameters())
        self.len_weights = len(list(model.parameters()))
        self.criterion = nn.MSELoss()
        self.meta_optimizer = torch.optim.SGD(self.weights, meta_lr)

        # hyperparameters
        self.inner_lr = inner_lr
        self.meta_lr = meta_lr
        self.K = K
        self.inner_steps = inner_steps
        self.task_per_meta_batch = tasks_per_meta_batch

        self.plot_every = 10
        self.print_every = 10
        self.meta_losses = []

    def inner_loop(self, task):

        temp_weights = [w.clone() for w in self.weights]

        X, y = task.sample_data(self.K)
        for step in range(self.inner_steps):
            loss = self.criterion(self.model.parameterised(X, temp_weights), y) / self.K

            grad = torch.autograd.grad(loss, temp_weights)
            temp_weights = [w - self.inner_lr * g for w, g in zip(temp_weights, grad)]

        # updating loss
        loss = self.criterion(self.model.parameterised(X, temp_weights), y) / self.K

        return loss
    
    def outer_loop(self, num_iterations):
        epoch_loss = 0

        for iteration in range(1, num_iterations+1):
            # meta loss
            meta_loss = 0
            for i in range(self.task_per_meta_batch):
                task = self.tasks.sample_task()
                meta_loss += self.inner_loop(task)
            
            meta_grads = torch.autograd.grad(meta_loss, self.weights)

            for w, g in zip(self.weights, meta_grads):
                w.grad = g
            self.meta_optimizer.step()

            epoch_loss += meta_loss.item() / self.task_per_meta_batch

            if iteration % self.print_every == 0:
                print("{}/{}. loss: {}".format(iteration, num_iterations, epoch_loss / self.plot_every))
            if iteration % self.plot_every == 0:
                self.meta_losses.append(epoch_loss / self.plot_every)
                epoch_loss = 0

tasks = Sine_Task_Distribution(0.1, 5, 0, np.pi, -5, 5)
maml = MAML(MAMLModel(), tasks, inner_lr= 0.01, meta_lr= 0.001)
print(maml.model)
maml.outer_loop(num_iterations=50)

# plt.plot(maml.meta_losses)