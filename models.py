#%%
import torch
from torch import nn 
import torch.nn.functional as F
from torch.distributions import Normal
from torchgan import layers as gnn
from torchinfo import summary as torch_summary

import numpy as np
from math import degrees
from random import choice
import matplotlib.pyplot as plt

from utils import args, device, ConstrainedConv2d, delete_these, \
    init_weights, shape_out, flatten_shape, reshape_shape, cat_shape



class Transitioner(nn.Module):

    def __init__(self):
        super(Transitioner, self).__init__()

        self.image_in = nn.Sequential(
            ConstrainedConv2d(
                in_channels = 4, 
                out_channels = 16,
                kernel_size = (3,3),
                padding = (1,1),
                padding_mode="reflect"),
            nn.LeakyReLU(),
            nn.MaxPool2d(
                kernel_size = (3,3), 
                stride = (2,2),
                padding = (1,1)))

        self.speed_in = nn.Sequential(
            nn.Linear(1, args.hidden_size),
            nn.LeakyReLU())

        shape = (1, 4, args.image_size, args.image_size)
        next_shape = shape_out(self.image_in, shape)
        next_shape = flatten_shape(next_shape, 1)

        self.lstm = nn.LSTM(
            input_size = next_shape[-1] + args.hidden_size,
            hidden_size = args.lstm_size,
            batch_first = True)

        self.encode = nn.Sequential(
            nn.LeakyReLU(),
            nn.Linear(args.lstm_size, args.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(args.hidden_size, args.encode_size),
            nn.LeakyReLU())

        self.action_in = nn.Sequential(
            nn.Linear(2, args.hidden_size),
            nn.LeakyReLU())

        self.next_image_1 = nn.Sequential(
            nn.Linear(args.encode_size + args.hidden_size, args.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(args.hidden_size, 32 * args.image_size//4 * args.image_size//4),
            nn.LeakyReLU()) 

        self.next_image_2 = nn.Sequential(
            ConstrainedConv2d(
                in_channels = 32, 
                out_channels = 32,
                kernel_size = (3,3),
                padding = (1,1),
                padding_mode="reflect"),
            nn.LeakyReLU(),
            nn.Upsample(
                scale_factor = 2,
                mode = "bilinear", align_corners=True),
            ConstrainedConv2d(
                in_channels = 32, 
                out_channels = 16,
                kernel_size = (3,3),
                padding = (1,1),
                padding_mode="reflect"),
            nn.LeakyReLU(),
            nn.Upsample(
                scale_factor = 2,
                mode = "bilinear", align_corners=True),
            ConstrainedConv2d(
                in_channels = 16, 
                out_channels = 4,
                kernel_size = (1,1)),
            nn.Tanh()) 

        self.next_speed = nn.Sequential(
            nn.Linear(args.encode_size + args.hidden_size, 1)) 

        self.image_in.apply(init_weights)
        self.speed_in.apply(init_weights)
        self.lstm.apply(init_weights)
        self.encode.apply(init_weights)
        self.action_in.apply(init_weights)
        self.next_image_1.apply(init_weights)
        self.next_image_2.apply(init_weights)
        self.next_speed.apply(init_weights)
        self.to(device)

    def just_encode(self, image, speed, hidden = None):
        image = image.to(device); speed = speed.to(device)
        if(len(image.shape) == 4):  sequence = False
        else:                       sequence = True
        image = image.permute((0,1,-1,2,3) if sequence else (0, -1, 1, 2))
        batch_size = image.shape[0]
        if(sequence): image = image.reshape(image.shape[0]*image.shape[1], image.shape[2], image.shape[3], image.shape[4])
        image = self.image_in(image).flatten(1)
        if(sequence): image = image.reshape(batch_size, image.shape[0]//batch_size, image.shape[1])
        speed = (speed - args.min_speed) / (args.max_speed - args.min_speed)
        speed = (speed*2)-1
        speed = self.speed_in(speed.float())
        x = torch.cat([image, speed], -1)
        if(not sequence): x = x.view(x.shape[0], 1, x.shape[1])
        self.lstm.flatten_parameters()
        if(hidden == None): x, hidden = self.lstm(x)
        else:               x, hidden = self.lstm(x, (hidden[0], hidden[1]))
        if(not sequence): x = x.view(x.shape[0], x.shape[-1])
        encoding = self.encode(x)
        delete_these(False, image, speed, x)
        return(encoding, hidden) 

    def forward(self, image, speed, action):
        action = action.to(device)
        encoding, _ = self.just_encode(image, speed)
        action = self.action_in(action)
        x = torch.cat((encoding, action), dim=-1)
        next_image = self.next_image_1(x)
        batch_size = next_image.shape[0]
        next_image = next_image.reshape(next_image.shape[0]*next_image.shape[1], 32, args.image_size//4, args.image_size//4)
        next_image = self.next_image_2(next_image)   
        next_image = next_image.reshape(batch_size, next_image.shape[0]//batch_size, 4, args.image_size, args.image_size)
        next_image = next_image.permute(0, 1, 3, 4, 2)
        next_image = torch.clamp(next_image, -1, 1)
        next_speed = self.next_speed(x)
        delete_these(False, x, action)
        return(next_image, next_speed)

    def DKL(self, images, speeds, action, next_images, next_speeds, masks):
        with torch.no_grad(): pred_next_images, pred_next_speeds = self(images.detach(), speeds.detach(), action.detach())
        predictions = torch.cat([pred_next_images.flatten(2), pred_next_speeds], dim = -1)
        targets = torch.cat([next_images.flatten(2), next_speeds], dim = -1)
        divergence = F.kl_div(
            F.log_softmax(predictions * masks, dim=-1), 
            F.log_softmax(targets * masks, dim=-1), 
            reduction="none", log_target=True)
        divergence = sum([divergence[:,:,i] for i in range(divergence.shape[-1])])
        return(divergence.unsqueeze(-1))



class Actor(nn.Module):

    def __init__(
            self, 
            log_std_min=-20, 
            log_std_max=2):

        super(Actor, self).__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.lin = nn.Sequential(
            nn.Linear(args.encode_size, args.hidden_size*2),
            nn.LeakyReLU(),
            nn.Linear(args.hidden_size*2, args.hidden_size*2),
            nn.LeakyReLU())
        self.mu = nn.Linear(args.hidden_size*2, 2)
        self.log_std_linear = nn.Linear(args.hidden_size*2, 2)

        self.lin.apply(init_weights)
        self.mu.apply(init_weights)
        self.log_std_linear.apply(init_weights)
        self.to(device)

    def forward(self, encode):
        x = self.lin(encode)
        mu = self.mu(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        delete_these(False, encode)
        return mu, log_std

    def evaluate(self, encode, epsilon=1e-6):
        mu, log_std = self.forward(encode)
        std = log_std.exp()
        dist = Normal(0, 1)
        e = dist.sample().to(device)
        action = torch.tanh(mu + e * std)
        log_prob = Normal(mu, std).log_prob(mu + e * std) - \
            torch.log(1 - action.pow(2) + epsilon)
        return action, log_prob

    def get_action(self, encode):
        mu, log_std = self.forward(encode)
        std = log_std.exp()
        dist = Normal(0, 1)
        e      = dist.sample().to(device)
        action = torch.tanh(mu + e * std).cpu()
        return action[0]



class Critic(nn.Module):

    def __init__(self):

        super(Critic, self).__init__()
        self.lin = nn.Sequential(
            nn.Linear(args.encode_size+2, args.hidden_size*2),
            nn.LeakyReLU(),
            nn.Linear(args.hidden_size*2, args.hidden_size*2),
            nn.LeakyReLU(),
            nn.Linear(args.hidden_size*2, 2))

        self.lin.apply(init_weights)
        self.to(device)

    def forward(self, encode, action):
        x = torch.cat((encode, action), dim=-1)
        x = self.lin(x)
        delete_these(False, encode, action)
        return x



if __name__ == "__main__":

    transitioner = Transitioner()
    actor = Actor()
    critic = Critic()

    print("\n\n")
    print(transitioner)
    print()
    print(torch_summary(transitioner, 
                        ((1, 1, args.image_size, args.image_size, 4), # Image
                         (1, 1, 1),         # Speed
                         (1, 1, 2))))       # Action

    print("\n\n")
    print(actor)
    print()
    print(torch_summary(actor, (1, args.encode_size)))

    print("\n\n")
    print(critic)
    print()
    print(torch_summary(critic, ((1, args.encode_size),(1,2))))
# %%