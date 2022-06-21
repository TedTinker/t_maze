#%%

# To do:
#  Stop seeing through walls (done, but do it better)
#  Cut off extra, unused steps (nan instead of 0)

import argparse
from math import pi
import numpy as np

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arena_name',         type=str,   default = "t")    
    parser.add_argument('--boxes_per_cube',     type=int,   default = 2)    
    parser.add_argument('--end_rewards',        type=tuple, default = (1, ((.5, 3),(.5,.5))))    
    parser.add_argument('--reward_scaling',     type=float, default = .999)    
    parser.add_argument('--body_size',          type=float, default = 2)    
    parser.add_argument('--max_steps',          type=int,   default = 100)
    parser.add_argument('--image_size',         type=int,   default = 8)
    parser.add_argument('--min_speed',          type=float, default = 10)
    parser.add_argument('--max_speed',          type=float, default = 50)
    parser.add_argument('--max_yaw_change',     type=float, default = pi/2)
    parser.add_argument('--wall_punishment',    type=float, default = .1)
    
    parser.add_argument('--max_epochs',         type=int,   default = 1000)
    parser.add_argument('--episodes_per_epoch', type=int,   default = 1)
    parser.add_argument('--show_and_save',      type=int,   default = 25)
    parser.add_argument('--too_long',           type=int,   default = 300)
    parser.add_argument('--iterations',         type=int,   default = 16)
    parser.add_argument('--batch_size',         type=int,   default = 32)
    parser.add_argument('--hidden_size',        type=int,   default = 128)
    parser.add_argument('--encode_size',        type=int,   default = 128)
    parser.add_argument('--lstm_size',          type=int,   default = 256)
    
    parser.add_argument('--lr',                 type=float, default = .001) # Learning rate
    parser.add_argument("-alpha",               type=float, default = None) # Soft-Actor-Critic entropy aim
    parser.add_argument("-d",                   type=int,   default = 2)    # Delay to train actors
    parser.add_argument("-eta",                 type=float, default = 5)    # Scale curiosity
    parser.add_argument("-eta_rate",            type=float, default = 1)    # Scale eta
    parser.add_argument("-gamma",               type=float, default = .99)  # For discounting reward
    parser.add_argument("-tau",                 type=float, default = 1e-2) # For soft-updating target critics
    
    args, _ = parser.parse_known_args()
    return args

args = get_args()

def change_args(**kwargs):
    args = get_args()
    for key, value in kwargs.items():
        setattr(args, key, value)
    return(args)


### A few utilities
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE" # Without this, pyplot crashes the kernal



import torch
from torch import nn
from torch.distributions import Normal
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if(device.type == "cpu"):   print("\n\nAAAAAAAUGH! CPU! >:C\n")
else:                       print("\n\nUsing CUDA! :D\n")



def shape_out(layer, shape_in):
    example = torch.zeros(shape_in)
    example = layer(example)
    return(example.shape)

from math import prod
def flatten_shape(shape, num):
    new_shape = tuple(s for i,s in enumerate(shape) if i < num)
    new_shape += (prod(shape[num:]),)
    return(new_shape)

def cat_shape(shape_1, shape_2, dim):
    assert(len(shape_1) == len(shape_2))
    new_shape = ()
    for (s1, s2, d) in zip(shape_1, shape_2, range(len(shape_1))):
        if(d != dim): 
            assert(s1 == s2)
            new_shape += (s1,)
        else:
            new_shape += (s1+s2,)
    return(new_shape)

def reshape_shape(shape, new_shape):
    assert(prod(shape) == prod(new_shape))
    return(new_shape)

class ConstrainedConv2d(nn.Conv2d):
    def forward(self, input):
        return nn.functional.conv2d(input, self.weight.clamp(min=-1.0, max=1.0), self.bias, self.stride,
                                    self.padding, self.dilation, self.groups)
    
def init_weights(m):
    try:
        torch.nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0.01)
    except: pass

            
            
# How to get rolling average.
def get_rolling_average(wins, roll = 100):
    if(len(wins) < roll):
        return(sum(wins)/len(wins))
    return(sum(wins[-roll:])/roll)       


# How to add discount to a list.
def add_discount(rewards, GAMMA = .99):
    d = rewards[-1]
    for i, r in enumerate(rewards[:-1]):
        rewards[i] += d*(GAMMA)**(len(rewards) - i)
    return(rewards)



# Track seconds starting right now. 
import datetime
start_time = datetime.datetime.now()
def reset_start_time():
    global start_time
    start_time = datetime.datetime.now()
def duration():
    change_time = datetime.datetime.now() - start_time
    change_time = change_time - datetime.timedelta(microseconds=change_time.microseconds)
    return(change_time)
  
  

# How to save plots.
import matplotlib.pyplot as plt
import shutil

def remove_folder(folder):
    files = os.listdir("saves")
    if(folder not in files): return
    shutil.rmtree("saves/" + folder)

def make_folder(folder):
    if(folder == None): return
    files = os.listdir("saves")
    if(folder in files): return
    os.mkdir("saves/"+folder)
    os.mkdir("saves/"+folder+"/plots")
    os.mkdir("saves/"+folder+"/agents")
    
def save_plot(name, folder = "default"):
    make_folder(folder)
    plt.savefig("saves/"+folder+"/plots/"+name+".png")
  
def delete_with_name(name, folder = "default", subfolder = "plots"):
    files = os.listdir("saves/{}/{}".format(folder, subfolder))
    for file in files:
        if(file.startswith(name)):
            os.remove("saves/{}/{}/{}".format(folder, subfolder, file))
            


# How to plot an episode's rewards.
def plot_rewards(rewards, name = None, folder = "default"):
    total_length = len(rewards)
    x = [i for i in range(1, total_length + 1)]
    plt.plot(x, [0 for _ in range(total_length)], "--", color = "black", alpha = .5)
    plt.plot(x, rewards, color = "turquoise")
    plt.title("Rewards")
    plt.xlabel("Time")
    plt.ylabel("Reward")
    if(name!=None): save_plot(name, folder)
    plt.show()
    plt.close()
    
# How to plot cumulative rewards.
def plot_cumulative_rewards(rewards, punishments, name = None, folder = "default"):
    total_length = len(rewards)
    x = [i for i in range(1, total_length + 1)]
    rewards = np.cumsum(rewards)
    punishments = np.cumsum(punishments)
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(x, [0 for _ in range(total_length)], "--", color = "black", alpha = .5)
    ax1.plot(x, rewards, color = "turquoise")
    ax2.plot(x, punishments, color = "pink")
    plt.title("Cumulative Rewards and Punishments")
    plt.xlabel("Time")
    ax1.set_ylabel("Rewards")
    ax1.set_ylim(-rewards[-1], rewards[-1]+1)
    ax2.set_ylabel("Punishments")
    ax2.set_ylim(punishments[-1]-1, -punishments[-1])
    if(name!=None): save_plot(name, folder)
    plt.show()
    plt.close()
    
    
# Compare rewards to curiosity.
def plot_curiosity(rewards, curiosity, masks, name = None, folder = "default"):
    fig, ax1 = plt.subplots()    
    ax2 = ax1.twinx()
    for i in range(len(rewards)):
        r = rewards[i].squeeze(-1).tolist()
        r = [r_ for j, r_ in enumerate(r) if masks[i,j] == 1]
        c = curiosity[i].squeeze(-1).tolist()
        c = [c_ for j, c_ in enumerate(c) if masks[i,j] == 1]
        x = [i for i in range(len(r))]
        ax1.plot(x, r, color = "blue", alpha = .5, label = "Reward" if i == 0 else "")
        ax2.plot(x, c, color = "green", alpha = .5, label = "Curiosity" if i == 0 else "")
    plt.title("Value of rewards vs curiosity")
    ax1.set_ylabel("Rewards")
    ax1.legend(loc = 'upper left')
    ax2.set_ylabel("Curiosity")
    ax2.legend(loc = 'lower left')
    plt.show()
    plt.close()
    
            
# How to plot losses.
def get_x_y(losses, too_long):
    x = [i for i in range(len(losses)) if losses[i] != None]
    y = [l for l in losses if l != None]
    if(too_long != None and len(x) > too_long):
        x = x[-too_long:]; y = y[-too_long:]
    return(x, y)

def plot_losses(losses, too_long, d, name = None, folder = "default"):

    trans_losses   = losses[:,0]
    alpha_losses   = losses[:,1]
    actor_losses   = losses[:,2]
    critic1_losses = losses[:,3]
    critic2_losses = losses[:,4]
    
    trans_x, trans_y     = get_x_y(trans_losses, too_long)
    alpha_x, alpha_y     = get_x_y(alpha_losses, too_long)
    actor_x, actor_y     = get_x_y(actor_losses, too_long)
    critic1_x, critic1_y = get_x_y(critic1_losses, None if too_long == None else too_long * d)
    critic2_x, critic2_y = get_x_y(critic2_losses, None if too_long == None else too_long * d)
    
    # Plot trans_loss
    plt.xlabel("Epochs")
    plt.plot(trans_x, trans_y, color = "green", label = "Trans")
    plt.ylabel("Trans losses")
    plt.legend(loc = 'upper left')
    plt.title("Transitioner loss")
    if(name!=None): save_plot(name+"_trans", folder)
    plt.show()
    plt.close()
    
    # Plot losses for actor, critics, and alpha
    fig, ax1 = plt.subplots()
    plt.xlabel("Epochs")

    ax1.plot(actor_x, actor_y, color='red', label = "Actor")
    ax1.set_ylabel("Actor losses")
    ax1.legend(loc = 'upper left')

    ax2 = ax1.twinx()
    ax2.plot(critic1_x, critic1_y, color='blue', linestyle = "--", label = "Critic")
    ax2.plot(critic2_x, critic2_y, color='blue', linestyle = ":", label = "Critic")
    ax2.set_ylabel("Critic losses")
    ax2.legend(loc = 'lower left')
    
    ax3 = ax1.twinx()
    ax3.spines["right"].set_position(("axes", 1.2))
    ax3.plot(alpha_x, alpha_y, color = (0,0,0,.5), label = "Alpha")
    ax3.set_ylabel("Alpha losses")
    ax3.legend(loc = 'upper right')
    
    plt.title("Agent losses")
    fig.tight_layout()
    if(name!=None): save_plot(name+"_agent", folder)
    plt.show()
    plt.close()
    
    
  
# How to plot victory-rates.
def plot_wins(wins, name = None, folder = "default"):
    total_length = len(wins)
    x = [i for i in range(1, len(wins)+1)]
    plt.plot(x, wins, color = "gray")
    plt.ylim([0, 1])
    plt.title("Win-rates")
    plt.xlabel("Episodes")
    plt.ylabel("Win-rate")
    if(name!=None): save_plot(name, folder)
    plt.show()
    plt.close()
    
# How to plot kinds of victory.
def plot_which(which, name = None, folder = "default"):
    total_length = len(which)
    x = [i for i in range(1, len(which)+1)]
    plt.scatter(x, which, color = "gray")
    plt.title("Kind of Win")
    plt.xlabel("Episodes")
    plt.ylabel("Which Victory")
    if(name!=None): save_plot(name, folder)
    plt.show()
    plt.close()

def plot_positions(positions_list, arena_name, agent_name, folder = "default"):
    arena_map = plt.imread("arenas/" + arena_name + ".png")
    arena_map = np.flip(arena_map, 0)    
    h, w, _ = arena_map.shape
    fig, ax = plt.subplots()
    #ax.xaxis.set_visible(False)
    #ax.yaxis.set_visible(False)
    ax.imshow(arena_map, extent=[-.5, w-.5, -h+.5, .5], zorder = 1, origin='lower')
    for positions in positions_list:
        x = [p[1] for p in positions]
        y = [-p[0] for p in positions]
        ax.plot(x, y, zorder = 2)
        ax.scatter(x[-1], y[-1], s = 100, color = "black", marker = "*", zorder = 3)
        ax.scatter(x[-1], y[-1], s = 75, marker = "*", zorder = 4)
    plt.title("Tracks of agent {}".format(agent_name))
    save_plot("tracks_" + agent_name, folder)
    plt.show()
    plt.close()

      
  
  
# How to save/load agent

def save_agent(agent, suf = "", folder = None):
    if(folder == None): return
    if(type(suf) == int): suf = str(suf).zfill(5)
    torch.save(agent.state_dict(), "saves/" + folder + "/agents/agent_{}.pt".format(suf))

def load_agent(agent, suf = "last", folder = "default"):
    if(type(suf) == int): suf = str(suf).zfill(5)
    agent.load_state_dict(torch.load("saves/" + folder + "/agents/agent_{}.pt".format(suf)))
    return(agent)
# %%
