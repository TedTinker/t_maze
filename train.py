#%%

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import torch
import numpy as np
from tqdm import tqdm
from tkinter import Tk
from copy import deepcopy
from time import sleep

from utils import device, args, save_agent, load_agent, get_rolling_average,reset_start_time, duration, \
    remove_folder, make_folder, delete_with_name, plot_rewards, plot_losses, plot_wins, plot_extrinsic_intrinsic, plot_which, plot_cumulative_rewards, plot_positions
from env import Env
from agent import Agent

def episode(env, agent, push = True, delay = False):
    env.reset()  
    done = False
    positions = []
    with torch.no_grad():
        while(done == False):
            done, win, which, pos = env.step(agent)
            positions.append(pos)
            if(delay): sleep(.5)
            torch.cuda.synchronize(device=device)
    env.body.to_push.finalize_rewards()
    rewards = deepcopy(env.body.to_push.rew)
    if(push): env.body.to_push.push(agent.memory)
    else:     env.body.to_push.empty()
    env.close()
    return(win, which, rewards, positions)



env_gui = Env(args, GUI = True)



q = False
tk= Tk()
tk.geometry("200x200")
def q_press(e):
    global q
    if(not q and e.char == "q"): 
        print("\nStarting GUI!\n")
        q = True
    if(q and e.char == "w"): 
        print("\nStopping GUI!\n")
        q = False

tk.bind('<KeyPress>',q_press)



class Trainer():
    def __init__(
            self, my_args = args, delete = True,
            save_folder = "default", 
            load_folder = None, 
            load_name = "last"):
        
        self.args = my_args
        self.delete = delete
        self.save_folder = save_folder
        self.load_folder = load_folder; self.load_name = load_name
        
        self.env = Env(self.args, GUI = False)
        self.restart()

    def get_GUI(self):
        global env_gui
        env_gui.change(self.args, True)
        return(env_gui)
    
    def restart(self):
        reset_start_time()
        if(self.delete):
            print("\nDELETING OLD FILES\n")
            remove_folder(self.save_folder)
            make_folder(self.save_folder)
        self.e = 0
        self.agent = Agent(my_args = self.args)
        if(self.load_folder != None):
            self.agent = load_agent(
                self.agent, suf = self.load_name, folder = self.load_folder)
        else:
            save_agent(self.agent, suf = str(self.e), folder = self.save_folder)
        self.wins = []; self.wins_rolled = []; self.which = []
        self.extrinsics = []; self.intrinsic_curiosities = []; self.intrinsic_entropies = []
        self.rewards = []; self.punishments = []
        self.losses = np.array([[None]*5])
        
    def one_episode(self, push = True, GUI = False, delay = False):     
        tk.update()   
        if(GUI == False): GUI = q
        if(GUI): env = self.get_GUI()
        else:    env = self.env
        win, which, rewards, positions = \
            episode(env, self.agent, push, delay)
        tk.update()   
        if(q): plot_rewards(rewards)
        return(int(win), which, rewards, positions)

    def epoch(self, plot = False):
        for _ in range(self.args.episodes_per_epoch):
            win, which, rewards, _ = self.one_episode()
            self.wins.append(win)
            self.wins_rolled.append(get_rolling_average(self.wins))
            self.which.append(which)
            rewards = sum(rewards)
            if(rewards > 0): self.rewards.append(rewards); self.punishments.append(0)
            else:            self.punishments.append(rewards); self.rewards.append(0)
                    
        losses, extrinsic, intrinsic_curiosity, intrinsic_entropy = \
            self.agent.learn(batch_size = self.args.batch_size, iterations = self.args.iterations, plot = plot)
        self.losses = np.concatenate([self.losses, losses])
        self.extrinsics.append(extrinsic)
        self.intrinsic_curiosities.append(intrinsic_curiosity)
        self.intrinsic_entropies.append(intrinsic_entropy)

        if(self.args.iterations == 1):  losses = np.expand_dims(losses,0)

        tk.update()   
        if(q): plot_losses(self.losses, too_long = self.args.too_long, d = self.args.d)

    def train(self):
        
        self.agent.train()
        prb = tqdm(range(self.args.max_epochs))
        for e in prb:
            self.e += 1
            self.epoch(plot = self.e % 25 == 0)
            if(self.e % self.args.show_and_save == 0): 
                save_agent(self.agent, suf = self.e, folder = self.save_folder)
                plot_wins(self.wins_rolled, name = "wins_{}".format(str(self.e).zfill(5)), folder = self.save_folder)
                plot_which(self.which, name = "which_{}".format(str(self.e).zfill(5)), folder = self.save_folder)                
                plot_cumulative_rewards(self.rewards, self.punishments)
                plot_extrinsic_intrinsic(self.extrinsics, self.intrinsic_curiosities, self.intrinsic_entropies)
                plot_losses(self.losses, too_long = self.args.too_long, d = self.args.d)
                
            if(self.e >= self.args.max_epochs):
                print("\n\nFinished!\n\n")
                save_agent(self.agent, suf = "last", folder = self.save_folder)
                delete_with_name("wins", folder = self.save_folder, subfolder = "plots")
                delete_with_name("which", folder = self.save_folder, subfolder = "plots")
                plot_wins(self.wins_rolled, name = "wins_last", folder = self.save_folder)
                plot_which(self.which, name = "which_last", folder = self.save_folder)
                plot_cumulative_rewards(self.rewards, self.punishments, name = "cumulative_rewards", folder = self.save_folder)
                plot_extrinsic_intrinsic(self.extrinsics, self.intrinsic_curiosities, self.intrinsic_entropies, name = "extrinsic_intrinsic", folder = self.save_folder)
                plot_losses(self.losses, too_long = None, d = self.args.d, name = "losses", folder = self.save_folder)
                break
    
    def test(self, size = 100):
        self.agent.eval()
        wins = 0
        for i in tqdm(range(size)):
            w, which, rewards, positions = self.one_episode(push = False, GUI = True, delay = False)
            wins += w
        print("Agent wins {} out of {} games ({}%).".format(wins, size, round(100*(wins/size))))
        
    def get_positions(self, size = 5, agent_name = "last"):
        self.agent.eval()
        positions_list = []
        for i in tqdm(range(size)):
            w, which, rewards, positions = self.one_episode(push = False)
            positions_list.append(positions)
        plot_positions(positions_list, self.args.arena_name)
