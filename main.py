#%%

import os

from utils import change_args, plot_positions
from train import Trainer

trainer_dict = {    
    
	"t_none" : lambda test, num, load_name = "last":
    	Trainer(
        	change_args(eta = 0, alpha = 0, target_entropy = 0),
        	delete = not test,
        	save_folder = "t_none_{}".format(str(num).zfill(3)),
        	load_folder = "t_none_{}".format(str(num).zfill(3)) if test else None,
        	load_name = load_name),
        
	"t_entropy" : lambda test, num, load_name = "last":
    	Trainer(
        	change_args(eta = 0),
        	delete = not test,
        	save_folder = "t_entropy_{}".format(str(num).zfill(3)),
        	load_folder = "t_entropy_{}".format(str(num).zfill(3)) if test else None,
        	load_name = load_name),
        
	"t_curious" : lambda test, num, load_name = "last":
    	Trainer(
        	change_args(alpha = 0, target_entropy = 0),
        	delete = not test,
        	save_folder = "t_curious_{}".format(str(num).zfill(3)),
        	load_folder = "t_curious_{}".format(str(num).zfill(3)) if test else None,
        	load_name = load_name)}

def train(trainer_name, num):
	trainer = trainer_dict[trainer_name](False, num)
	trainer.train()
	trainer.env.close(forever=True)
 
def test(trainer_name, num):
	trainer = trainer_dict[trainer_name](True, num)
	trainer.test()
	trainer.env.close(forever=True)
    
def positions(trainer_name, num, load_name, size = 5):
	trainer = trainer_dict[trainer_name[:-4]](True, num, load_name)
	positions_list, arena_name = trainer.get_positions(size = size)
	return(positions_list, arena_name)
 
agent_variety = 5
    
# %%
for i in range(agent_variety):
	train("t_none", i)
 
#%%
folders = []
f = os.listdir("saves")
for folder in f:
    if(folder[:-4] == "t_none"):
        folders.append(folder)
folders.sort()
 
load_names = os.listdir("saves/" + folders[0] + "/agents")
load_names.sort()

for load_name in load_names:
    load_name = load_name[6:-3]
    positions_lists = []
    for i, folder in enumerate(folders):
        print(folder, load_name)
        positions_list, arena_name = positions(folder, i, load_name, 10)
        positions_lists.append(positions_list)
    plot_positions(positions_lists, arena_name, load_name, folder = "t_none_positions")

#%%
for i in range(agent_variety):
	train("t_entropy", i)
 
#%%
folders = []
f = os.listdir("saves")
for folder in f:
    if(folder[:-4] == "t_entropy"):
        folders.append(folder)
folders.sort()
 
load_names = os.listdir("saves/" + folders[0] + "/agents")
load_names.sort()
 
for load_name in load_names:
    load_name = load_name[6:-3]
    positions_lists = []
    for i, folder in enumerate(folders):
        print(folder, load_name)
        positions_list, arena_name = positions(folder, i, load_name, 10)
        positions_lists.append(positions_list)
    plot_positions(positions_lists, arena_name, load_name, folder = "t_entropy_positions")
    
#%%
for i in range(agent_variety):
	train("t_curious", i)
 
#%%
folders = []
f = os.listdir("saves")
for folder in f:
    if(folder[:-4] == "t_curious"):
        folders.append(folder)
folders.sort()
 
load_names = os.listdir("saves/" + folders[0] + "/agents")
load_names.sort()
 
for load_name in load_names:
    load_name = load_name[6:-3]
    positions_lists = []
    for i, folder in enumerate(folders):
        print(folder, load_name)
        positions_list, arena_name = positions(folder, i, load_name, 10)
        positions_lists.append(positions_list)
    plot_positions(positions_lists, arena_name, load_name, folder = "t_curious_positions")