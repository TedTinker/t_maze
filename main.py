#%%

import os

from utils import args, change_args
from train import Trainer

trainer_dict = {    
    "t_non_curious" : lambda test, agent_name = "last": 
        Trainer(
            change_args(eta = 0, alpha = 0), 
            delete = not test,
            save_folder = "t_non_curious",
            load_folder = "t_non_curious" if test else None,
            load_name = agent_name),
    
    "t_curious" : lambda test, agent_name = "last": 
        Trainer(
            change_args(eta = 5, eta_rate = .99999), 
            delete = not test,
            save_folder = "t_curious",
            load_folder = "t_curious" if test else None,
            load_name = agent_name)
    }

def train(trainer_name):
    trainer = trainer_dict[trainer_name](False)
    trainer.train()
    trainer.env.close(forever=True)

def test(trainer_name):
    trainer = trainer_dict[trainer_name](True)
    trainer.test()
    trainer.env.close(forever=True)
    
def positions(trainer_name, agent_name, size = 5):
    trainer = trainer_dict[trainer_name](True, agent_name)
    trainer.get_positions(size = size, agent_name = agent_name)
    trainer.env.close(forever=True)
    
# %%
train("t_non_curious")

#%%
agent_names = os.listdir("saves/t_non_curious/agents")
agent_names.sort()
for agent_name in agent_names:
    agent_name = agent_name[6:-3]
    print(agent_name)
    positions("t_non_curious", agent_name, 10)

# %%

test("t_non_curious")

# %%

train("t_curious")

#%%
agent_names = os.listdir("saves/t_curious/agents")
agent_names.sort()
for agent_name in agent_names:
    agent_name = agent_name[6:-3]
    print(agent_name)
    positions("t_curious", agent_name, 10)

# %%

test("t_curious")
# %%
