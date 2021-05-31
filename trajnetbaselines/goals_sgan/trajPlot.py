"Plotting of goals and trajectories"

import matplotlib.pyplot as plt
import torch
import numpy as np

import trajnetplusplustools
from ..lstm.data_load_utils import prepare_data

from .goals import goalModel



## Global variables
sganModelPath = ''
goalModelPath = 'OUTPUT_BLOCK/synth_data/goalsModel.pkl'
dataPath = 'DATA_BLOCK/synth_data'
obs_length = 9


## Load model
with open(goalModelPath, 'rb') as f:
    goal_Model = torch.load(f)
    

## Load scene
scenes = prepare_data(dataPath, subset='/train/')
filename, scene_id, paths = scenes[0][0] #taking first scene
scene = trajnetplusplustools.Reader.paths_to_xy(paths)
scene = np.array(scene)
# scene [length, actors, xy]


## Predict goal
batch_split = torch.Tensor([0, scene.shape[1]])
goals = goal_Model(torch.Tensor(scene), batch_split)
goals = goals.detach().numpy()


## Plot
for i in range(scene.shape[1]):
    if i==0:
        plt.plot(scene[:obs_length, i, 0], scene[:obs_length, i, 1], 'r-')
        plt.plot(scene[obs_length-1:, i, 0], scene[obs_length-1:, i, 1], 'r--')
    else:
        plt.plot(scene[:obs_length, i, 0], scene[:obs_length, i, 1], 'k-')
        plt.plot(scene[obs_length-1:, i, 0], scene[obs_length-1:, i, 1], 'k--')
        
for i in range(goals.shape[1]):
    plt.scatter(goals[0, i, 0], goals[0, i, 1], c='r', marker='*')        

plt.show()




