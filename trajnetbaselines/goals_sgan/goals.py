import torch
import torch.nn as nn
import os
import pickle
import numpy as np
import trajnetplusplustools

from ..lstm.modules import InputEmbedding
from ..lstm.utils import center_scene
from .. import augmentation

class goalModel(torch.nn.Module):
    """ Model that learns predicting the goal destination of actors. As we are using multimodal SGAN, we also need multimodal goals.
    During training, the ground truth can be used to calculate the loss. """
    def __init__(self, emb_dim=32, in_dim=2, hid_dim=64, out_dim=2, k=3):
        """Initialization 
        
        Parameters
        ----------
        k: integer
            Number of modes 
        
        """
            
        super(goalModel, self).__init__()
        # TODO: Write this class with all necessary functions
        
        # parameters 
        self.emb_dim = emb_dim
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.k = k       
        
        # layers
        self.input_embedding = InputEmbedding(in_dim, emb_dim, 4.0)
        self.encoder = nn.LSTMCell(emb_dim, hid_dim)
        self.decoder = nn.LSTMCell(emb_dim, hid_dim)
        self.linear = nn.Linear(hid_dim, out_dim*k)
         
    def step(self, lstm, hidden_cell_state, obs):
        # consider only the hidden states of pedestrains present in scene
        track_mask = (torch.isnan(obs[:,0]) == 0)
        
        # masked hidden cell state
        hidden_cell_stacked = [
            torch.stack([h for m, h in zip(track_mask, hidden_cell_state[0]) if m], dim=0),
            torch.stack([h for m, h in zip(track_mask, hidden_cell_state[1]) if m], dim=0),
        ]
        
        # mask embed
        input_emb = self.input_embedding(obs[track_mask])
        
        # forward
        hidden_cell_stacked = lstm(input_emb, hidden_cell_stacked)
        
        # unmask
        mask_index = [i for i, m in enumerate(track_mask) if m]
        hidden_cell = (hidden_cell_state[0].clone(), hidden_cell_state[1].clone())
        for i, h, c in zip(mask_index, hidden_cell_stacked[0], hidden_cell_stacked[1]):
            hidden_cell[0][i] = h
            hidden_cell[1][i] = c
        
        return hidden_cell

    def forward(self, batch_scene, obs_len=9):
        """ Forward pass, we ignore the inner relation of a scene, take num_tracks as batch size.
        
        Parameters
        ----------
        batch_scene: Tensor (seq_len, num_tracks, out_dim=2)
            Tensor of batch of scenes
        
        Return
        ------
        output: Tensor (num_tracks, k, out_dim=2)
        """
        # take the observations as input 
        observations = batch_scene[:obs_len] # (obs_len, num_tracks, 2)
#         if observations.isnan().sum().item() != 0:
#             print("{} nan in observations".format(observations.isnan().sum().item()))
        
        _, num_tracks, num_coor = observations.shape
        hidden_cell_state = (torch.zeros((num_tracks, self.hid_dim)), 
                             torch.zeros((num_tracks, self.hid_dim)))
        # encoder
        for t in range(obs_len):
            hidden_cell_state = self.step(self.encoder, hidden_cell_state, observations[t])
            
        # decoder
        go = torch.zeros((num_tracks, num_coor))
        hidden_cell_state = self.step(self.decoder, hidden_cell_state, go)
        
        output = self.linear(hidden_cell_state[0])
        return output.reshape(-1, self.k, self.out_dim)

    def save(self, state, filename):
        with open(filename, 'wb') as f:
            torch.save(self, f)

        # # during development, good for compatibility across API changes:
        # # Save state for optimizer to continue training in future
        with open(filename + '.state', 'wb') as f:
            torch.save(state, f)

    @staticmethod
    def load(filename):
        with open(filename, 'rb') as f:
            return torch.load(f)


def get_goals(scene, obs_length, pred_length):
    """ Given a scene, extract the goal from each actor. 
    
    Parameters
    ----------
    scene: Tensor [time_steps, num_actors, 2]
    
    Returns
    -------
    goal: Tensor [n_actors, 2]
    """
    
    goal = scene[-1,:,:]
    return goal



def prepare_goals_data(path, subset='/train/', sample=1.0):
    """ Prepares the train/val scenes and corresponding goals 
    
    Parameters
    ----------
    subset: String ['/train/', '/val/']
        Determines the subset of data to be processed
    sample: Float (0.0, 1.0]
        Determines the ratio of data to be sampled
    goals: Bool
        If true, the goals of each track are extracted
        The corresponding goal file must be present in the 'goal_files' folder
        The name of the goal file must be the same as the name of the training file

    Returns
    -------
    all_scenes: List
        List of all processed scenes
    all_goals: Dictionary
        Dictionary of goals corresponding to each dataset file.
        None if 'goals' argument is False.
    Flag: Bool
        True if the corresponding folder exists else False.
    """

    ## Check if folder exists
    if not os.path.isdir(path + subset):
        if 'train' in subset:
            print("Train folder does NOT exist")
            exit()
        if 'val' in subset:
            print("Validation folder does NOT exist")
            return None, False

    ## read goal files
    all_scenes = []

    ## List file names
    files = [f.split('.')[-2] for f in os.listdir(path + subset) if f.endswith('.ndjson')]
    ## Iterate over file names
    for file in files:
        reader = trajnetplusplustools.Reader(path + subset + file + '.ndjson', scene_type='paths')
        ## Necessary modification of train scene to add filename
        scene = [(file, s_id, s) for s_id, s in reader.scenes(sample=sample)]
        
        all_scenes += scene

    return all_scenes, True


class goalLoss(torch.nn.Module):
    def __init__(self, keep_batch_dim=False):
        super(goalLoss, self).__init__()
        self.keep_batch_dim = keep_batch_dim
        
    def forward(self, goal_pred, goal_gt):
        """"Forward function calculating the loss.
        
        Parameters
        ----------
        goal_pred: Tensor [num_tracks, k, 2]
            Contains the k predicted goals per scene
        goal_gt: Tensor [num_tracks, 2]
            Containts the goal ground truth
        
        Returns
        -------
        loss: Tensor [1,]
            L2-norm variety loss
        """
        # mask true goals
        track_mask = (torch.isnan(goal_gt[:,0]) == 0)
        
        loss = self.L2_variety_loss(goal_pred[track_mask], 
                                    goal_gt[track_mask])
        
        if self.keep_batch_dim:
            return loss
        else:
            return loss.sum()
        
    def L2_variety_loss(self, pred, gt):
        L2 = self.L2norm(pred, gt)
        loss, _ = torch.min(L2, dim=1)
        return loss

    def L2norm(self, pred, gt):
        """Calculation of L2-norm of inputs.
        
        Parameters
        ----------
        pred: Tensor [batch_size, modes, dim]
            Contains predictions
        gt: Tensor [batch_size, dim]
            Containts the ground truth
        
        Returns
        -------
        loss: Tensor [batch_size, modes]
            L2 norm, modewise
        """
        
        gt = gt[:,None,:]
        L2 = (pred - gt).norm(p=2, dim=2)
        assert L2.shape == pred.shape[:-1], "Size missmatch"
        return L2
    
    
def interpolate_batch_scene(batch_scene):
    """Find NaN's and replace them by interpolation."""
        
    if not np.isnan(batch_scene).any():
        return batch_scene # all good with this batch
    else:
        mask = np.isnan(batch_scene)
        ind = np.vstack(np.where(mask)).T #index of nans
        seq_length = batch_scene.shape[0]
        for i in ind:
            if (i[0] != 0) & (i[0] != seq_length-1):
                ## Interpolate
                prev = batch_scene[i[0]-1, i[1], i[2]]
                next = batch_scene[i[0]+1, i[1], i[2]]
                batch_scene[i[0], i[1], i[2]] = prev + (next-prev)/2
            else:
                ## Extrapolate
                if i[0] == 0:
                    next = batch_scene[1, i[1], i[2]]
                    nextnext = batch_scene[2, i[1], i[2]]
                    batch_scene[i[0], i[1], i[2]] = next - (nextnext - next)
                else: # last one is nan
                    prev = batch_scene[i[0]-1, i[1], i[2]]
                    prevprev = batch_scene[i[0]-2, i[1], i[2]]
                    batch_scene[i[0], i[1], i[2]] = prev - (prevprev - prev)
        return batch_scene
    

class goalSGANPredictor(object):
    def __init__(self, goalModel, SGANModel):
        self.goalModel = goalModel
        self.SGANModel = SGANModel
        
    def save(self, state, filename):
        with open(filename, 'wb') as f:
            torch.save(state, f)
            
        with open(filename + '.state', 'wb') as f:
            torch.save(state, f)
            
    @staticmethod
    def load(filename):
        with open(filename, 'rb') as f:
            return torch.load(f)
        
    def __call__(self, paths, n_predict=12, SGAN_modes=1, predict_all=True, obs_length=9, start_length=0, args=None):
        self.goalModel.eval()
        self.SGANModel.eval()
        
        if SGAN_modes is not None:
            self.SGANModel.k = SGAN_modes
            
        with torch.no_grad():
            xy = trajnetplusplustools.Reader.paths_to_xy(paths)
            batch_split = [0, xy.shape[1]]
            
        if args.normalize_scene:
                xy, rotation, center, _ = center_scene(xy, obs_length)   
                
        xy = torch.Tensor(xy)  #.to(self.device)
        batch_split = torch.Tensor(batch_split).long()
                
        ## Goal predictions
        goals = self.goalModel(batch_scene=xy)
        
        
        ## Trajectory predictions
        multimodal_outputs = {}
        for i in range(goals.shape[1]): #Iterating over predicted goals
            goal = goals[:,i,:]
            
            _, output_scenes, _, _ = self.SGANModel(xy[:obs_length], goal, batch_split, n_predict=n_predict)
            output_scenes = output_scenes.numpy()
            
            if args.normalize_scene:
                output_scenes = augmentation.inverse_scene(output_scenes, rotation, center)
            
            output_primary = output_scenes[-n_predict:, 0]
            output_neighs = output_scenes[-n_predict:, 1:]
            
            if i == 0:
                multimodal_outputs[i] = [output_primary, output_neighs]
            else:
                multimodal_outputs[i] = [output_primary, []]
        ## Return Dictionary of predictions. Each key corresponds to one mode
        return multimodal_outputs
                      
                
if __name__ == '__main__':

    print('create data')
    batch_scene = torch.empty(21,40,2).uniform_(0,1)
    batch_scene[-1,1,:] = np.nan
    print(torch.isnan(batch_scene).sum().item())
    
    batch_split = torch.Tensor([0,5,9,15,20,22,30,33,40])
    targets = torch.ones(40,2)

    print('create model')
    model = goalModel()

    print('forward pass')
    output = model(batch_scene)
    print("{} nan in output \n{}".format(torch.isnan(output).sum().item(), output))
