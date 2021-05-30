import torch
import torch.nn as nn
import os
import pickle

import trajnetplusplustools


class goalModel(torch.nn.Module):
    """ Model that learns predicting the goal destination of actors. As we are using multimodal SGAN, we also need multimodal goals.
    During training, the ground truth can be used to calculate the loss. """
    def __init__(self, in_dim=2, hid_dim=32, num_layers=2, out_dim=2, k=3):
        """Initialization 
        
        Parameters
        ----------
        k: integer
            Number of modes 
        
        """
            
        super(goalModel, self).__init__()
        # TODO: Write this class with all necessary functions
        
        # parameters 
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.num_layers = num_layers
        self.out_dim = out_dim
        self.k = k       
        
        # layers 
        self.lstm = nn.LSTM(in_dim, hid_dim, num_layers=num_layers)
        self.linear1 = nn.Linear(hid_dim*num_layers, hid_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hid_dim, out_dim*k)
         
   
    def forward(self, batch_scene, batch_split, obs_len=9):
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
        primary_tracks = batch_scene[:obs_len, batch_split[:-1].tolist()] # (obs_len, batch_size, 2)
        
        # encode
        _, hn = self.lstm(primary_tracks) 
        
        # predict goals
        batch_size = batch_split.shape[0]-1
        hn = hn[0] # (num_layers, batch_size, hid_dim)
        hn = hn.permute(1,0,2).reshape(batch_size, self.num_layers*self.hid_dim) # (batch_size, num_layers*hid_dim)
        output = self.relu(self.linear1(hn))
        output = self.linear2(output)
        output = output.reshape(batch_size, self.k, self.out_dim)
        
        return output

    
class goalPredictor(object):
    """ Class that is used to make predictions (eg. for validation or when creating the prediction)"""
    def __init__(self, model):
        self.model = model

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

    def __call__(self, paths, scene_goal, n_predict=12, modes=1, predict_all=True, obs_length=9, start_length=0, args=None):
        
        # TODO: This code has been copied from lstm.lstm. It needs to be understood and adapted for our purpose 
        
        # self.model.eval()
        # with torch.no_grad():
        #     xy = trajnetplusplustools.Reader.paths_to_xy(paths)
        #     batch_split = [0, xy.shape[1]]

        #     if args.normalize_scene:
        #         xy, rotation, center, scene_goal = center_scene(xy, obs_length, goals=scene_goal)
            
        #     xy = torch.Tensor(xy)  #.to(self.device)
        #     scene_goal = torch.Tensor(scene_goal) #.to(device)
        #     batch_split = torch.Tensor(batch_split).long()

        #     multimodal_goals = {}
        #     for num_p in range(modes):
        #         # _, output_scenes = self.model(xy[start_length:obs_length], scene_goal, batch_split, xy[obs_length:-1].clone())
        #         _, output_scenes, _ = self.model(xy[start_length:obs_length], scene_goal, batch_split, n_predict=n_predict)
        #         output_scenes = output_scenes.numpy()
        #         if args.normalize_scene:
        #             output_scenes = augmentation.inverse_scene(output_scenes, rotation, center)
        #         output_primary = output_scenes[-n_predict:, 0]
        #         output_neighs = output_scenes[-n_predict:, 1:]
        #         ## Dictionary of predictions. Each key corresponds to one mode
        #         multimodal_goals[num_p] = [output_primary, output_neighs]

        ## Return Dictionary of predictions. Each key corresponds to one mode
        return multimodal_goals



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
        goal_pred: Tensor [batch_size, k, 2]
            Contains the k predicted goals per scene
        goal_gt: Tensor [batch_size, 2]
            Containts the goal ground truth
        
        Returns
        -------
        loss: Tensor [1,]
            L2-norm variety loss
        """
        loss = self.L2_variety_loss(goal_pred, goal_gt)
        
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