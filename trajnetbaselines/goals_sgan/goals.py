import torch

import os
import pickle

import trajnetplusplustools





class goalModel(torch.nn.Module):
    """ Model that learns predicting the goal destination of actors. As we are using multimodel SGAN, we also need multimodal goals.
    During training, the ground truth can be used to calculate the loss. """
    def __init__(self, in_dim, out_dim):
        super(goalModel, self).__init__()
        # TODO: Write this class with all necessary functions
        raise NotImplementedError 
   
    def forward(self):
        raise NotImplementedError
    
   
   
   
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

       
class goalLoss(torch.nn.Module):
    """ Calculating the loss that we want to minimize during training. As we have multimodal goals, we maybe should use the L2 norm
    of the diffence between the ground truth and the goal_prediction closest to the ground truth. But other ideas are welcome :) """
    def __init__(self):
        super(goalLoss, self).__init__()
        # TODO: Write this class with all necessary functions
        raise NotImplementedError


def prepare_goals_data(path, subset='/train/', sample=1.0, goals=True):
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
            return None, None, False

    ## read goal files
    all_goals = {}
    all_scenes = []

    ## List file names
    files = [f.split('.')[-2] for f in os.listdir(path + subset) if f.endswith('.ndjson')]
    ## Iterate over file names
    for file in files:
        reader = trajnetplusplustools.Reader(path + subset + file + '.ndjson', scene_type='paths')
        ## Necessary modification of train scene to add filename
        scene = [(file, s_id, s) for s_id, s in reader.scenes(sample=sample)]
        if goals:
            goal_dict = pickle.load(open('goal_files/' + subset + file +'.pkl', "rb"))
            ## Get goals corresponding to train scene
            all_goals[file] = {s_id: [goal_dict[path[0].pedestrian] for path in s] for _, s_id, s in scene}
        all_scenes += scene

    if goals:
        return all_scenes, all_goals, True
    return all_scenes, None, True