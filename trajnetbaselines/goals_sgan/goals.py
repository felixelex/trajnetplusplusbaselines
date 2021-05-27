import torch

class goalModel(torch.nn.Module):
    """ Model that learns predicting the goal destination of actors. As we are using multimodel SGAN, we also need multimodal goals.
    During training, the ground truth can be used to calculate the loss. """
    def __init__(self):
        super(goalModel, self).__init__()
        # TODO: Write this class with all necessary functions
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

        
        
### HELPER FUNCTIONS FOR TRAINER ###

def extractGroundTruthGoals():
    """ Function that takes a batch of scenes as input and returns a batch of goal coordinates. """
    # TODO: Write this function
    raise NotImplementedError