import math
import random
import numpy as np
import torch
import torch.nn as nn
import pdb

from .lstm import drop_distant

class SocialNCE():
    '''
        Social NCE: Contrastive Learning of Socially-aware Motion Representations (https://arxiv.org/abs/2012.11717)
    '''
    def __init__(self, obs_length, pred_length, head_projection, encoder_sample, temperature, horizon, sampling):

        # problem setting
        self.obs_length = obs_length
        self.pred_length = pred_length

        # nce models
        self.head_projection = head_projection # psi(), projection head
        self.encoder_sample = encoder_sample # phi(), event encoder

        # nce loss
        self.criterion = nn.CrossEntropyLoss()

        # nce param
        self.temperature = temperature # = 0.1
        self.horizon = horizon # sampling horizon = 4

        # sampling param
        self.noise_local = 0.1
        self.min_seperation = 0.2 # rho
        self.agent_zone = self.min_seperation * torch.tensor([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0], [0.707, 0.707], [0.707, -0.707], [-0.707, 0.707], [-0.707, -0.707], [0.0, 0.0]])
        
        self.i = 1

    def spatial(self, batch_scene, batch_split, batch_feat):
        '''
            Social NCE with spatial samples, i.e., samples are locations at a specific time of the future
            
        Input:
            batch_scene: coordinates of agents in the scene, tensor of shape 
                [obs_length + pred_length, total num of agents in the batch, 2]
            batch_split: index of scene split in the batch, tensor of shape [batch_size + 1]
            batch_feat: encoded features of observations, tensor of shape [pred_length, scene, feat_dim]
        Output:
            loss: social nce loss
        '''

        # -----------------------------------------------------
        #               Visualize Trajectories 
        #       (Use this block to visualize the raw data)
        # -----------------------------------------------------


        # -----------------------------------------------------
        #               Contrastive Sampling 
        # -----------------------------------------------------
        
        sample_pos, sample_neg = self._sampling_spatial(batch_scene, batch_split) # (8,2), (8,9x,2)
        if torch.isnan(sample_pos).sum() != 0 or torch.isnan(sample_neg).sum() != 0:
            print('NAN_pos={}, NAN_neg={}'.format(torch.isnan(sample_pos).sum(), torch.isnan(sample_neg).sum()))
        
        # -----------------------------------------------------
        #              Lower-dimensional Embedding 
        # -----------------------------------------------------
        
        emb_obs = self.head_projection(batch_feat[self.horizon-1,batch_split.tolist()[:-1],:]) # (num_scene, head_dim) = (8,8)        
        emb_pos = self.encoder_sample(sample_pos) # (num_scene, head_dim) = (8,8)
        emb_neg = self.encoder_sample(sample_neg) # (num_scene, num_neighbors*9, head_dim) = (8,36,8)
        
        # normalized embedding
        query = nn.functional.normalize(emb_obs, dim=-1)
        key_pos = nn.functional.normalize(emb_pos, dim=-1)
        key_neg = nn.functional.normalize(emb_neg, dim=-1)
        # -----------------------------------------------------
        #                   Compute Similarity 
        # -----------------------------------------------------

        sim_pos = (query * key_pos).sum(dim=1) # (8,)
        sim_neg = (query[:,None,:] * key_neg).sum(dim=2) # (8,9x)
        
        # -----------------------------------------------------
        #                       NCE Loss 
        # -----------------------------------------------------
        logits = torch.cat([sim_pos.unsqueeze(1), sim_neg], dim=1) / self.temperature # (8,9x+1)
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device) #(8,)
        loss = self.criterion(logits, labels)
        
        # visualize samples and raw data
        if self.i == 0:
            for i in range(batch_split.shape[0] - 1):
                traj_primary = batch_scene[:, batch_split[i]] # [time, 2]
                traj_neighbor = batch_scene[:, batch_split[i]+1:batch_split[i+1]] # [time, num, 2]
                plot_scene_with_samples(traj_primary, traj_neighbor, self.obs_length, sample_pos[i], sample_neg[i], fname='scene_{:d}.png'.format(i))
                self.i += 1

        return loss

    def event(self, batch_scene, batch_split, batch_feat):
        '''
            Social NCE with event samples, i.e., samples are spatial-temporal events at various time steps of the future
            
            Input:
            batch_scene: coordinates of agents in the scene, tensor of shape 
                [obs_length + pred_length, total num of agents in the batch, 2]
            batch_split: index of scene split in the batch, tensor of shape [batch_size + 1]
            batch_feat: encoded features of observations, tensor of shape [pred_length, scene, feat_dim]
        Output:
            loss: social nce loss
        '''
        
        # contrastive sampling
        sample_pos, sample_neg = self._sampling_event(batch_scene, batch_split) # (8,4,2), (8,4,9x,2)
        
        # lower-dimensional embedding
        emb_obs = self.head_projection(batch_feat[self.horizon-1,batch_split.tolist()[:-1],:]) # (num_scene, head_dim) = (8,8)        
        time_pos = (torch.ones(sample_pos.size(0))[:, None] * (torch.arange(self.horizon) - (self.horizon-1.0)*(0.5))[None, :]) # (num_scene, horizon)
        time_neg = (torch.ones(sample_neg.size(0), sample_neg.size(2))[:, None, :] * (torch.arange(self.horizon) - (self.horizon-1.0)*(0.5))[None, :, None]) # (num_scene, horizon, 9x)
        emb_pos = self.encoder_sample(sample_pos, time_pos[:,:,None]) # (num_scene, horizon, head_dim) = (8,4,8)
        emb_neg = self.encoder_sample(sample_neg, time_neg[:,:,:,None]) # (num_scene, horizon, num_neighbors*9, head_dim) = (8,4,9x,8)
        
        # normalized embedding
        query = nn.functional.normalize(emb_obs, dim=-1)
        key_pos = nn.functional.normalize(emb_pos, dim=-1)
        key_neg = nn.functional.normalize(emb_neg, dim=-1)
        
        # compute similarity
        sim_pos = (query[:,None,:] * key_pos).sum(dim=-1) # (num_scene, horizon) = (8,4)
        sim_neg = (query[:,None,None,:] * key_neg).sum(dim=-1) # (num_scene, horizon, num_nbr*9) = (8,4,9x)
        
        # compute NCE loss
        logits = torch.cat([sim_pos.view(-1).unsqueeze(1), 
                            sim_neg.view(sim_neg.size(0),-1).repeat_interleave(self.horizon, dim=0)], dim=1) / self.temperature # (8*4,9x+1)
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device) #(8,)
        loss = self.criterion(logits, labels)

        return loss

    def _sampling_spatial(self, batch_scene, batch_split):
        '''
            Rule is based on the Social-NCE paper. For each scene, make one positive sample and 9*n negative samples.
            where 9=self.agent_zone.size(0), n is the number of neighbor in one scene (which is different from scene to scene).
            
        inputs:
            batch_scene: (seq_len, tot_num_agents, 2) = (21,b+n,2)
            batch_split: (num_scene+1) = (9,)
        
        return:
            sample_pos: (num_scene, 2) = (8,2)
            sample_nes: (num_scene, 9*max_num_nbr, 2) = (8,9x,2)
        
        notation:
            number of scene in the batch = b (=8)
            number of neighbors in the batch = n 
            total number of agents in the batch = b+n
            max number of neighbor = x (>= n/b)
            
        '''
        
        gt_future = batch_scene[self.obs_length: self.obs_length+self.pred_length] # (pred_len, tot_num_agents, 2) = (12,b+n,2)

        # #####################################################
        #           TODO: fill the following code
        # #####################################################     
              
        # -----------------------------------------------------
        #                  Positive Samples
        # -----------------------------------------------------

        gt_primary = gt_future[self.horizon-1, batch_split.tolist()[:-1], :] # (num_scene, 2) = (8,2)
        sample_pos = gt_primary + \
            torch.rand(gt_primary.size()).sub(0.5) * self.noise_local # (num_scene, 2) = (8,2)
        assert torch.isnan(sample_pos).sum().sum() == 0, "Unvailid entries: sample_pos contains NaN's"
        
        # -----------------------------------------------------
        #                  Negative Samples
        # -----------------------------------------------------

        num_scene = batch_split.size(0) - 1 # = 8
        max_num_nbr = int(max(batch_split[1:] - batch_split[:-1]).item()) - 1
        num_neg = self.agent_zone.size(0) # = 9
        num_coor = gt_future.size(-1) # = 2
        
        neighbors = [s for s in range(batch_split[-1]) if s not in batch_split.tolist()]
        gt_neighbors = gt_future[self.horizon-1, neighbors, :] # (tot_num_nbr, num_coor) = (n,2)
        gt_neighbors = torch.tile(gt_neighbors, (1,num_neg)) # (tot_num_nbr, num_neg*num_coor) = (n,18)
        gt_neighbors = gt_neighbors.reshape(-1, num_neg, num_coor) # (tot_num_nbr, num_neg, num_coor) = (n,9,2)
        pert = self.agent_zone[None, :, :] # (1,9,2)
        sample_neg = gt_neighbors + pert + \
            torch.rand(gt_neighbors.size()).sub(0.5) * self.noise_local # (tot_num_nbr, num_neg, num_coor) = (n,9,2)
              
        # Since each scene has different number of neighbors, 
        # in order to put all scenes in one matrix,
        # inserting random negative samples to those having smaller size than the largest one at the end.
        for i,bs in enumerate(batch_split[1:]):
            nb = bs - batch_split[i] - 1 # number of neighbors in one scene
            if nb != 0: # if having neighbors in the scene
                # fill nan values with random negative sample in the same scene
                mask = torch.isnan(sample_neg[i*max_num_nbr:i*max_num_nbr+nb]) # mask of locations of nan values
                if (~mask).sum() != 0:
                    sample_rand = random.choices(np.unique(np.where(~mask)[0]), 
                                                 k=torch.where(mask)[0].unique().size(0))
                    sample_neg[i*max_num_nbr:i*max_num_nbr+nb][mask] = sample_neg[i*max_num_nbr:i*max_num_nbr+nb][[sample_rand]].view(-1)
                else:
                    sample_neg[i*max_num_nbr:i*max_num_nbr+nb][mask] = -10*torch.ones((torch.where(mask)[0].unique().size(0), num_neg, num_coor)).view(-1)
            else: 
                print('Number of neighbor in scene {} is {}'.format(i+1, nb))
            if nb < max_num_nbr:
                n_rand = (max_num_nbr-nb) # number of random samples we have to draw from sample_neg
                if nb == 0:
                    sample_rand = -10*torch.ones((n_rand, num_neg, num_coor))
                else:
                    sample_rand = sample_neg[i*max_num_nbr + torch.randint(0, nb, (n_rand,))]
                sample_neg = torch.cat([sample_neg[:(i*max_num_nbr+nb)], 
                                        sample_rand, 
                                        sample_neg[(i*max_num_nbr+nb):]], dim=0)
        
        assert sample_neg.size(0) == max_num_nbr * num_scene, f"sample_neg of wrong dimensions: {sample_neg.size(0)} instead of {max_num_nbr * num_scene}"
        sample_neg = sample_neg.reshape(num_scene, -1, num_coor) # (num_scene, max_num_nbr*num_neg, num_coor) = (8,9x,2)
                
        # -----------------------------------------------------
        #       Remove negatives that are too easy (optional)
        #       Remove negatives that are too hard (optional)
        # -----------------------------------------------------
        """"
        Dimensions:
            sample_pos  (num_scene, xy) = (8, 2)
            sample_neg  (num_scene, num_samples, xy) = (8, 9x, 2)
        """
        
        # # Parameters
        # max_dist = 6 # [m]
        # min_dist = 1.5 * self.min_seperation #0.3 [m]
        
        # # Distance
        # dist = sample_neg - torch.tile(sample_pos[:,None,:], (1, sample_neg.size(1), 1))
        # dist = dist.pow(2).sum(2).pow(0.5)
        
        # print("sample_neg: ", sample_neg.shape)
        
        
        # rand_samples = sample_neg[(dist < max_dist), :]
        # print("good_samples: ",  good_samples.shape)
        
        # print((dist > max_dist).sum(1))
        
        # # Set samples to 0
        # sample_neg[dist < max_dist, :]
        # sample_neg = sample_neg[dist < max_dist, :]
        # print(sample_neg.shape)
        # sample_neg = sample_neg[dist > min_dist, :]
        
        return sample_pos, sample_neg
    
    def _sampling_event(self, batch_scene, batch_split):
        '''
        Sample positive and negative samples for event case. 
        This will preserve the temporal dimension.
        
        inputs:
            batch_scene: (seq_len, tot_num_agents, 2) = (21,tot_num_agents,2)
            batch_split: (num_scene+1) = (9,)
        
        return:
            sample_pos: (num_scene, horizon, 2) = (8,4,2)
            sample_nes: (num_scene, horizon, max_num_nbr*num_neg, num_coor) = (8,4,9x,2)
        '''
        gt_future = batch_scene[self.obs_length: self.obs_length+self.pred_length] # (pred_len, tot_num_agents, 2) = (12,n,2)
        
        # positive samples
        gt_primary = gt_future[:self.horizon, batch_split.tolist()[:-1], :].permute(1,0,2) # (num_scene, horizon, 2) = (8,4,2)
        sample_pos = gt_primary + \
            torch.rand(gt_primary.size()).sub(0.5) * self.noise_local # (num_scene, pred_len, 2) = (8,4,2)
        
        # some parameters
        num_scene = batch_split.size(0) - 1 # = 8
        max_num_nbr = int(max(batch_split[1:] - batch_split[:-1]).item()) - 1 
        num_neg = self.agent_zone.size(0) # = 9
        num_coor = gt_future.size(-1) # = 2
        
        # negative samples
        neighbors = [s for s in range(batch_split[-1]) if s not in batch_split.tolist()]
        gt_neighbors = gt_future[:self.horizon, neighbors, :] # (horizon, tot_num_nbr, num_coor) = (4,n,2)
        gt_neighbors = torch.tile(gt_neighbors, (1,num_neg)) # (horizon, tot_num_nbr, num_neg*num_coor) = (4,n,18)
        gt_neighbors = gt_neighbors.reshape(self.horizon, -1, num_neg, num_coor) # (horizon, tot_num_nbr, num_neg, num_coor) = (4,n,9,2)
        pert = self.agent_zone[None, None, :, :] # (1,1,9,2)
        sample_neg = gt_neighbors + pert + \
            torch.rand(gt_neighbors.size()).sub(0.5) * self.noise_local # (horizon, tot_num_nbr, num_neg, num_coor) = (4,n,9,2)
        
        # padding zero
        for i,bs in enumerate(batch_split[1:]):
            nb = bs - batch_split[i] - 1 
            if nb < max_num_nbr:
                n_rand = (max_num_nbr-nb) # number of random samples we have to draw from sample_neg
                sample_rand = sample_neg[:, i*max_num_nbr + torch.randint(0, nb, (n_rand,))]
                sample_neg = torch.cat([sample_neg[:,:(i*max_num_nbr+nb)], 
                                        sample_rand, 
                                        sample_neg[:,(i*max_num_nbr+nb):]], dim=1) # (horizon, num_scene*max_num_nbr, num_neg, num_coor)
                        
        assert sample_neg.size(1) == max_num_nbr * num_scene, f"sample_neg of wrong dimensions: {sample_neg.size(1)} instead of {max_num_nbr * num_scene}"
        sample_neg = sample_neg.reshape(self.horizon, num_scene, -1, num_coor) # (horizon, num_scene, max_num_nbr*num_neg, num_coor) = (4,8,9x,2)
        sample_neg = sample_neg.permute(1,0,2,3) # (num_scene, horizon, max_num_nbr*num_neg, num_coor) = (8,4,9x,2)
        
        return sample_pos, sample_neg

    
class EventEncoder(nn.Module):
    '''
        Event encoder that maps an sampled event (location & time) to the embedding space
    '''
    def __init__(self, hidden_dim, head_dim):

        super(EventEncoder, self).__init__()
        self.temporal = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(inplace=True)
            )
        self.spatial = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(inplace=True)
            )
        self.encoder = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, head_dim)
        )

    def forward(self, state, time):
        emb_state = self.spatial(state)
        emb_time = self.temporal(time)
        out = self.encoder(torch.cat([emb_time, emb_state], axis=-1))
        return out


class SpatialEncoder(nn.Module):
    '''
        Spatial encoder that maps an sampled location to the embedding space
    '''
    def __init__(self, hidden_dim, head_dim):
        super(SpatialEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, head_dim)
        )

    def forward(self, state):
        return self.encoder(state)


class ProjHead(nn.Module):
    '''
        Nonlinear projection head that maps the extracted motion features to the embedding space
    '''
    def __init__(self, feat_dim, hidden_dim, head_dim):
        super(ProjHead, self).__init__()
        self.head = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, head_dim)
            )

    def forward(self, feat):
        return self.head(feat)


def plot_scene(primary, neighbor, fname):
    '''
        Plot raw trajectories
    '''
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig = plt.figure(frameon=False)
    fig.set_size_inches(16, 9)
    ax = fig.add_subplot(1, 1, 1)

    ax.plot(primary[:, 0], primary[:, 1], 'k-')
    for i in range(neighbor.size(1)):
        ax.plot(neighbor[:, i, 0], neighbor[:, i, 1], 'b-.')

    ax.set_aspect('equal')
    plt.grid()
    plt.savefig(fname, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    
    
def plot_scene_with_samples(primary, neighbor, obs_len, sample_pos, sample_neg, fname):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig = plt.figure(frameon=False)
    fig.set_size_inches(16, 9)
    ax = fig.add_subplot(1, 1, 1)
    
    ax.plot(primary[0,0], primary[0,1], 'kx')
    ax.plot(primary[:obs_len, 0], primary[:obs_len, 1], 'k.-', label='primary (observed)')
    ax.plot(primary[(obs_len-1):, 0], primary[(obs_len-1):, 1], 'k.:', label='primary (truth)')
    
    for i in range(neighbor.size(1)):
        ax.plot(neighbor[:obs_len, i, 0], neighbor[:obs_len, i, 1], 'k-', alpha=0.3, label='neighbors (observed)')
        ax.plot(neighbor[(obs_len-1):, i, 0], neighbor[(obs_len-1):, i, 1], 'k:', alpha=0.3, label='neighbors (truth)')
        ax.plot(neighbor[0,i,0], neighbor[0,i,1], 'kx', label='start')
        ax.plot(neighbor[-1,i,0], neighbor[-1,i,1], 'k.', label='end')
    
    # plot positive and negative samples
    ax.plot(sample_pos[0], sample_pos[1], 'g*', label='positive samples')
    for i in range(sample_neg.size(0)):
        ax.plot(sample_neg[i,0], sample_neg[i,1], 'r.', label='negative samples')
    
    lines, labels = [], []
    for line, label in zip(ax.get_legend_handles_labels()[0], ax.get_legend_handles_labels()[1]):
        if label not in labels:
            lines.append(line)
            labels.append(label)
            
    ax.set_aspect('equal')
    ax.legend(lines, labels, loc='best')
    plt.grid()
    plt.savefig(fname, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
