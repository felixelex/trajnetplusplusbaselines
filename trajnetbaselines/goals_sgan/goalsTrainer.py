"""Command line tool to train goal model."""

import argparse
import logging
import socket
import sys
import time
import random
import os
import pickle
import copy

import numpy as np

import torch
import torch.nn as nn
import trajnetplusplustools

from .. import augmentation
from .. import __version__ as VERSION

from .sgan import drop_distant
from ..lstm.utils import center_scene, random_rotation

from .goals import goalModel, goalLoss, prepare_goals_data, get_goals, interpolate_batch_scene


class GoalsTrainer(object):
    def __init__(self, model=None, optimizer=None, lr_scheduler=None, device = None, batch_size=8, 
                 obs_length=9, pred_length=12, augment=True, normalize_scene=False, save_every=1, start_length=0, 
                 val_flag=True):
        self.model = model if model is not None else goalModel()
        self.optimizer = optimizer if optimizer is not None else torch.optim.Adam(
                           model.parameters(), lr=1e-3, weight_decay=1e-4)
        self.lr_scheduler = lr_scheduler if lr_scheduler is not None else \
                              torch.optim.lr_scheduler.StepLR(optimizer, 10)
                              
        self.criterion = goalLoss()
        self.device = device if device is not None else torch.device('cpu')
        self.model = self.model.to(self.device)
        self.criterion = self.criterion.to(self.device)
        self.log = logging.getLogger(self.__class__.__name__)
        self.save_every = save_every

        self.batch_size = batch_size
        self.obs_length = obs_length
        self.pred_length = pred_length
        self.seq_length = self.obs_length+self.pred_length
        self.start_length = start_length

        self.augment = augment
        self.normalize_scene = normalize_scene

        self.val_flag = val_flag

    def loop(self, train_scenes, val_scenes, out, epochs=35, start_epoch=0):
        for epoch in range(start_epoch, epochs):
            if epoch % self.save_every == 0:
                state = {'epoch': epoch, 'state_dict': self.model.state_dict(),
                         'optimizer': self.optimizer.state_dict(),
                         'lr_scheduler': self.lr_scheduler.state_dict()}
                self.model.save(state, out + '.epoch{}'.format(epoch))
                        
            self.train(train_scenes, epoch)
            
            if self.val_flag:               
                self.val(val_scenes, epoch)

        state = {'epoch': epoch + 1, 'state_dict': self.model.state_dict(),
                 'optimizer': self.optimizer.state_dict(),
                 'lr_scheduler': self.lr_scheduler.state_dict()}
        self.model.save(state, out + '.epoch{}'.format(epoch + 1))
        self.model.save(state, out)
        
    def get_lr(self):
        for param_group in self.optimizer.param_groups:
            return param_group['lr']
              
        
    def train(self, scenes, epoch):
        start_time = time.time()

        print('epoch', epoch)

        random.shuffle(scenes)
        epoch_loss = 0.0
        self.model.train()
        self.optimizer.zero_grad()

        ## Initialize batch of scenes
        batch_scene = []
        batch_scene_goal = []
        batch_split = [0]

        for scene_i, (filename, scene_id, paths) in enumerate(scenes):
            scene_start = time.time()

            ## make new scene
            scene = trajnetplusplustools.Reader.paths_to_xy(paths)

            ## get goals
            scene_goal = get_goals(scene, self.obs_length, self.pred_length)

            ## Drop Distant
            scene, mask = drop_distant(scene)
            scene_goal = scene_goal[mask]

            """ Format:
            scene_goal  [n_actors, 2]
            scene       [time_steps, num_actors, 2]
            """

            ##process scene
            if self.normalize_scene:
                scene, _, _, scene_goal = center_scene(scene, self.obs_length, goals=scene_goal)
            if self.augment:
                scene, scene_goal = random_rotation(scene, goals=scene_goal)
            
            ## Augment scene to batch of scenes
            batch_scene.append(scene)
            batch_split.append(int(scene.shape[1]))
            batch_scene_goal.append(scene_goal)

            if ((scene_i + 1) % self.batch_size == 0) or ((scene_i + 1) == len(scenes)):
                ## Construct Batch
                batch_scene = np.concatenate(batch_scene, axis=1)
                batch_scene_goal = np.concatenate(batch_scene_goal, axis=0)
                batch_split = np.cumsum(batch_split)
                
                batch_scene = torch.Tensor(batch_scene).to(self.device)
                batch_scene_goal = torch.Tensor(batch_scene_goal).to(self.device)
                batch_split = torch.Tensor(batch_split).to(self.device).long()

                ## Select only goals of primary actor
#                 batch_scene_goal = batch_scene_goal[batch_split[:-1],:]

                preprocess_time = time.time() - scene_start

                ## Train Batch
                loss = self.train_batch(batch_scene, batch_scene_goal, batch_split)
                epoch_loss += loss
                total_time = time.time() - scene_start

                ## Reset Batch
                batch_scene = []
                batch_scene_goal = []
                batch_split = [0]

            if (scene_i + 1) % (100*self.batch_size) == 0:
                self.log.info({
                    'type': 'train',
                    'epoch': epoch, 'batch': scene_i, 'n_batches': len(scenes),
                    'time': round(total_time, 3),
                    'data_time': round(preprocess_time, 3),
                    'lr': self.get_lr(),
                    'loss': round(loss, 3),
                })

        self.lr_scheduler.step()

        self.log.info({
            'type': 'train-epoch',
            'epoch': epoch + 1,
            'loss': round(epoch_loss / (len(scenes)), 5),
            'time': round(time.time() - start_time, 1),
        })


    def val(self, scenes, epoch):
        eval_start = time.time()

        val_loss = 0.0
        test_loss = 0.0
        self.model.train()  # so that it does not return positions but still normals

        ## Initialize batch of scenes
        batch_scene = []
        batch_scene_goal = []
        batch_split = [0]

        for scene_i, (filename, scene_id, paths) in enumerate(scenes):
            # make new scene
            scene = trajnetplusplustools.Reader.paths_to_xy(paths)

            ## get goals
            scene_goal = get_goals(scene, self.obs_length, self.pred_length)
            
            ## Drop Distant
            scene, mask = drop_distant(scene)
            scene_goal = scene_goal[mask]

            ## process scene
            if self.normalize_scene:
                scene, _, _, scene_goal = center_scene(scene, self.obs_length, goals=scene_goal)

            ## Augment scene to batch of scenes
            batch_scene.append(scene)
            batch_split.append(int(scene.shape[1]))
            batch_scene_goal.append(scene_goal)

            if ((scene_i + 1) % self.batch_size == 0) or ((scene_i + 1) == len(scenes)):
                ## Construct Batch
                batch_scene = np.concatenate(batch_scene, axis=1)
                batch_scene_goal = np.concatenate(batch_scene_goal, axis=0)
                batch_split = np.cumsum(batch_split)
                
                batch_scene = torch.Tensor(batch_scene).to(self.device)
                batch_scene_goal = torch.Tensor(batch_scene_goal).to(self.device)
                batch_split = torch.Tensor(batch_split).to(self.device).long()
                
                ## Select only goals of primary actor
                batch_scene_goal = batch_scene_goal[batch_split[:-1],:]
                
                loss_val_batch, loss_test_batch = self.val_batch(batch_scene, batch_scene_goal, batch_split)
                val_loss += loss_val_batch
                test_loss += loss_test_batch

                ## Reset Batch
                batch_scene = []
                batch_scene_goal = []
                batch_split = [0]

        eval_time = time.time() - eval_start

        self.log.info({
            'type': 'val-epoch',
            'epoch': epoch + 1,
            'loss': round(val_loss / (len(scenes)), 3),
            'test_loss': round(test_loss / len(scenes), 3),
            'time': round(eval_time, 1),
        })
        
    def train_batch(self, batch_scene, goal_gt, batch_split):
        """Training of B batches in parallel, B : batch_size

        Parameters
        ----------
        batch_scene : Tensor [seq_length, num_tracks, 2]
            Tensor of batch of scenes.
        goal_gt : Tensor [num_tracks, 2]
            Tensor of goals of each track in batch
        batch_split : Tensor [batch_size + 1]
            Tensor defining the split of the batch.
            Required to identify the tracks of to the same scene

        Returns
        -------
        loss : scalar
            Training loss of the batch
        """
        
        seq_length = batch_scene.shape[0]
        batch_scene = interpolate_batch_scene(batch_scene, seq_length)
        
        # assert not batch_scene.isnan().any(), 'NaNs in batch_scene'
        goal_pred = self.model(batch_scene, obs_len=self.obs_length)
        assert not goal_pred.isnan().any(), 'NaNs in goal_pred: \n{}'.format(goal_pred)
        
        # goal_pred [num_tracks, k, out_dim=2]
        loss = self.criterion(goal_pred, goal_gt)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1e+2)
        self.optimizer.step()

        return loss.item()

    def val_batch(self, batch_scene, batch_scene_goal, batch_split):
        """Validation of B batches in parallel, B : batch_size

        Parameters
        ----------
        batch_scene : Tensor [seq_length, num_tracks, 2]
            Tensor of batch of scenes.
        batch_scene_goal : Tensor [num_tracks, 2]
            Tensor of goals of each track in batch
        batch_split : Tensor [batch_size + 1]
            Tensor defining the split of the batch.
            Required to identify the tracks of to the same scene

        Returns
        -------
        loss : scalar
            Validation loss of the batch when groundtruth of neighbours
            is not provided
        """
        
        with torch.no_grad():
            goal_pred = self.model(batch_scene, obs_len=self.obs_length)
            loss = self.criterion(goal_pred, batch_scene_goal)

        return 0.0, loss.item()
    
    
    
def main(epochs=15):
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=epochs, type=int,
                        help='number of epochs')
    parser.add_argument('--save_every', default=5, type=int,
                        help='frequency of saving model (in terms of epochs)')
    parser.add_argument('--obs_length', default=9, type=int,
                        help='observation length')
    parser.add_argument('--pred_length', default=12, type=int,
                        help='prediction length')
    parser.add_argument('--start_length', default=0, type=int,
                        help='starting time step of encoding observation')
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('-o', '--output', default=None,
                        help='output file')
    parser.add_argument('--disable-cuda', action='store_true',
                        help='disable CUDA')
    parser.add_argument('--path', default='trajdata',
                        help='glob expression for data files')
    parser.add_argument('--sample', default=1.0, type=float,
                        help='sample ratio when loading train/val scenes')

    ## Augmentations
    parser.add_argument('--augment', action='store_true',
                        help='perform rotation augmentation')
    parser.add_argument('--normalize_scene', action='store_true',
                        help='rotate scene so primary pedestrian moves northwards at end of observation')
    
    ## Loading pre-trained models
    pretrain = parser.add_argument_group('pretraining')
    pretrain.add_argument('--load-state', default=None,
                          help='load a pickled model state dictionary before training')
    pretrain.add_argument('--load-full-state', default=None,
                          help='load a pickled full state dictionary before training')
    pretrain.add_argument('--nonstrict-load-state', default=None,
                          help='load a pickled state dictionary before training')
    
    ## Hyperparameters
    hyperparameters = parser.add_argument_group('hyperparameters')
    hyperparameters.add_argument('--lr', default=1e-3, type=float,
                                 help='initial learning rate')
    hyperparameters.add_argument('--k', type=int, default=3,
                                 help='number of samples for variety loss')
    hyperparameters.add_argument('--step_size', default=10, type=int,
                                 help='step_size of lr scheduler')
    
    args = parser.parse_args()
    
    ## Fixed set of scenes if sampling
    if args.sample < 1.0:
        torch.manual_seed("080819")
        random.seed(1)

    if not os.path.exists('OUTPUT_BLOCK/{}'.format(args.path)):
        os.makedirs('OUTPUT_BLOCK/{}'.format(args.path))
    args.output = 'OUTPUT_BLOCK/{}/goalsModel.pkl'.format(args.path)
    
    # configure logging
    from pythonjsonlogger import jsonlogger
    if args.load_full_state:
        file_handler = logging.FileHandler(args.output + '.log', mode='a')
    else:
        file_handler = logging.FileHandler(args.output + '.log', mode='w')
    file_handler.setFormatter(jsonlogger.JsonFormatter('%(message)s %(levelname)s %(name)s %(asctime)s'))
    stdout_handler = logging.StreamHandler(sys.stdout)
    logging.basicConfig(level=logging.INFO, handlers=[stdout_handler, file_handler])
    logging.info({
        'type': 'process',
        'argv': sys.argv,
        'args': vars(args),
        'version': VERSION,
        'hostname': socket.gethostname(),
    })
    
    # refactor args for --load-state
    args.load_state_strict = True
    if args.nonstrict_load_state:
        args.load_state = args.nonstrict_load_state
        args.load_state_strict = False
    if args.load_full_state:
        args.load_state = args.load_full_state
        
    args.device = torch.device('cpu')
    
    args.path = 'DATA_BLOCK/' + args.path
    
    ## Prepare data
    train_scenes, _ = prepare_goals_data(args.path, subset='/train/', sample=args.sample)
    val_scenes, val_flag = prepare_goals_data(args.path, subset='/val/', sample=args.sample)
        
    # Goal model (TO BE MODIFIED)
    model = goalModel(in_dim=2, out_dim=2, k=args.k)
    
    # Optimizer and Scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.step_size)
    start_epoch = 0
        
    # train
    if args.load_state:
        # load pretrained model.
        # useful for tranfer learning
        print("Loading Model Dict")
        with open(args.load_state, 'rb') as f:
            checkpoint = torch.load(f)
        pretrained_state_dict = checkpoint['state_dict']
        model.load_state_dict(pretrained_state_dict, strict=args.load_state_strict)

        if args.load_full_state:
        # load optimizers from last training
        # useful to continue model training
            print("Loading Optimizer Dict")
            optimizer.load_state_dict(checkpoint['g_optimizer'])
            lr_scheduler.load_state_dict(checkpoint['g_lr_scheduler'])
            start_epoch = checkpoint['epoch']
    
    #trainer
    trainer = GoalsTrainer(model, optimizer=optimizer, lr_scheduler=lr_scheduler, device=args.device,
                      batch_size=args.batch_size, obs_length=args.obs_length, pred_length=args.pred_length,
                      augment=args.augment, normalize_scene=args.normalize_scene, save_every=args.save_every,
                      start_length=args.start_length, val_flag=val_flag)
    trainer.loop(train_scenes, val_scenes, args.output, epochs=args.epochs, start_epoch=start_epoch)


if __name__ == '__main__':
    main()
    
    
    
    
    
    
    