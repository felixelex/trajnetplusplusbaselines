"""Command line tool to train an LSTM model."""

import argparse
import logging
import socket
import sys
import time
import random
import os
import pickle
import torch
import numpy as np

import trajnetplusplustools

from .. import augmentation
from .loss import PredictionLoss, L2Loss
from .lstm import LSTM, LSTMPredictor, drop_distant
from .gridbased_pooling import GridBasedPooling
from .non_gridbased_pooling import NN_Pooling, HiddenStateMLPPooling, AttentionMLPPooling, DirectionalMLPPooling
from .non_gridbased_pooling import NN_LSTM, TrajectronPooling, SAttention_fast
from .more_non_gridbased_pooling import NMMP

from .. import __version__ as VERSION

from .utils import center_scene, random_rotation
from .data_load_utils import prepare_data
from .contrastive import SocialNCE, ProjHead, EventEncoder, SpatialEncoder

class Trainer(object):
    def __init__(self, projection_head=None, encoder_sample=None, 
                 contrast_weight=1.0, contrast_temperature=0.07, contrast_horizon=4, contrast_sampling='single',
                 model=None, criterion=None, optimizer=None, lr_scheduler=None,
                 device=None, batch_size=8, obs_length=9, pred_length=12, augment=True,
                 normalize_scene=False, save_every=1, start_length=0, obs_dropout=False,
                 augment_noise=False, col_weight=0.0, col_gamma=2.0, val_flag=True):

        self.model = model if model is not None else LSTM()
        self.criterion = criterion if criterion is not None else PredictionLoss()
        self.optimizer = optimizer if optimizer is not None else \
                         torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        self.lr_scheduler = lr_scheduler if lr_scheduler is not None else \
                            torch.optim.lr_scheduler.StepLR(self.optimizer, 15)

        self.device = device if device is not None else torch.device('cpu')
        self.model = self.model.to(self.device)
        self.criterion = self.criterion.to(self.device)
        self.log = logging.getLogger(self.__class__.__name__)
        self.save_every = save_every

        self.batch_size = batch_size
        self.obs_length = obs_length
        self.pred_length = pred_length
        self.seq_length = self.obs_length+self.pred_length

        self.augment = augment
        self.augment_noise = augment_noise
        self.col_weight = col_weight
        self.col_gamma = col_gamma		
        self.normalize_scene = normalize_scene

        self.start_length = start_length
        self.obs_dropout = obs_dropout

        self.contrastive = SocialNCE(obs_length, pred_length, projection_head.to(self.device), encoder_sample.to(self.device), contrast_temperature, contrast_horizon, contrast_sampling)
        self.contrast_weight = contrast_weight
        self.contrast_sampling = contrast_sampling

        self.val_flag = val_flag

    def loop(self, train_scenes, val_scenes, train_goals, val_goals, out, epochs=35, start_epoch=0):
        for epoch in range(start_epoch, start_epoch + epochs):
            if epoch % self.save_every == 0:
                state = {'epoch': epoch, 'state_dict': self.model.state_dict(),
                         'optimizer': self.optimizer.state_dict(),
                         'scheduler': self.lr_scheduler.state_dict()}
                LSTMPredictor(self.model).save(state, out + '.epoch{}'.format(epoch))
            self.train(train_scenes, train_goals, epoch)
            if self.val_flag:
                self.val(val_scenes, val_goals, epoch)


        state = {'epoch': epoch + 1, 'state_dict': self.model.state_dict(),
                 'optimizer': self.optimizer.state_dict(),
                 'scheduler': self.lr_scheduler.state_dict()}
        LSTMPredictor(self.model).save(state, out + '.epoch{}'.format(epoch + 1))
        LSTMPredictor(self.model).save(state, out)

    def get_lr(self):
        for param_group in self.optimizer.param_groups:
            return param_group['lr']

    def train(self, scenes, goals, epoch):
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
        skip = 0

        for scene_i, (filename, scene_id, paths) in enumerate(scenes):
            scene_start = time.time()

            ## make new scene
            scene = trajnetplusplustools.Reader.paths_to_xy(paths)

            ## get goals
            if goals is not None:
                scene_goal = np.array(goals[filename][scene_id])
            else:
                scene_goal = np.array([[0, 0] for path in paths])

            ## Drop Distant
            scene, mask = drop_distant(scene)
            scene_goal = scene_goal[mask]

            ##process scene
            if self.normalize_scene:
                scene, _, _, scene_goal = center_scene(scene, self.obs_length, goals=scene_goal)
            if self.augment:
                scene, scene_goal = random_rotation(scene, goals=scene_goal)
            if self.augment_noise:
                scene = augmentation.add_noise(scene, thresh=0.02, ped='neigh')

            ## Augment scene to batch of scenes
            if np.isnan(scene).sum() == 0:
                batch_scene.append(scene)
                batch_split.append(int(scene.shape[1]))
                batch_scene_goal.append(scene_goal)
            else:
                # remove the scenes with missing data
                skip += 1
                        
            if ((scene_i + 1) % self.batch_size == 0) or ((scene_i + 1) == len(scenes)):
                ## Construct Batch
                batch_scene = np.concatenate(batch_scene, axis=1)
                batch_scene_goal = np.concatenate(batch_scene_goal, axis=0)
                batch_split = np.cumsum(batch_split)
                
                batch_scene = torch.Tensor(batch_scene).to(self.device)
                batch_scene_goal = torch.Tensor(batch_scene_goal).to(self.device)
                batch_split = torch.Tensor(batch_split).to(self.device).long()

                preprocess_time = time.time() - scene_start

                ## Train Batch
                loss, loss_pred, loss_nce = self.train_batch(batch_scene, batch_scene_goal, batch_split)
                epoch_loss += loss
                total_time = time.time() - scene_start

                ## Reset Batch
                batch_scene = []
                batch_scene_goal = []
                batch_split = [0]

            if (scene_i + 1) % (10*self.batch_size) == 0:
                self.log.info({
                    # 'type': 'train',
                    'epoch': epoch, 'batch': '{:d} / {:d}'.format(scene_i, len(scenes)),
                    # 'time': round(total_time, 2),
                    # 'data_time': round(preprocess_time, 2),
                    'lr': '{:.1e}'.format(self.get_lr()),
                    'loss': round(loss, 2),
                    'pred': round(loss_pred, 2),
                    'nce': round(loss_nce, 2),
                })

        self.lr_scheduler.step()
        self.log.info({
            'type': 'train-epoch',
            'epoch': epoch + 1,
            'loss': round(epoch_loss / (len(scenes)), 4),
            'time': round(time.time() - start_time, 1),
            'skip': skip,
        })

    def val(self, scenes, goals, epoch):
        eval_start = time.time()

        val_loss = 0.0
        test_loss = 0.0
        self.model.eval()

        ## Initialize batch of scenes
        batch_scene = []
        batch_scene_goal = []
        batch_split = [0]

        for scene_i, (filename, scene_id, paths) in enumerate(scenes):
            ## make new scene
            scene = trajnetplusplustools.Reader.paths_to_xy(paths)

            ## get goals
            if goals is not None:
                scene_goal = np.array(goals[filename][scene_id])
            else:
                scene_goal = np.array([[0, 0] for path in paths])

            ## Drop Distant
            scene, mask = drop_distant(scene)
            scene_goal = scene_goal[mask]

            ##process scene
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

    def train_batch(self, batch_scene, batch_scene_goal, batch_split):
        """Training of B batches in parallel, B : batch_size

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
            Training loss of the batch
        """

        ## If observation dropout active
        if self.obs_dropout:
            self.start_length = random.randint(0, self.obs_length - 2)

        observed = batch_scene[self.start_length:self.obs_length].clone()
        prediction_truth = batch_scene[self.obs_length:self.seq_length-1].clone()
        targets = batch_scene[self.obs_length:self.seq_length] - batch_scene[self.obs_length-1:self.seq_length-1]

        rel_outputs, outputs, batch_feat = self.model(observed, batch_scene_goal, batch_split, prediction_truth)

        ## Loss wrt primary tracks of each scene only
        loss_predict = self.criterion(rel_outputs[-self.pred_length:], targets, batch_split) * self.batch_size

        # ------------- Social NCE ----------------
        if self.contrast_weight > 0:
            if self.contrast_sampling == 'single':
                loss_contrastive = self.contrastive.spatial(batch_scene, batch_split, batch_feat)
            elif self.contrast_sampling == 'multi':
                loss_contrastive = self.contrastive.event(batch_scene, batch_split, batch_feat)
            else:
                raise NotImplementedError
            loss = loss_predict + loss_contrastive * self.contrast_weight
        else:
            loss = loss_predict
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item(), loss_predict.item(), loss_contrastive.item() if self.contrast_weight > 0 else 0.0

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
            is provided
        loss_test : scalar
            Validation loss of the batch when groundtruth of neighbours
            is not provided (evaluation scenario)
        """

        if self.obs_dropout:
            self.start_length = 0

        observed = batch_scene[self.start_length:self.obs_length]
        prediction_truth = batch_scene[self.obs_length:self.seq_length-1].clone()  ## CLONE
        targets = batch_scene[self.obs_length:self.seq_length] - batch_scene[self.obs_length-1:self.seq_length-1]
        observed_test = observed.clone()

        with torch.no_grad():
            ## groundtruth of neighbours provided (Better validation curve to monitor model)
            rel_outputs, _, _ = self.model(observed, batch_scene_goal, batch_split, prediction_truth)
            loss = self.criterion(rel_outputs[-self.pred_length:], targets, batch_split) * self.batch_size

            ## groundtruth of neighbours not provided
            rel_outputs_test, _, _ = self.model(observed_test, batch_scene_goal, batch_split, n_predict=self.pred_length)
            loss_test = self.criterion(rel_outputs_test[-self.pred_length:], targets, batch_split) * self.batch_size

        return loss.item(), loss_test.item()

def main(epochs=25):
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=epochs, type=int,
                        help='number of epochs')
    parser.add_argument('--save_every', default=1, type=int,
                        help='frequency of saving model (in terms of epochs)')
    parser.add_argument('--obs_length', default=9, type=int,
                        help='observation length')
    parser.add_argument('--pred_length', default=12, type=int,
                        help='prediction length')
    parser.add_argument('--start_length', default=0, type=int,
                        help='starting time step of encoding observation')
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--lr', default=1e-3, type=float,
                        help='initial learning rate')
    parser.add_argument('--scheduler_gamma', default=0.5, type=float,
                        help='')    
    parser.add_argument('--step_size', default=10, type=int,
                        help='step_size of lr scheduler')
    parser.add_argument('-o', '--output', default=None,
                        help='output file')
    parser.add_argument('--disable-cuda', action='store_true',
                        help='disable CUDA')
    parser.add_argument('--path', default='trajdata',
                        help='glob expression for data files')
    parser.add_argument('--goals', action='store_true',
                        help='flag to consider goals of pedestrians')
    parser.add_argument('--loss', default='pred', choices=('L2', 'pred'),
                        help='loss objective, L2 loss (L2) and Gaussian loss (pred)')
    parser.add_argument('--type', default='vanilla',
                        choices=('vanilla', 'occupancy', 'directional', 'social', 'hiddenstatemlp', 's_att_fast',
                                 'directionalmlp', 'nn', 'attentionmlp', 'nn_lstm', 'traj_pool', 'nmmp', 'dir_social'),
                        help='type of interaction encoder')
    parser.add_argument('--sample', default=1.0, type=float,
                        help='sample ratio when loading train/val scenes')

    ## Augmentations
    parser.add_argument('--augment', action='store_true',
                        help='perform rotation augmentation')
    parser.add_argument('--normalize_scene', action='store_true',
                        help='rotate scene so primary pedestrian moves northwards at end of observation')
    parser.add_argument('--augment_noise', action='store_true',
                        help='flag to add noise to observations for robustness')
    parser.add_argument('--obs_dropout', action='store_true',
                        help='perform observation length dropout')

    ## Loading pre-trained models
    pretrain = parser.add_argument_group('pretraining')
    pretrain.add_argument('--load-state', default=None,
                          help='load a pickled model state dictionary before training')
    pretrain.add_argument('--load-full-state', default=None,
                          help='load a pickled full state dictionary before training')
    pretrain.add_argument('--nonstrict-load-state', default=None,
                          help='load a pickled state dictionary before training')

    ## Sequence Encoder Hyperparameters
    hyperparameters = parser.add_argument_group('hyperparameters')
    hyperparameters.add_argument('--hidden-dim', type=int, default=128,
                                 help='LSTM hidden dimension')
    hyperparameters.add_argument('--coordinate-embedding-dim', type=int, default=64,
                                 help='coordinate embedding dimension')
    hyperparameters.add_argument('--pool_dim', type=int, default=256,
                                 help='output dimension of interaction vector')
    hyperparameters.add_argument('--goal_dim', type=int, default=64,
                                 help='goal embedding dimension')

    ## Grid-based pooling
    hyperparameters.add_argument('--cell_side', type=float, default=0.6,
                                 help='cell size of real world (in m) for grid-based pooling')
    hyperparameters.add_argument('--n', type=int, default=12,
                                 help='number of cells per side for grid-based pooling')
    hyperparameters.add_argument('--layer_dims', type=int, nargs='*', default=[512],
                                 help='interaction module layer dims for gridbased pooling')
    hyperparameters.add_argument('--embedding_arch', default='one_layer',
                                 help='interaction encoding arch for gridbased pooling')
    hyperparameters.add_argument('--pool_constant', default=0, type=int,
                                 help='background value (when cell empty) of gridbased pooling')
    hyperparameters.add_argument('--norm_pool', action='store_true',
                                 help='normalize the scene along direction of movement during grid-based pooling')
    hyperparameters.add_argument('--front', action='store_true',
                                 help='flag to only consider pedestrian in front during grid-based pooling')
    hyperparameters.add_argument('--latent_dim', type=int, default=16,
                                 help='latent dimension of encoding hidden dimension during social pooling')
    hyperparameters.add_argument('--norm', default=0, type=int,
                                 help='normalization scheme for input batch during grid-based pooling')

    ## Non-Grid-based pooling
    hyperparameters.add_argument('--no_vel', action='store_true',
                                 help='flag to not consider relative velocity of neighbours')
    hyperparameters.add_argument('--spatial_dim', type=int, default=32,
                                 help='embedding dimension for relative position')
    hyperparameters.add_argument('--vel_dim', type=int, default=32,
                                 help='embedding dimension for relative velocity')
    hyperparameters.add_argument('--neigh', default=4, type=int,
                                 help='number of nearest neighbours to consider')
    hyperparameters.add_argument('--mp_iters', default=5, type=int,
                                 help='message passing iterations in NMMP')

    ## Collision Loss
    hyperparameters.add_argument('--col_weight', default=0., type=float,
                                 help='collision loss weight')
    hyperparameters.add_argument('--col_gamma', default=2.0, type=float,
                                 help='hyperparameter in collision loss')

    # Social-NCE
    hyperparameters.add_argument('--contrast_weight', default=0.0, type=float,
                                 help='loss weight')
    hyperparameters.add_argument('--contrast_temperature', default=0.07, type=float,
                                 help='')
    hyperparameters.add_argument('--contrast_sampling', type=str, default='single',
                                 help='single, multi')
    hyperparameters.add_argument('--contrast_horizon', default=4, type=int,
                                 help='')
    hyperparameters.add_argument('--contrast_pretrain', default=0, type=int,
                                 help='number of epoch to pretrain contrastive heads')
    hyperparameters.add_argument('--contrast_dim', default=8, type=int,
                                 help='dimension of projected embedding')
    args = parser.parse_args()

    ## Fixed set of scenes if sampling
    if args.sample < 1.0:
        torch.manual_seed("080819")
        random.seed(1)

    ## Prepare data
    train_scenes, train_goals, _ = prepare_data('DATA_BLOCK/' + args.path, subset='/train/', sample=args.sample, goals=args.goals)
    val_scenes, val_goals, val_flag = prepare_data('DATA_BLOCK/' + args.path, subset='/val/', sample=args.sample, goals=args.goals)

    args.path += '/{}/'.format(args.type)

    # ------------- Social NCE ----------------
    if args.contrast_weight > 0:
        args.path += 'snce_{}_w_{:.2f}_h_{:d}_t_{:.2f}_p_{:d}'.format(args.contrast_sampling, args.contrast_weight, args.contrast_horizon, args.contrast_temperature, args.contrast_pretrain)
    else:
        args.path += 'baseline'

    ## Define location to save trained model
    if not os.path.exists('OUTPUT_BLOCK/{}'.format(args.path)):
        os.makedirs('OUTPUT_BLOCK/{}'.format(args.path))
    if args.goals:
        args.output = 'OUTPUT_BLOCK/{}/lstm_goals_{}_{}.pkl'.format(args.path, args.type, args.output)
    else:
        args.output = 'OUTPUT_BLOCK/{}/lstm_{}_{}.pkl'.format(args.path, args.type, args.output)

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
    # loading a previously saved model
    args.load_state_strict = True
    if args.nonstrict_load_state:
        args.load_state = args.nonstrict_load_state
        args.load_state_strict = False
    if args.load_full_state:
        args.load_state = args.load_full_state

    # add args.device
    args.device = torch.device('cpu')
    # if not args.disable_cuda and torch.cuda.is_available():
    #     args.device = torch.device('cuda')

    ## pretrained pool model (if any)
    pretrained_pool = None

    # create interaction/pooling modules
    pool = None
    if args.type == 'hiddenstatemlp':
        pool = HiddenStateMLPPooling(hidden_dim=args.hidden_dim, out_dim=args.pool_dim,
                                     mlp_dim_vel=args.vel_dim)
    elif args.type == 'nmmp':
        pool = NMMP(hidden_dim=args.hidden_dim, out_dim=args.pool_dim, k=args.mp_iters)
    elif args.type == 'attentionmlp':
        pool = AttentionMLPPooling(hidden_dim=args.hidden_dim, out_dim=args.pool_dim,
                                   mlp_dim_spatial=args.spatial_dim, mlp_dim_vel=args.vel_dim)
    elif args.type == 'directionalmlp':
        pool = DirectionalMLPPooling(out_dim=args.pool_dim)
    elif args.type == 'nn':
        pool = NN_Pooling(n=args.neigh, out_dim=args.pool_dim, no_vel=args.no_vel)
    elif args.type == 'nn_lstm':
        pool = NN_LSTM(n=args.neigh, hidden_dim=args.hidden_dim, out_dim=args.pool_dim)
    elif args.type == 'traj_pool':
        pool = TrajectronPooling(hidden_dim=args.hidden_dim, out_dim=args.pool_dim)
    elif args.type == 's_att_fast':
        pool = SAttention_fast(hidden_dim=args.hidden_dim, out_dim=args.pool_dim)
    elif args.type != 'vanilla':
        pool = GridBasedPooling(type_=args.type, hidden_dim=args.hidden_dim,
                                cell_side=args.cell_side, n=args.n, front=args.front,
                                out_dim=args.pool_dim, embedding_arch=args.embedding_arch,
                                constant=args.pool_constant, pretrained_pool_encoder=pretrained_pool,
                                norm=args.norm, layer_dims=args.layer_dims, latent_dim=args.latent_dim)

    # create forecasting model
    model = LSTM(pool=pool,
                 embedding_dim=args.coordinate_embedding_dim,
                 hidden_dim=args.hidden_dim,
                 goal_flag=args.goals,
                 goal_dim=args.goal_dim)
    
    # ------------- Social NCE ----------------    
    projection_head = ProjHead(feat_dim=args.hidden_dim, hidden_dim=args.contrast_dim*4, head_dim=args.contrast_dim)
    if args.contrast_sampling == 'single':
        encoder_sample = SpatialEncoder(hidden_dim=args.contrast_dim, head_dim=args.contrast_dim)
    elif args.contrast_sampling == 'multi':
        encoder_sample = EventEncoder(hidden_dim=args.contrast_dim, head_dim=args.contrast_dim)
    else:
        raise NotImplementedError

    param = list(model.parameters()) + list(projection_head.parameters()) + list(encoder_sample.parameters())

    # optimizer and schedular
    optimizer = torch.optim.Adam(param, lr=args.lr, weight_decay=1e-4)
    lr_scheduler = None
    if args.step_size is not None:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.step_size, args.scheduler_gamma)
    start_epoch = 0

    # Loss Criterion
    criterion = L2Loss() if args.loss == 'L2' else PredictionLoss()

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
            optimizer = torch.optim.Adam(param, lr=args.lr) # , weight_decay=1e-4
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 15)
            lr_scheduler.load_state_dict(checkpoint['scheduler'])
            start_epoch = checkpoint['epoch']

    #trainer
    trainer = Trainer(projection_head, encoder_sample,
                      args.contrast_weight, args.contrast_temperature, args.contrast_horizon, args.contrast_sampling,
                      model=model, optimizer=optimizer, lr_scheduler=lr_scheduler, device=args.device,
                      criterion=criterion, batch_size=args.batch_size, obs_length=args.obs_length,
                      pred_length=args.pred_length, augment=args.augment, normalize_scene=args.normalize_scene,
                      save_every=args.save_every, start_length=args.start_length, obs_dropout=args.obs_dropout,
                      augment_noise=args.augment_noise, col_weight=args.col_weight, col_gamma=args.col_gamma,
                      val_flag=val_flag)

    # ------------- Social NCE ----------------
    if args.contrast_pretrain > 0 and args.contrast_weight > 0:
        # freeze forecasting model parameters
        for param in model.parameters():
            param.requires_grad = False
        # pretrain contrastive heads
        for i in range(args.contrast_pretrain):
            trainer.train(train_scenes, train_goals, i-args.contrast_pretrain)
        # release forecasting model parameters
        for param in model.parameters():
            param.requires_grad = True

    # train
    trainer.loop(train_scenes, val_scenes, train_goals, val_goals, args.output, epochs=args.epochs, start_epoch=start_epoch)


if __name__ == '__main__':
    main()
