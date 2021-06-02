from .goals import goalSGANPredictor
from .sgan import SGAN

import torch
import argparse
import pickle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--goalModel_path', default=None,
                        help='path to goal Model')
    parser.add_argument('--SGANModel_path', default=None,
                        help='path to SGAN Model')
    parser.add_argument('--output', default='goalSGANPredictor.pkl',
                        help='path + name of output file')
    
    args = parser.parse_args()
    
    
    
    ## Loading goal model
    with open(args.goalModel_path, 'rb') as f:
        goalModel = torch.load(f)
        
    ## Loading SGAN model
    SGANModel = SGAN()
    with open(args.SGANModel_path, 'rb') as f:
        sganPredictor = torch.load(f)
    
    ## Creating Predictor
    predictor = goalSGANPredictor(goalModel, SGANModel)
    
    ## Saving Predictor
    with open(args.output, 'wb') as output:
        pickle.dump(predictor, output, pickle.HIGHEST_PROTOCOL)
    


if __name__ == '__main__':
    main()