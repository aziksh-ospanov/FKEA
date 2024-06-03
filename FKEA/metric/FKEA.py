import torch
import pandas as pd
from argparse import ArgumentParser, Namespace
from .algorithm_utils import *
from os.path import join
from FKEA.features.CLIPFeatureExtractor import CLIPFeatureExtractor
from FKEA.features.SWAVFeatureExtractor import SWAVFeatureExtractor
from FKEA.features.DINOv2FeatureExtractor import DINOv2FeatureExtractor
from FKEA.features.InceptionFeatureExtractor import InceptionFeatureExtractor
from FKEA.features.PixelFeatureExtractor import MonoPixelFeatureExtractor, ColorPixelFeatureExtractor
from FKEA.features.BERTFeatureExtractor import BERTFeatureExtractor
from FKEA.features.GPTFeatureExtractor import GPTFeatureExtractor

import time
import logging
import sys

def get_logger(filepath='./logs/diversity.log'):
    '''
        Information Module:
            Save the program execution information to a log file and output to the terminal at the same time
    '''

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.DEBUG)
    stdout_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(filepath)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stdout_handler)

    return logger

class FKEA_Evaluator():
    def __init__(self, logger_path : str, sigma : float, result_name: str, num_samples: int = 5000, batchsize: int = 128, rff_dim: int = 0, normalise: bool = False,
                 api_key = 'your_api_key'):
        self.logger_path = logger_path
        self.sigma = sigma
        self.num_samples = num_samples
        self.batchsize = batchsize
        self.rff_dim = rff_dim
        self.normalise = normalise

        self.current_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        self.result_name = '{}_num_{}_sigma_{}'.format(result_name, num_samples, sigma)
        self.save_feats_name = '{}_num_{}'.format(result_name, num_samples)


        self.feature_extractor = None
        self.name_feature_extractor = None
        self.running_logger = None

        self.init_running_logger()
        self.running_logger.info("FKEA Evaluator Initialized.")
        
        self.api_key = api_key
    
    def init_running_logger(self):
        self.running_logger = get_logger(join(self.logger_path, 'run_{}_{}.log'.format(self.result_name, self.current_time)))

    
    def set_feature_extractor(self, name: str, save_path=None):
        if name.lower() == 'inception':
            self.feature_extractor = InceptionFeatureExtractor(save_path, logger=self.running_logger)
        elif name.lower() == 'dinov2': # CHANGE BACK LATER
            self.feature_extractor = DINOv2FeatureExtractor(save_path, logger=self.running_logger)
        elif name.lower() == 'clip':
            self.feature_extractor = CLIPFeatureExtractor(save_path, logger=self.running_logger)
        elif name.lower() == 'swav':
            self.feature_extractor = SWAVFeatureExtractor(save_path, logger=self.running_logger)    
        elif name.lower() =='colored_pixel':
            self.feature_extractor = ColorPixelFeatureExtractor(save_path, logger=self.running_logger)
        elif name.lower() =='mono_pixel':
            self.feature_extractor = MonoPixelFeatureExtractor(save_path, logger=self.running_logger)
        elif name.lower() == 'bert':
            self.feature_extractor = BERTFeatureExtractor(save_path, logger=self.running_logger)
        elif name.lower() == 'gpt-large':
            self.feature_extractor = GPTFeatureExtractor(save_path, logger=self.running_logger, API_KEY=self.api_key)
        else:
            raise NotImplementedError(
                f"Cannot get image feature extractor '{name}'. Expected one of ['inception', 'dinov2', 'clip', 'swav', 'mono_pixel', 'colored_pixel']"
                f"Cannot get text feature extractor '{name}'. Expected one of ['bert', 'gpt_large']"
            )
        self.name_feature_extractor = name.lower()
        self.running_logger.info("Initialized feature-extractor network: {}".format(self.name_feature_extractor))
    
    def rff_clustering_modes_of_dataset(self,
                                        test_dataset: torch.utils.data.Dataset):
        
        assert self.rff_dim > 0
        
        args = Namespace(num_samples=self.num_samples, 
                         batchsize=self.batchsize, 
                         sigma=self.sigma, 
                         rff_dim=self.rff_dim,
                         logger=self.running_logger,
                         backbone=self.name_feature_extractor,
                         visual_name=self.result_name,
                         current_time=self.current_time,
                         path_save_visual='./visuals/modes_rff',
                         num_visual_mode=10,
                         num_img_per_mode=50,
                         resize_img_to=224,
                         normalise = self.normalise
        )
        
        self.running_logger.info("Running RFF approximation with dim: {}x2".format(args.rff_dim))
        self.running_logger.info("Num_samples_per_distribution: {}, Sigma: {}".format(args.num_samples, args.sigma))
        self.running_logger.info('test dataset length: {}'.format(len(test_dataset)))

        if self.feature_extractor is None:
            self.running_logger.info("Feature extractor is not specified, use default Inception-V3.")
            self.set_feature_extractor(name='inception', logger=self.running_logger)
        
        with torch.no_grad():
            self.running_logger.info("Calculating test feats:")
            test_feats, test_idxs = self.feature_extractor.get_features_and_idxes(test_dataset, 
                                                                    name = 'test_' + self.save_feats_name, 
                                                                    recompute=False, 
                                                                    num_samples=args.num_samples, 
                                                                    batchsize=args.batchsize)
        
        self.running_logger.info("number of test feature: {}".format(len(test_feats)))
        visualize_mode_by_eigenvectors_rff(test_feats, test_dataset, test_idxs, args)
        
        
    def rff_text_clustering_modes_of_dataset(self, test_dataset, test_feats = None):
        
        args = Namespace(num_samples=self.num_samples, 
                         batchsize=self.batchsize, 
                         sigma=self.sigma, 
                         rff_dim=self.rff_dim,
                         logger=self.running_logger,
                         backbone=self.name_feature_extractor,
                         visual_name=self.result_name,
                         current_time=self.current_time,
                         path_save_visual='./visuals/modes_rff',
                         num_visual_mode=10,
                         num_txt_per_mode=100,
                         resize_img_to=224,
                         normalise = self.normalise
        )
        
        self.running_logger.info("Running RFF approximation with dim: {}x2".format(args.rff_dim))
        self.running_logger.info("Num_samples_per_distribution: {}, Sigma: {}".format(args.num_samples, args.sigma))
        self.running_logger.info('test dataset length: {}'.format(len(test_dataset)))
        
        if self.feature_extractor is None:
            self.running_logger.info("Feature extractor is not specified, use default BERT.")
            self.set_feature_extractor(name='bert', logger=self.running_logger)
        
        if test_feats is None:
            with torch.no_grad():
                self.running_logger.info("Calculating test feats:")
                test_feats, test_texts = self.feature_extractor.get_features_and_idxes(test_dataset, 
                                                                        name = 'test_' + self.save_feats_name, 
                                                                        recompute=False, 
                                                                        num_samples=args.num_samples, 
                                                                        batchsize=args.batchsize)
        else:
            test_texts = test_dataset
        
        
        test_idxs = torch.arange(0, args.num_samples)
        test_feats = torch.Tensor(test_feats)
        
        visualize_txt_modes_by_eigenvectors_rff(test_feats, test_texts, test_idxs, args)
    
    def f_diversity_scores(self,
                           test_dataset: torch.utils.data.Dataset, 
                           alpha=2,
                           test_feats=None):
        assert self.rff_dim > 0
        
        args = Namespace(num_samples=self.num_samples, 
                         batchsize=self.batchsize, 
                         sigma=self.sigma, 
                         rff_dim=self.rff_dim,
                         logger=self.running_logger,
                         backbone=self.name_feature_extractor,
                         visual_name=self.result_name,
                         current_time=self.current_time,
                         path_save_visual='./visuals/modes_rff',
                         num_visual_mode=10,
                         num_img_per_mode=25,
                         resize_img_to=224,
                         normalise = self.normalise
        )
        
        self.running_logger.info("Running RFF approximation with dim: {}x2".format(args.rff_dim))
        self.running_logger.info("Num_samples_per_distribution: {}, Sigma: {}".format(args.num_samples, args.sigma))
        self.running_logger.info('test dataset length: {}'.format(len(test_dataset)))

        if self.feature_extractor is None:
            self.running_logger.info("Feature extractor is not specified, use default Inception-V3.")
            self.set_feature_extractor(name='inception', logger=self.running_logger)
        
        if test_feats is None:
            with torch.no_grad():
                self.running_logger.info("Calculating test feats:")
                test_feats, test_idxs = self.feature_extractor.get_features_and_idxes(test_dataset, 
                                                                        name = 'test_' + self.save_feats_name, 
                                                                        recompute=False, 
                                                                        num_samples=args.num_samples, 
                                                                        batchsize=args.batchsize)
        
        self.running_logger.info("number of test feature: {}".format(len(test_feats)))
        
        test_feats = torch.tensor(test_feats).float()
        
        f_scores = fourier_scores(test_feats, args, alpha)
        return f_scores
    
    def non_f_diversity_scores(self,
                           test_dataset: torch.utils.data.Dataset, 
                           alpha=2,
                           test_feats=None):
        assert self.rff_dim > 0
        
        args = Namespace(num_samples=self.num_samples, 
                         batchsize=self.batchsize, 
                         sigma=self.sigma, 
                         rff_dim=self.rff_dim,
                         logger=self.running_logger,
                         backbone=self.name_feature_extractor,
                         visual_name=self.result_name,
                         current_time=self.current_time,
                         path_save_visual='./visuals/modes_rff',
                         num_visual_mode=10,
                         num_img_per_mode=25,
                         resize_img_to=224,
                         normalise = self.normalise
        )
        
        self.running_logger.info("Running RFF approximation with dim: {}x2".format(args.rff_dim))
        self.running_logger.info("Num_samples_per_distribution: {}, Sigma: {}".format(args.num_samples, args.sigma))
        self.running_logger.info('test dataset length: {}'.format(len(test_dataset)))

        if self.feature_extractor is None:
            self.running_logger.info("Feature extractor is not specified, use default Inception-V3.")
            self.set_feature_extractor(name='inception', logger=self.running_logger)
        
        if test_feats is None:
            with torch.no_grad():
                self.running_logger.info("Calculating test feats:")
                test_feats, test_idxs = self.feature_extractor.get_features_and_idxes(test_dataset, 
                                                                        name = 'test_' + self.save_feats_name, 
                                                                        recompute=False, 
                                                                        num_samples=args.num_samples, 
                                                                        batchsize=args.batchsize)
        
        self.running_logger.info("number of test feature: {}".format(len(test_feats)))
        test_feats = torch.tensor(test_feats).float()
        scores = non_fourier_scores(test_feats, args, alpha)
        return scores
            
        
            
            
        
        
        
        
        
        
        
        

        



