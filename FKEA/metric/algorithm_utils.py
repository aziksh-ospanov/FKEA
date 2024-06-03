import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.linalg import eigh, eigvalsh, eigvals
from torch.distributions import Categorical
from torchvision.utils import save_image
import os
from torchvision.transforms import ToTensor, Resize, Compose
from tqdm import tqdm

def normalized_gaussian_kernel(x, y, sigma, batchsize):
    '''
    calculate the kernel matrix, the shape of x and y should be equal except for the batch dimension

    x:      
        input, dim: [batch, dims]
    y:      
        input, dim: [batch, dims]
    sigma:  
        bandwidth parameter
    batchsize:
        Batchify the formation of kernel matrix, trade time for memory
        batchsize should be smaller than length of data

    return:
        scalar : mean of kernel values
    '''
    batch_num = (y.shape[0] // batchsize) + 1
    assert (x.shape[1:] == y.shape[1:])

    total_res = torch.zeros((x.shape[0], 0), device=x.device)
    for batchidx in range(batch_num):
        y_slice = y[batchidx*batchsize:min((batchidx+1)*batchsize, y.shape[0])]
        res = torch.norm(x.unsqueeze(1)-y_slice, dim=2, p=2).pow(2)
        res = torch.exp((- 1 / (2*sigma*sigma)) * res)
        total_res = torch.hstack([total_res, res])

        del res, y_slice

    total_res = total_res / np.sqrt(x.shape[0] * y.shape[0])

    return total_res


def visualize_txt_modes_by_eigenvectors_rff(test_feats, test_dataset, test_idxs, args):
    args.logger.info('Start compute covariance matrix')
    x_cov, _, x_feature = cov_rff(test_feats, args.rff_dim, args.sigma, args.batchsize, args.normalise)

    test_idxs = test_idxs.to(dtype=torch.long)

    args.logger.info('Start compute eigen-decomposition')
    eigenvalues, eigenvectors = torch.linalg.eigh(x_cov)
    eigenvalues = eigenvalues.real
    if args.normalise:
        eigenvalues = eigenvalues / torch.sum(eigenvalues)
    eigenvectors = eigenvectors.real

    transform = []
    if args.resize_img_to is not None:
        transform += [Resize((args.resize_img_to, args.resize_img_to))]
    transform += [ToTensor()]
    transform = Compose(transform)

    m, max_id = eigenvalues.topk(args.num_visual_mode)

    now_time = args.current_time

    for i in range(args.num_visual_mode):

        top_eigenvector = eigenvectors[:, max_id[i]]

        top_eigenvector = top_eigenvector.reshape((2*args.rff_dim, 1)) # [2 * feature_dim, 1]
        s_value = (x_feature @ top_eigenvector).squeeze() # [B, ]
        if s_value.sum() < 0:
            s_value = -s_value
            print("reverse with mode {}".format(i+1))
        topk_id = s_value.topk(args.num_txt_per_mode)[1]
        save_folder_name = os.path.join(args.path_save_visual, 'backbone_{}_norm_{}/{}_{}/'.format(args.backbone, args.normalise, args.visual_name, now_time), 'top{}'.format(i+1))
        os.makedirs(save_folder_name)
        summary = []

        for j, idx in enumerate(test_idxs[topk_id.cpu()]):
            sample = test_dataset[idx]
            summary.append(sample)

        with open(os.path.join(save_folder_name, 'modes.txt'), 'w', encoding='utf-8') as f:
            for s in summary:
                f.write(f'{s}\n{"-" * 50}\n')
    
def visualize_mode_by_eigenvectors_rff(test_feats, test_dataset, test_idxs, args):
    nrow = 3
    args.logger.info('Start compute covariance matrix')
    x_cov, _, x_feature = cov_rff(test_feats, args.rff_dim, args.sigma, args.batchsize, args.normalise)

    test_idxs = test_idxs.to(dtype=torch.long)

    args.logger.info('Start compute eigen-decomposition')
    eigenvalues, eigenvectors = torch.linalg.eigh(x_cov)
    
    eigenvalues = eigenvalues.real
    eigenvectors = eigenvectors.real
    args.logger.info(f'Sum of Eigenvalues: {torch.sum(eigenvalues)}')

    transform = []
    if args.resize_img_to is not None:
        transform += [Resize((args.resize_img_to, args.resize_img_to))]
    transform += [ToTensor()]
    transform = Compose(transform)
    
    # top eigenvalues
    m, max_id = eigenvalues.topk(args.num_visual_mode)

    now_time = args.current_time

    for i in range(args.num_visual_mode):

        top_eigenvector = eigenvectors[:, max_id[i]]

        top_eigenvector = top_eigenvector.reshape((2*args.rff_dim, 1)) # [2 * feature_dim, 1]
        s_value = (x_feature @ top_eigenvector).squeeze() # [B, ]
        if s_value.sum() < 0:
            s_value = -s_value
            print("reverse with mode {}".format(i+1))
        topk_id = s_value.topk(args.num_img_per_mode)[1]
        
        save_folder_name = os.path.join(args.path_save_visual, 'backbone_{}_norm_{}/{}_{}/'.format(args.backbone, args.normalise, args.visual_name, now_time), 'top{}'.format(i+1))
        os.makedirs(save_folder_name)
        summary = []

        for j, idx in enumerate(test_idxs[topk_id.cpu()]):
            top_imgs = transform(test_dataset[idx][0])
            summary.append(top_imgs)
            save_image(top_imgs, os.path.join(save_folder_name, '{}.png'.format(j)), nrow=1)
        
        save_image(summary[:int(nrow**2)], os.path.join(save_folder_name, f'summary_top{i+1}.png'.format(j)), nrow=nrow)


def cov_rff2(x, feature_dim, std, batchsize=16, presign_omeaga=None, normalise = True):
    assert len(x.shape) == 2 # [B, dim]

    x_dim = x.shape[-1]

    if presign_omeaga is None:
        omegas = torch.randn((x_dim, feature_dim), device=x.device) * (1 / std)
    else:
        omegas = presign_omeaga
    product = torch.matmul(x, omegas)
    batched_rff_cos = torch.cos(product) # [B, feature_dim]
    batched_rff_sin = torch.sin(product) # [B, feature_dim]

    batched_rff = torch.cat([batched_rff_cos, batched_rff_sin], dim=1) / np.sqrt(feature_dim) # [B, 2 * feature_dim]

    batched_rff = batched_rff.unsqueeze(2) # [B, 2 * feature_dim, 1]

    cov = torch.zeros((2 * feature_dim, 2 * feature_dim), device=x.device)
    batch_num = (x.shape[0] // batchsize) + 1
    i = 0
    for batchidx in tqdm(range(batch_num)):
        batched_rff_slice = batched_rff[batchidx*batchsize:min((batchidx+1)*batchsize, batched_rff.shape[0])] # [mini_B, 2 * feature_dim, 1]
        cov += torch.bmm(batched_rff_slice, batched_rff_slice.transpose(1, 2)).sum(dim=0)
        i += batched_rff_slice.shape[0]
    cov /= x.shape[0]
    assert i == x.shape[0]

    assert cov.shape[0] == cov.shape[1] == feature_dim * 2

    return cov, batched_rff.squeeze()

def cov_diff_rff(x, y, feature_dim, std, batchsize=16):
    assert len(x.shape) == len(y.shape) == 2 # [B, dim]

    B, D = x.shape
    x = x.to('cuda' if torch.cuda.is_available() else 'cpu')
    y = y.to('cuda' if torch.cuda.is_available() else 'cpu')

    omegas = torch.randn((D, feature_dim), device=x.device) * (1 / std)

    x_cov, x_feature = cov_rff2(x, feature_dim, std, batchsize=batchsize, presign_omeaga=omegas)
    y_cov, y_feature = cov_rff2(y, feature_dim, std, batchsize=batchsize, presign_omeaga=omegas)

    return x_cov, y_cov, omegas, x_feature, y_feature # [2 * feature_dim, 2 * feature_dim], [D, feature_dim], [B, 2 * feature_dim], [B, 2 * feature_dim]

def cov_rff(x, feature_dim, std, batchsize=16, normalise=True):
    assert len(x.shape) == 2 # [B, dim]

    x = x.to('cuda' if torch.cuda.is_available() else 'cpu')
    B, D = x.shape
    omegas = torch.randn((D, feature_dim), device=x.device) * (1 / std)

    x_cov, x_feature = cov_rff2(x, feature_dim, std, batchsize=batchsize, presign_omeaga=omegas, normalise=normalise)

    return x_cov, omegas, x_feature # [2 * feature_dim, 2 * feature_dim], [D, feature_dim], [B, 2 * feature_dim]

def non_fourier_scores(x_feats, args, alpha=2, t=100):
    K = normalized_gaussian_kernel(x_feats, x_feats, args.sigma, args.batchsize)
    
    eigenvalues, eigenvectors = torch.linalg.eigh(K)
    
    eigenvalues = eigenvalues.real
    eigenvectors = eigenvectors.real
    
    results = calculate_stats(eigenvalues, alpha)
    
    
    return results

def fourier_scores(test_feats, args, alpha=2):
    args.logger.info('Start compute covariance matrix')
    x_cov, _, x_feature = cov_rff(test_feats, args.rff_dim, args.sigma, args.batchsize, args.normalise)

    eigenvalues, eigenvectors = torch.linalg.eigh(x_cov)
    
    eigenvalues = eigenvalues.real
    eigenvectors = eigenvectors.real
    
    results = calculate_stats(eigenvalues, alpha)
    
    prefix = 'FKEA-'
    results = {prefix + key: value for key, value in results.items()}
    return results

def calculate_stats(eigenvalues, alphas=[2]):
    results = {}
    epsilon = 1e-10  # Small constant to avoid log of zero

    # Ensure eigenvalues are positive and handle alpha = 1 case
    eigenvalues = torch.clamp(eigenvalues, min=epsilon)
    
    
    
    if isinstance(alphas, (int, float, str)):
        alphas = [alphas]
    for alpha in alphas:
        if isinstance(alpha, str):
            assert alpha == 'inf'
    
    for alpha in alphas:
        if isinstance(alpha, (int, float)):
            if alpha != 1:
                entropy = (1 / (1-alpha)) * torch.log(torch.sum(eigenvalues**alpha))
                score = torch.exp(entropy)
            else:
                log_eigenvalues = torch.log(eigenvalues)
                
                entanglement_entropy = -torch.sum(eigenvalues * log_eigenvalues)# * 100
                score = torch.exp(entanglement_entropy)
        else:
            if alpha == 'inf':
                score = 1 / torch.max(eigenvalues)
        
        if alpha == 2:
            results[f'RKE'] = np.around(score.item(), 2)
        else:
            results[f'VENDI-{alpha}'] = np.around(score.item(), 2)
    return results
    










