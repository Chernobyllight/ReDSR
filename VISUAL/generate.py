

import torch
from torch.utils.data import DataLoader


import yaml
import os
import cv2
import numpy as np
import math

from DATA.dataset import test_dataset
from VISUAL.DA_MODEL import DANet
from UTILS import utils, tsne_utils



curPath = os.path.abspath(os.path.dirname(__file__))
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def calc_psnr(sr, hr, scale, rgb_range, benchmark=False):
    diff = (sr - hr).data.div(rgb_range)
    if benchmark:
        shave = scale
        if diff.size(1) > 1:
            convert = diff.new(1, 3, 1, 1)
            convert[0, 0, 0, 0] = 65.738
            convert[0, 1, 0, 0] = 129.057
            convert[0, 2, 0, 0] = 25.064
            diff.mul_(convert).div_(256)
            diff = diff.sum(dim=1, keepdim=True)
    else:
        shave = scale + 6
    import math
    shave = math.ceil(shave)
    valid = diff[:, :, shave:-shave, shave:-shave]
    mse = valid.pow(2).mean()

    return -10 * math.log10(mse)






def calc_ssim(img1, img2, scale=2, benchmark=False):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if benchmark:
        border = math.ceil(scale)
    else:
        border = math.ceil(scale) + 6

    img1 = img1.data.squeeze().float().clamp(0, 255).round().cpu().numpy()
    img1 = np.transpose(img1, (1, 2, 0))
    img2 = img2.data.squeeze().cpu().numpy()
    img2 = np.transpose(img2, (1, 2, 0))

    img1_y = np.dot(img1, [65.738, 129.057, 25.064]) / 255.0 + 16.0
    img2_y = np.dot(img2, [65.738, 129.057, 25.064]) / 255.0 + 16.0
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    h, w = img1.shape[:2]
    img1_y = img1_y[border:h - border, border:w - border]
    img2_y = img2_y[border:h - border, border:w - border]

    if img1_y.ndim == 2:
        return ssim(img1_y, img2_y)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')


def ssim(img1, img2):
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

def crop_border(img_hr, scale):
    b, n, c, h, w = img_hr.size()

    img_hr = img_hr[:, :, :, :int(h//scale*scale), :int(w//scale*scale)]

    return img_hr





if __name__=='__main__':

    torch.cuda.empty_cache()

    with open('generate.yml', 'r') as stream:
        opt = yaml.load(stream, Loader=yaml.FullLoader)


    # fix random seeds
    torch.manual_seed(1)

    val_set = test_dataset(img_dir=opt['B100_DATA'])
    val_set2 = test_dataset(img_dir=opt['U100_DATA'])

    val_loader = DataLoader(dataset=val_set+val_set2, num_workers=1, batch_size=1, shuffle=False)




    DA_MODEL = DANet(opt).cuda()

    if opt['DANET_CHECKPOINT'] is not None:
        checkpoint = torch.load(opt['DANET_CHECKPOINT'])
        DA_MODEL.DP_MODEL.load_state_dict(checkpoint['model_DP_state_dict'])

        DA_MODEL.Module_eval()

    else:
        print('Use --checkpoint to define the SRN parameters used')
        exit()


    degrade1 = utils.SRMDPreprocessing(
        opt['scale'],
        kernel_size=opt['blur_kernel'],
        blur_type=opt['blur_type'],
        sig_min=opt['sig_min'],
        sig_max=opt['sig_max'],
        sig=0.5,
        lambda_min=['lambda_min'],
        lambda_max=opt['lambda_max'],
        noise=opt['noise']
    )

    degrade2 = utils.SRMDPreprocessing(
        opt['scale'],
        kernel_size=opt['blur_kernel'],
        blur_type=opt['blur_type'],
        sig_min=opt['sig_min'],
        sig_max=opt['sig_max'],
        sig=1.4, # 1.4
        lambda_min=['lambda_min'],
        lambda_max=opt['lambda_max'],
        noise=opt['noise']
    )
    degrade3 = utils.SRMDPreprocessing(
        opt['scale'],
        kernel_size=opt['blur_kernel'],
        blur_type=opt['blur_type'],
        sig_min=opt['sig_min'],
        sig_max=opt['sig_max'],
        sig=2.3,
        lambda_min=['lambda_min'],
        lambda_max=opt['lambda_max'],
        noise=opt['noise']
    )
    degrade4 = utils.SRMDPreprocessing(
        opt['scale'],
        kernel_size=opt['blur_kernel'],
        blur_type=opt['blur_type'],
        sig_min=opt['sig_min'],
        sig_max=opt['sig_max'],
        sig=3.2,
        lambda_min=['lambda_min'],
        lambda_max=opt['lambda_max'],
        noise=opt['noise']
    )
    degrade5 = utils.SRMDPreprocessing(
        opt['scale'],
        kernel_size=opt['blur_kernel'],
        blur_type=opt['blur_type'],
        sig_min=opt['sig_min'],
        sig_max=opt['sig_max'],
        sig=4.1,
        lambda_min=['lambda_min'],
        lambda_max=opt['lambda_max'],
        noise=opt['noise']
    )

    degrade6 = utils.SRMDPreprocessing(
        opt['scale'],
        kernel_size=opt['blur_kernel'],
        blur_type=opt['blur_type'],
        sig_min=opt['sig_min'],
        sig_max=opt['sig_max'],
        sig=5.0,
        lambda_min=['lambda_min'],
        lambda_max=opt['lambda_max'],
        noise=opt['noise']
    )

    latent_list = []
    label_list = []


    with torch.no_grad():

        eval_psnr_wlg = 0
        eval_ssim_wlg = 0

        cnt = 0


        for idx_img, (val_hr) in enumerate(val_loader):
            cnt += 1

            # DA_MODEL.module_to_CPU()
            val_hr = val_hr.cuda()

            val_hr = crop_border(val_hr, 4)

            val_lr1, _ = degrade1(val_hr, random=False)  # 1, c, h//4, w//4
            val_lr2, _ = degrade2(val_hr, random=False)  # 1, c, h//4, w//4
            val_lr3, _ = degrade3(val_hr, random=False)  # 1, c, h//4, w//4
            val_lr4, _ = degrade4(val_hr, random=False)  # 1, c, h//4, w//4
            val_lr5, _ = degrade5(val_hr, random=False)  # 1, c, h//4, w//4
            val_lr6, _ = degrade6(val_hr, random=False)  # 1, c, h//4, w//4


            latent1 = DA_MODEL.inference(val_lr1[:, 0, ...] / 255.0)
            latent2 = DA_MODEL.inference(val_lr2[:, 0, ...] / 255.0)
            latent3 = DA_MODEL.inference(val_lr3[:, 0, ...] / 255.0)
            latent4 = DA_MODEL.inference(val_lr4[:, 0, ...] / 255.0)
            latent5 = DA_MODEL.inference(val_lr5[:, 0, ...] / 255.0)
            latent6 = DA_MODEL.inference(val_lr6[:, 0, ...] / 255.0)

            latent_list.append(latent1.squeeze(dim=0).cpu().numpy())
            label_list.append(1)

            latent_list.append(latent2.squeeze(dim=0).cpu().numpy())
            label_list.append(2)

            latent_list.append(latent3.squeeze(dim=0).cpu().numpy())
            label_list.append(3)

            latent_list.append(latent4.squeeze(dim=0).cpu().numpy())
            label_list.append(4)

            latent_list.append(latent5.squeeze(dim=0).cpu().numpy())
            label_list.append(5)

            latent_list.append(latent6.squeeze(dim=0).cpu().numpy())
            label_list.append(6)

        latent = np.array(latent_list)
        label = np.array(label_list)

        np.savetxt('latent.txt', latent, fmt='%0.16f')
        np.savetxt('label.txt', label, fmt='%d')

        tsne_utils.main_tsne(latent, label)








