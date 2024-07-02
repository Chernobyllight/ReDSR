

import torch
from torch.utils.data import DataLoader


import yaml
import os
import cv2
import numpy as np
import math

from DATA.dataset import test_dataset
from TRAIN.TRAIN_setting1.DA_MODEL import DANet
from UTILS import utils

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

    with open('test.yml', 'r') as stream:
        opt = yaml.load(stream, Loader=yaml.FullLoader)


    # fix random seeds
    torch.manual_seed(1)

    # prepare data and DataLoaders
    val_set = test_dataset(img_dir=opt['dataset']['val_clean_dir'])
    val_loader = DataLoader(dataset=val_set, num_workers=1, batch_size=1, shuffle=False)

    DA_MODEL = DANet(opt)

    if opt['DANET_CHECKPOINT'] is not None:
        checkpoint = torch.load(opt['DANET_CHECKPOINT'])
        DA_MODEL.DP_MODEL.load_state_dict(checkpoint['model_DP_state_dict'])
        DA_MODEL.SRN_MODEL.load_state_dict(checkpoint['model_SRN_state_dict'])

        DA_MODEL.Module_eval()

    else:
        print('Use --checkpoint to define the SRN parameters used')
        exit()

    degrade = utils.SRMDPreprocessing(
        opt['scale'],
        kernel_size=opt['blur_kernel'],
        blur_type=opt['blur_type'],
        sig_min=opt['sig_min'],
        sig_max=opt['sig_max'],
        sig=opt['sig'],
    )

    with torch.no_grad():

        eval_psnr_wlg = 0
        eval_ssim_wlg = 0

        cnt = 0


        for idx_img, (val_hr) in enumerate(val_loader):
            cnt += 1

            val_hr = val_hr.cuda()

            val_hr = crop_border(val_hr, 4)

            val_lr,_ = degrade(val_hr, random=False)  # 1, c, h//4, w//4


            val_hr = val_hr[:, 0, ...]
            val_lr = val_lr
            sr = DA_MODEL.inference(val_lr[:, 0, ...]/255.0) * 255.0


            sr = utils.quantize(sr, 255)
            val_hr = utils.quantize(val_hr, 255)

            psnr_each = calc_psnr(sr, val_hr, scale=4, rgb_range=255,benchmark=True)
            eval_psnr_wlg += psnr_each

            eval_ssim_wlg += calc_ssim(sr, val_hr, scale=4, benchmark=True)



            print('------')
            print('psnr each:',cnt,'-',psnr_each)
            print('------')

            if opt['saving_results']:

                sr_save_path = curPath + '/SR/'
                lr_save_path = curPath + '/LR/'
                if not os.path.exists(os.path.join(sr_save_path)):
                    os.makedirs(os.path.join(sr_save_path))

                if not os.path.exists(os.path.join(lr_save_path)):
                    os.makedirs(os.path.join(lr_save_path))


                srfile_save_path = sr_save_path + str(cnt) + '.png'
                sr = np.array(sr.squeeze(0).permute(1, 2, 0).data.cpu())
                sr = sr[:, :, [2, 1, 0]]
                cv2.imwrite(srfile_save_path,sr)
                #
                val_lr = val_lr[:, 0, ...]
                lrfile_save_path = lr_save_path + str(cnt) + '.png'
                val_lr = np.array(val_lr.squeeze(0).permute(1, 2, 0).data.cpu())
                val_lr = val_lr[:, :, [2, 1, 0]]
                cv2.imwrite(lrfile_save_path, val_lr)

        ave_val_psnr = eval_psnr_wlg / len(val_loader)
        ave_val_ssim = eval_ssim_wlg / len(val_loader)
        print('val psnr wlg:', ave_val_psnr)
        print('val ssim wlg:', ave_val_ssim)












