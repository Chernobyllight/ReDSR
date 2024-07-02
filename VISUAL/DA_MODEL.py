import os

import torch.nn as nn
import torch
import torch.optim as optim

# generator
from MODEL.DP_NET import Degradation_Predictor_patch
from MODEL.REBLUR_BRANCH import G_img_samescale
from MODEL.SR_NET import DASR

from LOSS.ED_LOSS import ed_loss


class DANet(nn.Module):
    def __init__(self, opt):
        super(DANet, self).__init__()

        print('DASR')

        self.opt = opt

        opt_DP = opt['Degradation_Predictor']
        DP_MODEL = Degradation_Predictor_patch(in_nc=opt_DP['in_nc'], nf=opt_DP['nf'], num_params=opt_DP['num_params'], use_bias=opt_DP['use_bias'])
        print('# Degradation predictor parameters:', sum(param.numel() for param in DP_MODEL.parameters()), '\n')

        opt_RB = opt['Reblur_Branch']
        RB_MODEL = G_img_samescale(img_channel=opt_RB['img_channel'],output_img_channel=opt_RB['output_img_channel'],
                                   nc=opt_RB['nc'], nz=opt_RB['nz'],scale=opt_RB['scale'])
        print('# Reblur Branch parameters:', sum(param.numel() for param in RB_MODEL.parameters()), '\n')

        opt_SRN = opt['DASR']
        SRN_MODEL = DASR(scale=opt_SRN['scale'],rgb_range=opt_SRN['rgb_range'])
        print('# SRN parameters:', sum(param.numel() for param in SRN_MODEL.parameters()), '\n')


        if torch.cuda.is_available():
            self.DP_MODEL = DP_MODEL.cuda()
            self.RB_MODEL = RB_MODEL.cuda()
            self.SRN_MODEL = SRN_MODEL.cuda()

        self.optimizer_DP = optim.Adam(self.DP_MODEL.parameters(),
                                                        )
        self.optimizer_RB = optim.Adam(self.RB_MODEL.parameters(),
                                       )
        self.optimizer_SRN = optim.Adam(self.SRN_MODEL.parameters(),
                                       )

        self.criterionL1 = torch.nn.L1Loss()
        self.ed_loss_module = ed_loss(64)
        self.loss_sr = torch.tensor(0.0)
        self.focal_weight = torch.tensor(0.0)

    def Module_train(self):
        self.DP_MODEL.train()
        self.RB_MODEL.train()
        self.SRN_MODEL.train()

    def Module_eval(self):
        self.DP_MODEL.eval()
        self.RB_MODEL.eval()
        self.SRN_MODEL.eval()


    def model_save(self, path,epoch,save_path):
        state_dict = {
            'epoch': epoch,
            'model_DP_state_dict': self.DP_MODEL.state_dict(),
            'model_RB_state_dict': self.RB_MODEL.state_dict(),
            'model_SRN_state_dict': self.SRN_MODEL.state_dict(),
        }
        torch.save(state_dict,  path)
        path = os.path.join(save_path, 'checkpoints', 'final_epoch.tar')
        torch.save(state_dict, path)



    def feed_data(self,
            lr1,
            hr1,
            lr2,
            hr2
                  ):
        self.lr1 = lr1
        self.hr1 = hr1

        self.lr2 = lr2
        self.hr2 = hr2




    def forward(self):
        self.latent = self.DP_MODEL(self.lr1)
        self.c2blur = self.RB_MODEL([self.hr2,self.latent])
        self.SR = self.SRN_MODEL.forward(self.lr1,self.latent)


    def forward1(self):
        self.latent = self.DP_MODEL(self.lr1)
        self.c2blur = self.RB_MODEL([self.hr2,self.latent])




    def update_EG(self):
        self.optimizer_DP.zero_grad()
        self.optimizer_RB.zero_grad()
        self.optimizer_SRN.zero_grad()
        self.backward_EG()
        self.optimizer_DP.step()
        self.optimizer_RB.step()
        self.optimizer_SRN.step()


    def backward_EG(self):

        loss_kld = self.ed_loss_module(self.latent) * 0.01

        loss_c2_blur = self.criterionL1(self.c2blur,self.lr2) * 1.0

        if self.opt['use_focal_weight']:
            mse = torch.mean(torch.mean(
                torch.mean(
                    torch.square(self.lr2-self.c2blur),
                            dim=-1),
                        dim=-1),
                    dim=-1
                )
            psnr = torch.mean(
                10 * torch.log10(1 * 1 / mse)
                              )
            weight = 1.0 + 1 / (1 + torch.exp((psnr - 32.5) / 2.27))
            # weight = 1.0 + 1 / (1 + torch.exp((psnr - 30) / 2.27))
            weight = weight.detach()
            self.focal_weight = weight
            # print('weight:',weight)
            # print('psnr:',psnr)
            loss_sr = self.criterionL1(self.SR, self.hr1) * weight
        else:
            loss_sr = self.criterionL1(self.SR,self.hr1) * 2.0

        self.loss_kld = loss_kld
        self.loss_c2_blur = loss_c2_blur
        self.loss_sr = loss_sr

        loss_g_total = (loss_kld +
                        loss_c2_blur
                        + loss_sr)

        self.loss_g_total = loss_g_total

        loss_g_total.backward()



    def update_Degradation_branch_alone(self):
        self.optimizer_DP.zero_grad()
        self.optimizer_RB.zero_grad()
        self.backward_EG_alone()
        self.optimizer_DP.step()
        self.optimizer_RB.step()

    def backward_EG_alone(self):
        loss_kld = self.ed_loss_module(self.latent) * 0.01
        loss_c2_blur = self.criterionL1(self.c2blur, self.lr2) * 1.0

        self.loss_c2_blur = loss_c2_blur
        self.loss_kld = loss_kld

        loss_g_total = (
                    loss_kld +
                    loss_c2_blur)

        self.loss_g_total = loss_g_total

        loss_g_total.backward()




    def inference(self,lr):
        latent = self.DP_MODEL(lr)

        # sr = self.SRN_MODEL(lr,latent)

        return latent


