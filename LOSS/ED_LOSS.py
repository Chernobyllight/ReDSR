
import torch.nn as nn
import torch

class ed_loss(nn.Module):
    def __init__(self,sample_num=64):
        super(ed_loss, self).__init__()
        self.sample_num = sample_num


    def forward(self,latent):

        z = torch.randn((self.sample_num,latent.size()[1]),out=torch.cuda.FloatTensor(latent.size()))

        latent_interleave = latent.repeat_interleave(z.size()[0], dim=0)
        z_repeat = z.repeat(latent.size()[0], 1)
        a1 = torch.sum(torch.norm(latent_interleave - z_repeat, dim=1), dim=0) * (2.0 / (latent.size()[0] * z.size()[0])) * (1.0 / latent.size()[1])


        latent_interleave = latent.repeat_interleave(latent.size()[0], dim=0)
        latent_repeat = latent.repeat(latent.size()[0], 1)
        b1 = torch.sum(torch.norm(latent_interleave - latent_repeat, dim=1), dim=0) * (
                    1.0 / (latent.size()[0] * latent.size()[0])) * (1.0 / latent.size()[1])


        z_interleave = z.repeat_interleave(z.size()[0], dim=0)
        z_repeat = z.repeat(z.size()[0], 1)
        c1 = torch.sum(torch.norm(z_interleave - z_repeat, dim=1), dim=0) * (
                1.0 / (z.size()[0] * z.size()[0])) * (1.0 / z.size()[1])


        kld_loss = a1 - b1 - c1

        return kld_loss




