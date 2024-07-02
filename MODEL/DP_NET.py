import torch.nn as nn


class Degradation_Predictor_patch(nn.Module):
    def __init__(self, in_nc=3, nf=64, num_params=64, use_bias=True):
        super(Degradation_Predictor_patch, self).__init__()

        self.ConvNet = nn.Sequential(*[
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_nc, nf, kernel_size=4, stride=2, padding=0),
            nn.PReLU(), # 48->24

            nn.ReflectionPad2d(2),
            nn.Conv2d(nf, nf, kernel_size=5, stride=1, padding=0, bias=use_bias),
            nn.PReLU(), # 24->24

            nn.ReflectionPad2d(1),
            nn.Conv2d(nf, 2 * nf, kernel_size=4, stride=2, padding=0, bias=use_bias),
            nn.PReLU(), # 24->12

            nn.ReflectionPad2d(2),
            nn.Conv2d(2 * nf, 2 * nf, kernel_size=5, stride=1, padding=0, bias=use_bias),
            nn.PReLU(), # 12->12

            nn.ReflectionPad2d(1),
            nn.Conv2d(2 * nf, 4 * nf, kernel_size=4, stride=2, padding=0, bias=use_bias),
            nn.PReLU(), # 12->6

            nn.ReflectionPad2d(2),
            nn.Conv2d(4 * nf, num_params, kernel_size=5, stride=1, padding=0, bias=use_bias),
            nn.PReLU(), # 6->6
        ])

        self.globalPooling = nn.AdaptiveAvgPool2d((1, 1))



    def forward(self, input):
        conv = self.ConvNet(input)
        # print(conv.shape)
        flat = self.globalPooling(conv)
        # print(flat.shape)
        out_params = flat.view(flat.size()[:2])
        return out_params



