
import torch.nn as nn
import torch.nn.functional as F
import torch



class G_img_samescale(nn.Module):
    def conv3x3(self, dim_in, dim_out, stride=1):
        return [nn.ReflectionPad2d(1), nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=stride)]
    def __init__(self, img_channel=3,output_img_channel=3, nc=64, nz=64,scale=4):
        super(G_img_samescale, self).__init__()

        self.nc = nc
        self.img_channel = img_channel
        self.output_img_channel = output_img_channel
        self.scale = scale
        self.nz = nz
        self.kernel_size = 3
        self.reduction = 4


        self.mlpA = nn.Sequential(
            nn.Linear(64, 64),
            nn.PReLU(),
            nn.Linear(64, self.nz * 3),
            nn.PReLU()
        )

        A_head = []
        A_head += self.conv3x3(dim_in=self.img_channel, dim_out=self.nc)
        A_head += [nn.PReLU()]
        self.A_head = nn.Sequential(*A_head)

        self.blk1_A = DA_block(nc=self.nc,reduction=self.reduction,nz=self.nz,kernel_size=self.kernel_size)
        self.blk2_A = DA_block(nc=self.nc,reduction=self.reduction,nz=self.nz,kernel_size=self.kernel_size)
        self.blk3_A = DA_block(nc=self.nc,reduction=self.reduction,nz=self.nz,kernel_size=self.kernel_size)

        A_tail = []
        A_tail += [nn.Conv2d(self.nc, 4*self.nc, kernel_size=3, stride=4,padding=0)]
        A_tail += [nn.PReLU()]
        A_tail += self.conv3x3(dim_in=4*self.nc, dim_out=2*self.nc)
        A_tail += [nn.PReLU()]
        A_tail += self.conv3x3(dim_in=2*self.nc, dim_out=1*self.nc)
        A_tail += [nn.PReLU()]
        A_tail += self.conv3x3(dim_in=self.nc, dim_out=self.output_img_channel)
        self.A_tail = nn.Sequential(*A_tail)

    def forward(self,x):
        embedding_z = self.mlpA(x[1])
        z1, z2, z3 = torch.split(embedding_z, self.nz, dim=1)
        z1, z2, z3 = z1.contiguous(), z2.contiguous(), z3.contiguous()

        out = self.A_head(x[0])
        out = self.blk1_A([out,z1])
        out = self.blk2_A([out,z2])
        out = self.blk3_A([out,z3])
        out_A = self.A_tail(out)


        return out_A



class DA_block(nn.Module):
    def conv3x3(self, dim_in, dim_out, stride=1):
        return nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=stride))

    def conv1x1(self, dim_in, dim_out):
        return nn.Conv2d(dim_in, dim_out, kernel_size=1, stride=1, padding=0)

    def __init__(self,nc,reduction,nz=64,kernel_size=3):
        super(DA_block, self).__init__()


        self.da_conv1 = DA_conv(channels_in=nc,channels_out=nc,kernel_size=kernel_size,
                                reduction=reduction,nz=nz)
        self.conv1 = nn.Sequential(
            self.conv3x3(nc, nc, stride=1),
        )
        self.da_conv2 = DA_conv(channels_in=nc,channels_out=nc,kernel_size=kernel_size,
                                reduction=reduction,nz=nz)
        self.conv2 = nn.Sequential(
            self.conv3x3(nc, nc, stride=1),
        )

        self.prelu1 = nn.PReLU()
        self.prelu2 = nn.PReLU()
        self.prelu3 = nn.PReLU()

    def forward(self,x):
        out = self.prelu1(self.da_conv1(x))
        out = self.prelu2(self.conv1(out))
        out = self.prelu3(self.da_conv2([out,x[1]]))
        out = self.conv2(out) + x[0]

        return out






class DA_conv(nn.Module):

    def conv3x3(self, dim_in, dim_out, stride=1):
        return nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=stride))

    def __init__(self,channels_in,channels_out,kernel_size,reduction,nz):
        super(DA_conv, self).__init__()
        self.channels_in = channels_in
        self.channels_out = channels_out
        self.kernel_size = kernel_size

        padding_method = []
        padding_method += [nn.ReflectionPad2d(padding=(self.kernel_size-1)//2)]
        self.padding_method = nn.Sequential(*padding_method)

        self.kernel = nn.Sequential(
            nn.Linear(nz,channels_in,bias=False),
            nn.PReLU(),
            nn.Linear(channels_in,channels_in * self.kernel_size * self.kernel_size,bias=False)
        )


        self.ca = CA_layer(nz, reduction)
        self.conv = self.conv3x3(channels_in, channels_out)

        self.kernel.apply(gaussian_weights_init)

        self.prelu = nn.PReLU()

    def forward(self,x):
        '''

        :param x:
        x[0]:b,c,h,w
        x[1]:b,nz
        :return:
        '''
        b,c,h,w = x[0].size()

        kernel = self.kernel(x[1]).view(-1,1,self.kernel_size,self.kernel_size)
        out = self.padding_method(x[0])
        out = self.prelu(F.conv2d(out.view(1,-1,h+2,w+2),kernel,groups=b*c,padding=0))
        out = self.conv(out.view(b,-1,h,w))

        out = out + self.ca(x)

        return out








class CA_layer(nn.Module):
    def __init__(self,nz,reduction):
        super(CA_layer, self).__init__()
        self.conv_du = nn.Sequential(
            nn.Conv2d(nz,nz//reduction,1,1,0,bias=False),
            nn.PReLU(),
            nn.Conv2d(nz//reduction,nz,1,1,0,bias=False),
            nn.Sigmoid()
        )


        self.conv_du.apply(gaussian_weights_init)

    def forward(self,x):
        att = self.conv_du(x[1][:, :, None, None])
        return x[0] * att





def gaussian_weights_init(m):
  classname = m.__class__.__name__
  if classname.find('Conv') != -1 and classname.find('Conv') == 0:
    m.weight.data.normal_(0.0, 0.02)





