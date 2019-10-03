import torch
import torch.nn as nn
import torch.nn.functional as F

class InverseNet(nn.Module):

    def __int__(self):

        super(InverseNet, self).__init__()
        self.Conv1 = nn.Conv3d(32,32,2,2)
        self.Upsample = nn.ConvTranspose3d(32,32,2,2)
        ## Defining DownSampling Layers ##

        self.l_relu = nn.LeakyReLU(negative_slope=0.2)


    def createEncoder(self,src,tgt):

        x = torch.cat([src,tgt],1)
        self.Encoder_layer0 = nn.Conv3d(2,32,2,2)(x)
        self.Encoder_layer1 = nn.Conv3d(32, 32, 2, 2)(self.Encoder_layer0)
        self.Encoder_layer2 = nn.Conv3d(32, 32, 2, 2)(self.Encoder_layer1)
        self.Encoder_layer3 = nn.Conv3d(32, 32, 2, 2)(self.Encoder_layer2)

        return self.Encoder_layer3

    def ForwardFlow(self,x):

        x = nn.ConvTranspose3d(32,32,2,2)(x)
        x = x + self.Encoder_layer2
        x = torch.cat([x , self.Encoder_layer2],1)

        x = nn.ConvTranspose3d(64,32,2,2)(x)
        x = x + self.Encoder_layer1
        x = torch.cat([x, self.Encoder_layer1], 1)

        x=  nn.ConvTranspose3d(64,32,2,2)(x)
        x = x + self.Encoder_layer0
        x = torch.cat([x, self.Encoder_layer0], 1)

        x = nn.ConvTranspose3d(64,32,2,2)(x)

        x = nn.Conv3d(32,8,1,1)(x)

        x =  nn.Conv3d(8,3,1,1)(x)
        return x

    def InverseFlow(self,x):

        x = nn.ConvTranspose3d(32, 32, 2, 2)(x)
        x = x - self.Encoder_layer2
        x = torch.cat([x, self.Encoder_layer2], 1)

        x = nn.ConvTranspose3d(64, 32, 2, 2)(x)
        x = x - self.Encoder_layer1
        x = torch.cat([x, self.Encoder_layer1], 1)

        x = nn.ConvTranspose3d(64, 32, 2, 2)(x)
        x = x - self.Encoder_layer0
        x = torch.cat([x, self.Encoder_layer0], 1)

        x = nn.ConvTranspose3d(64, 32, 2, 2)(x)

        x = nn.Conv3d(32, 8, 1, 1)(x)

        x = nn.Conv3d(8, 3, 1, 1)(x)

        return x


    def forward(self,src,tgt):

        Enc = self.createEncoder(src,tgt)

        f_flow = self.ForwardFlow(Enc)

        i_flow = self.InverseFlow(Enc)
        return f_flow, i_flow


if __name__=="__main__":

    net =InverseNet()
    src = torch.randn(4,1,64,64,64)
    tgt = torch.randn(4, 1, 64, 64, 64)

    f_flow,i_flow = net(src,tgt)

    print (f.shape)



















