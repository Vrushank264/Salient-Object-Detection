# -*- coding: utf-8 -*-
"""
Created on Sat Oct 30 11:10:59 2021

@author: Admin
"""

import torch
import torch.nn as nn
import torch.nn.functional as fun
from torchsummary import summary


class CBR(nn.Module):
    
    """
        Conv2D + Batch-Normalization + ReLU

        Parameters
        ----------
        in_c : int
            number of input channels.
        out_c : int
            number of output channels.
        dilation_rate : int
            Dialation rate.

        Returns
        -------
        torch.Tensor 

    """
    
    
    def __init__(self, in_c, out_c, dilation_rate):
        
        super().__init__()
        
        self.conv = nn.Conv2d(in_c, out_c, kernel_size = 3, padding = dilation_rate, 
                              dilation = dilation_rate)
        self.bn = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace = True)
        
    def forward(self, x):
        
        out = self.relu(self.bn(self.conv(x)))
        return out
    

def Upsample(source, target):
    
    out = fun.interpolate(source, size = target.shape[2:], mode = 'bilinear')
    return out
           
    
class RSU7(nn.Module):
    
    def __init__(self, in_c, M, out_c):
        
        super().__init__()
        
        self.initial_layer = CBR(in_c, out_c, dilation_rate = 1)
        self.cbr1 = CBR(out_c, M, 1)
        self.cbr_encoder_block = CBR(M, M, 1)
        self.maxpool = nn.MaxPool2d(2, 2, ceil_mode = True)
        
        self.bottleneck = CBR(M, M, 2)
        
        self.cbr_decoder_block = CBR(2*M, M, 1)
        self.cbr_dec_out = CBR(2*M, out_c, 1)
        
    def forward(self, x):
        
        x_in = self.initial_layer(x)
        en1 = self.cbr1(x_in)
        en1_mp  = self.maxpool(en1)
        
        en2 = self.cbr_encoder_block(en1_mp)
        en2_mp = self.maxpool(en2)
        
        en3 = self.cbr_encoder_block(en2_mp)
        en3_mp = self.maxpool(en3)
        
        en4 = self.cbr_encoder_block(en3_mp)
        en4_mp = self.maxpool(en4)
        
        en5 = self.cbr_encoder_block(en4_mp)
        en5_mp = self.maxpool(en5)
        
        en6 = self.cbr_encoder_block(en5_mp)
        
        en7 = self.bottleneck(en6)
        
        dec6 = self.cbr_decoder_block(torch.cat((en7, en6), 1))
        dec6_up = Upsample(dec6, en5)
        
        dec5 = self.cbr_decoder_block(torch.cat((dec6_up, en5), 1))
        dec5_up = Upsample(dec5, en4)
        
        dec4 = self.cbr_decoder_block(torch.cat((dec5_up, en4), 1))
        dec4_up = Upsample(dec4, en3)
        
        dec3 = self.cbr_decoder_block(torch.cat((dec4_up, en3), 1))
        dec3_up = Upsample(dec3, en2)
        
        dec2 = self.cbr_decoder_block(torch.cat((dec3_up, en2), 1))
        dec2_up = Upsample(dec2, en1)
        
        dec1 = self.cbr_dec_out(torch.cat((dec2_up, en1), 1))
        
        return x_in + dec1
    
    
class RSU6(nn.Module):
    
    def __init__(self, in_c, M, out_c):
        
        super().__init__()
        
        self.initial_layer = CBR(in_c, out_c, 1)
        self.cbr1 = CBR(out_c, M, 1)
        self.encoder_block = CBR(M, M, 1)
        self.maxpool = nn.MaxPool2d(2, 2, ceil_mode = True)
        self.bottleneck = CBR(M, M, 2)
        
        self.decoder_block = CBR(M*2, M, 1)
        self.final_layer = CBR(M*2, out_c, 1)
    
    def forward(self, x):
        
        x_in = self.initial_layer(x)
        en1 = self.cbr1(x_in)
        en1_mp = self.maxpool(en1)
        
        en2 = self.encoder_block(en1_mp)
        en2_mp = self.maxpool(en2)
        
        en3 = self.encoder_block(en2_mp)
        en3_mp = self.maxpool(en3)
        
        en4 = self.encoder_block(en3_mp)
        en4_mp = self.maxpool(en4)
        
        en5 = self.encoder_block(en4_mp)
        
        en6 = self.bottleneck(en5)
        
        dec5 = self.decoder_block(torch.cat((en6,en5), 1))
        dec5_up = Upsample(dec5, en4)
        
        dec4 = self.decoder_block(torch.cat((dec5_up, en4), 1))
        dec4_up = Upsample(dec4, en3)
        
        dec3 = self.decoder_block(torch.cat((dec4_up, en3), 1))
        dec3_up = Upsample(dec3, en2)
        
        dec2 = self.decoder_block(torch.cat((dec3_up, en2), 1))
        dec2_up = Upsample(dec2, en1)
        
        dec1 = self.final_layer(torch.cat((dec2_up, en1), 1))
        
        return x_in + dec1
        

class RSU5(nn.Module):
    
    def __init__(self, in_c, M, out_c):
        
        super().__init__()
        
        self.initial_layer = CBR(in_c, out_c, 1)
        self.cbr1 = CBR(out_c, M, 1)
        self.encoder_block = CBR(M, M, 1)
        self.maxpool = nn.MaxPool2d(2, 2, ceil_mode = True)
        self.bottleneck = CBR(M, M, 2)
        
        self.decoder_block = CBR(M*2, M, 1)
        self.final_layer = CBR(M*2, out_c, 1)
        
    def forward(self, x):
        
        x_in = self.initial_layer(x)
        en1 = self.cbr1(x_in)
        en1_mp = self.maxpool(en1)
        
        en2 = self.encoder_block(en1_mp)
        en2_mp = self.maxpool(en2)
        
        en3 = self.encoder_block(en2_mp)
        en3_mp = self.maxpool(en3)
        
        en4 = self.encoder_block(en3_mp)
        
        en5 = self.bottleneck(en4)
        
        dec4 = self.decoder_block(torch.cat((en5, en4), 1))
        dec4_up = Upsample(dec4, en3)

        dec3 = self.decoder_block(torch.cat((dec4_up, en3), 1))
        dec3_up = Upsample(dec3, en2)

        dec2 = self.decoder_block(torch.cat((dec3_up, en2), 1))
        dec2_up = Upsample(dec2, en1)

        dec1 = self.final_layer(torch.cat((dec2_up, en1), 1))

        return x_in + dec1 


class RSU4(nn.Module):
    
    def __init__(self, in_c, M, out_c):
        
        super().__init__()
        self.initial_layer = CBR(in_c, out_c, 1)
        self.cbr1 = CBR(out_c, M, 1)
        self.encoder_block = CBR(M, M, 1)
        self.maxpool = nn.MaxPool2d(2, 2, ceil_mode = True)
        self.bottleneck = CBR(M, M, 2)
        
        self.decoder_block = CBR(M*2, M, 1)
        self.final_layer = CBR(M*2, out_c, 1)
        
    def forward(self, x):
        
        x_in = self.initial_layer(x)
        en1 = self.cbr1(x_in)
        en1_mp = self.maxpool(en1)
        
        en2 = self.encoder_block(en1_mp)
        en2_mp = self.maxpool(en2)
        
        en3 = self.encoder_block(en2_mp)
        
        en4 = self.bottleneck(en3)
        
        dec3 = self.decoder_block(torch.cat((en4, en3), 1))
        dec3_up = Upsample(dec3, en2)
        
        dec2 = self.decoder_block(torch.cat((dec3_up, en2), 1))
        dec2_up = Upsample(dec2, en1)
        
        dec1 = self.final_layer(torch.cat((dec2_up, en1), 1))
        
        return x_in + dec1


class RSU4F(nn.Module):
    
    def __init__(self, in_c, M, out_c):
        
        super().__init__()
        
        self.initial_layer = CBR(in_c, out_c, 1)
        self.e1 = CBR(out_c, M, 1)
        self.e2 = CBR(M, M, 2)
        self.e3 = CBR(M, M, 4)
        self.e4 = CBR(M, M, 8)
        
        self.d3 = CBR(M*2, M, 4)
        self.d2 = CBR(M*2, M, 2)
        self.final_layer = CBR(M*2, out_c, 1)
        
    def forward(self, x):
        
        x_in = self.initial_layer(x)
        en1 = self.e1(x_in)
        en2 = self.e2(en1)
        en3 = self.e3(en2)
        en4 = self.e4(en3)
        
        dec3 = self.d3(torch.cat((en4, en3), 1))
        dec2 = self.d2(torch.cat((dec3, en2), 1))
        dec1 = self.final_layer(torch.cat((dec2, en1), 1))
        
        return x_in + dec1
        
       
        
def test():
    
    model = RSU7(in_c = 3, M = 16, out_c = 64)
    print(summary(model, (3,128,128), device = 'cpu'))
    

if __name__ == '__main__':
    
    test()