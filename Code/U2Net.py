import torch
import torch.nn as nn
from torchsummary import summary
from Blocks import RSU7, RSU6, RSU5, RSU4, RSU4F, Upsample


class U2Net(nn.Module):
    
    def __init__(self, in_c = 3, out_c = 1):
        
        super().__init__()
        
        self.enc_stage1 = RSU7(in_c, 16, 64)
        self.enc_stage2 = RSU6(64, 16, 64)
        self.enc_stage3 = RSU5(64, 16, 64)
        self.enc_stage4 = RSU4(64, 16, 64)
        self.enc_stage5 = RSU4F(64, 16, 64)
        self.enc_stage6 = RSU4F(64, 16, 64)
        
        self.dec_stage5 = RSU4F(128, 16, 64)
        self.dec_stage4 = RSU4(128, 16, 64)
        self.dec_stage3 = RSU5(128, 16, 64)
        self.dec_stage2 = RSU6(128, 16, 64)
        self.dec_stage1 = RSU7(128, 16, 64)
        
        self.maxpool = nn.MaxPool2d(2, 2, ceil_mode = True)
        self.side_conv = nn.Conv2d(64, out_c, kernel_size = 3, padding = 1)
        
        self.final_conv = nn.Conv2d(6 * out_c, out_c, kernel_size = 1)
        
    def forward(self, x):
        
        '''Encoder'''
        
        en1 = self.enc_stage1(x)
        en1_mp = self.maxpool(en1)
        
        en2 = self.enc_stage2(en1_mp)
        en2_mp = self.maxpool(en2)
        
        en3 = self.enc_stage3(en2_mp)
        en3_mp = self.maxpool(en3)
        
        en4 = self.enc_stage4(en3_mp)
        en4_mp = self.maxpool(en4)
        
        en5 = self.enc_stage5(en4_mp)
        en5_mp = self.maxpool(en5)
        
        en6 = self.enc_stage6(en5_mp)
        en6_up = Upsample(en6, en5)
        
        '''Decoder'''
        
        dec5 = self.dec_stage5(torch.cat((en6_up, en5), 1))
        dec5_up = Upsample(dec5, en4)
        
        dec4 = self.dec_stage4(torch.cat((dec5_up, en4), 1))
        dec4_up = Upsample(dec4, en3)
        
        dec3 = self.dec_stage3(torch.cat((dec4_up, en3), 1))
        dec3_up = Upsample(dec3, en2)
        
        dec2 = self.dec_stage2(torch.cat((dec3_up, en2), 1))
        dec2_up = Upsample(dec2, en1)
        
        dec1 = self.dec_stage1(torch.cat((dec2_up, en1), 1))
        
        '''Side Outputs'''
        
        s1 = self.side_conv(dec1)
        
        s2 = self.side_conv(dec2)
        s2 = Upsample(s2, s1)
        
        s3 = self.side_conv(dec3)
        s3 = Upsample(s3, s1)
        
        s4 = self.side_conv(dec4)
        s4 = Upsample(s4, s1)
        
        s5 = self.side_conv(dec5)
        s5 = Upsample(s5, s1)
        
        s6 = self.side_conv(en6)
        s6 = Upsample(s6, s1)
        
        '''Final layer'''
        
        out = self.final_conv(torch.cat((s1,s2,s3,s4,s5,s6), 1))
        
        return torch.sigmoid(out), torch.sigmoid(s1), torch.sigmoid(s2), torch.sigmoid(s3), torch.sigmoid(s4), torch.sigmoid(s5), torch.sigmoid(s6)
    
    
def test():
    
    model = U2Net()
    x = torch.randn((1, 3, 288, 288))
    out = model(x)
    print(summary(model, (3, 288, 288), device = 'cpu'))
    print(out[0].shape)
    

if __name__ == '__main__':
    
    test()