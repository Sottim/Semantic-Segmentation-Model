import torch 
import torch.nn as nn

#OOPs : nn_Module is the parent class while conv_block is sub-class(child) of Pytorch's nn.Module
class conv_block(nn.Module):
    def __init__(self, in_c, out_c): #Constructor
        super().__init__()

        #Defining the set of layers
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1), #The first out_c is the output from the previous BatchNormalization
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )
    #Calls the set of layers defined above with input x
    def forward(self, x):
        return self.conv(x)


class encoder_block(nn.Module):
    def __init__(self, in_c, out_c): #Constructor
        super().__init__()

        self.conv = conv_block(in_c, out_c)
        self.pool = nn.MaxPool2d(2,2)

    def forward(self, x):
        s = self.conv(x) #Output of convolutional layer for x : s is the skip connection. 
        p = self.pool(s) #The skip connection 's' is the input to the pooling layer. 
        return s, p
    
    """Note: 1) The skip connection from here goes to the Attention block then to the decorder part.
             2) The output of the pooling layer i.e 'p' goes as input to the next encoder block.    """
    
class attention_gate(nn.Module):
    def __init__(self, in_c, out_c): #Here in_c is a list of two elements: attention gate and skip connection
        super().__init__()

        #First part of the list contains: 
        self.Wg = nn.Sequential(
            nn.Conv2d(in_c[0], out_c, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_c)
        )

        self.Ws = nn.Sequential(
            nn.Conv2d(in_c[1], out_c, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_c)
        )

        self.relu = nn.ReLU(inplace=True)

        self.output = nn.Sequential(
            nn.Conv2d(out_c, out_c, kernel_size=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, g, s):
        Wg = self.Wg(g)
        Ws = self.Ws(s)
        out = self.relu(Wg + Ws)
        out = self.output(out)
        return out

class decorder_block(nn.Module):
    def __init__(self, in_c, out_c): #Here also in_c is a list with two inputs
        super().__init__()
        #Note: Refer to the Images included for understanding the approch of each block/layer 
        """Input to the decorder block means: the output from the previous layer. 
        So now that input is first upsampled using bilinar interpolation."""
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.ag = attention_gate(in_c, out_c) #pass the input and output to the attention gate
        self.c1 = conv_block(in_c[0]+out_c, out_c) #Here input:in_c[0]+out_c represents in_c[0] means no. of 
        #input channels from the upsampled image, and out_c comes from the skip conncection. Note in_c is a list here.

    
    def forward(self, x, s):
        x = self.up(x)
        s = self.ag(x, s) #attention gate receives two inputs: one is the image and other is skip conncetion output
        x = torch.cat([x, s], axis=1) #During concatination, H and W should be same. 
        x = self.c1(x) #
        return x

class attention_unet(nn.Module):
    def __init__(self):
        super().__init__()

        """Assumption: we are building a 2D segmentation for a binary class i.e final output will be 0/1. """
        self.e1 = encoder_block(3, 64) #input=RGB image, output_channel=64
        self.e2 = encoder_block(64, 128)
        self.e3 = encoder_block(128, 256)

        #Bridge block
        self.b1 = conv_block(256, 512)

        #decorder block
        self.d1 = decorder_block([512, 256], 256)
        self.d2 = decorder_block([256, 128], 128)
        self.d3 = decorder_block([128, 64], 64)

        self.output = nn.Conv2d(64, 1, kernel_size=1, padding=0)

    def forward(self, x):
        s1, p1 = self.e1(x)
        #print(s1.shape, p1.shape) #Output: torch.Size([8, 64, 256, 256]) torch.Size([8, 64, 128, 128])
        """The above output shows that there will be same number of channels when the image is first passed to 
        # first encoder block. The pooling layer of the first encoder block reduces H*W by 2 i.e from 256 to 128."""

        s2, p2 = self.e2(p1)
        #print(s2.shape, p2.shape) #Output: torch.Size([8, 128, 128, 128]) torch.Size([8, 128, 64, 64])

        s3, p3 = self.e3(p2)
        #print(s3.shape, p3.shape) #Output: torch.Size([8, 256, 64, 64]) torch.Size([8, 256, 32, 32])

        b1 = self.b1(p3)
        #print(b1.shape) #Output: torch.Size([8, 512, 32, 32])


        """Now we go for the decorder block"""
        d1 = self.d1(b1, s3)
        d2 = self.d2(d1, s2)
        d3 = self.d3(d2, s1)

        output = self.output(d3)

        return output


if __name__ == "__main__":
    x = torch.randn((8, 3, 256, 256))
    model = attention_unet()
    y = model(x)
    print(y.shape)
    
    

