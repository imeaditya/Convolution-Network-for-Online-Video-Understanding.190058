import torch
import torch.nn as nn

class Conv2D(nn.Module):
    """ conv -> batchnorm -> relu Standard convolution layer class."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(Conv2D, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out = self.conv(x)
        return out

class Conv3D(nn.Module):
    """ conv -> (batchnorm -> relu) Standard Convolutional layer """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, flag=False):
        super(Conv3D, self).__init__()

        if flag == True:
            self.conv = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding),
                nn.BatchNorm3d(out_channels),
                nn.ReLU(inplace=True)
            )
        else:
            self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        out = self.conv(x)
        return out

class BasicConv(nn.Module):
    """ First convolutional network in ECO's 2D Net module """

    def __init__(self):
        super(BasicConv, self).__init__()

        self.conv1 = nn.Sequential(
            Conv2D(3, 64, kernel_size=7, stride=2, padding=3),      # size 1/2
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),  # size 1/2
        )
        self.conv2 = Conv2D(64, 64, kernel_size=1, stride=1)
        self.conv3 = nn.Sequential(
            Conv2D(64, 192, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),  # size 1/2
        )

    def forward(self, x):
        c1 = self.conv1(x)    # [3, 224, 224] -> [64, 56, 56]
        c2 = self.conv2(c1)   # [64, 56, 56] -> [64, 56, 56]
        out = self.conv3(c2)  # [64, 56, 56] -> [192, 28, 28]
        return out

class Inception_A(nn.Module):
    """ The first Inception module in the ECO's 2D Net module """
    
    def __init__(self):
        super(Inception_A, self).__init__()

        self.inception1 = Conv2D(192, 64, kernel_size=1, stride=1)
        self.inception2 = nn.Sequential(
            Conv2D(192, 64, kernel_size=1, stride=1),
            Conv2D(64, 64, kernel_size=3, stride=1, padding=1),
        )
        self.inception3 = nn.Sequential(
            Conv2D(192, 64, kernel_size=1, stride=1),
            Conv2D(64, 96, kernel_size=3, stride=1, padding=1),
            Conv2D(96, 96, kernel_size=3, stride=1, padding=1),
        )
        self.inception4 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            Conv2D(192, 32, kernel_size=1, stride=1),
        )

    def forward(self, x):
        out1 = self.inception1(x)  # [192, 28, 28] -> [64, 28, 28]
        out2 = self.inception2(x)  # [192, 28, 28] -> [64, 28, 28]
        out3 = self.inception3(x)  # [192, 28, 28] -> [96, 28, 28]
        out4 = self.inception4(x)  # [192, 28, 28] -> [32, 28, 28]
        # Join in channels direction，shape: [64+64+96+32 = 256, 28, 28]
        out = torch.cat([out1, out2, out3, out4], dim=1)
        return out

class Inception_B(nn.Module):
    """ The second Inception module in the ECO's 2D Net module """

    def __init__(self):
        super(Inception_B, self).__init__()

        self.inception1 = Conv2D(256, 64, kernel_size=1, stride=1)
        self.inception2 = nn.Sequential(
            Conv2D(256, 64, kernel_size=1, stride=1),
            Conv2D(64, 96, kernel_size=3, stride=1, padding=1),
        )
        self.inception3 = nn.Sequential(
            Conv2D(256, 64, kernel_size=1, stride=1),
            Conv2D(64, 96, kernel_size=3, stride=1, padding=1),
            Conv2D(96, 96, kernel_size=3, stride=1, padding=1),
        )
        self.inception4 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            Conv2D(256, 64, kernel_size=1, stride=1),
        )

    def forward(self, x):
        out1 = self.inception1(x)  # [256, 28, 28] -> [64, 28, 28]
        out2 = self.inception2(x)  # [256, 28, 28] -> [96, 28, 28]
        out3 = self.inception3(x)  # [256, 28, 28] -> [96, 28, 28]
        out4 = self.inception4(x)  # [256, 28, 28] -> [64, 28, 28]
        # Join in the channels direction，shape: [64+96+96+64 = 320, 28, 28]
        out = torch.cat([out1, out2, out3, out4], dim=1)
        return out

class Inception_C(nn.Module):
    """ The second Inception module in the ECO's 2D Net module """

    def __init__(self):
        super(Inception_C, self).__init__()

        self.inception = nn.Sequential(
            Conv2D(320, 64, kernel_size=1, stride=1),
            Conv2D(64, 96, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        out = self.inception(x)  # [320, 28, 28] -> [96, 28, 28]
        return out

class ECO_2D(nn.Module):
    """ The entire 2D Net module of ECO that connects Basic Conv and Inception A-C """

    def __init__(self):
        super(ECO_2D, self).__init__()

        self.basic_conv = BasicConv()
        self.inception_a = Inception_A()
        self.inception_b = Inception_B()
        self.inception_c = Inception_C()

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor, torch.Size([batch_size, 3, 224, 224])
           input
        """
        
        out = self.basic_conv(x)
        out = self.inception_a(out)
        out = self.inception_b(out)
        out = self.inception_c(out)
        return out

class Resnet3D_1(nn.Module):
    """ The first ResNet module in the ECO's 3D Net module """

    def __init__(self):
        super(Resnet3D_1, self).__init__()

        self.conv1 = Conv3D(96, 128, kernel_size=3, stride=1, padding=1,
                            flag=False)
        
        self.res_1 = nn.Sequential(
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            Conv3D(128, 128, kernel_size=3, stride=1, padding=1, flag=True),
            Conv3D(128, 128, kernel_size=3, stride=1, padding=1, flag=False),
        )

        self.bn_relu = nn.Sequential(
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        residual = self.conv1(x)
        
        out = self.res_1(residual)

        out += residual  # Skip Connection
        
        out = self.bn_relu(out)

        return out

class Resnet3D_2(nn.Module):
    """ The second ResNet module in the ECO's 3D Net module"""

    def __init__(self):
        super(Resnet3D_2, self).__init__()

        self.res1 = nn.Sequential(
            Conv3D(128, 256, kernel_size=3, stride=2, padding=1, flag=True),
            Conv3D(256, 256, kernel_size=3, stride=1, padding=1, flag=False),
        )
        self.skip1 = Conv3D(128, 256, kernel_size=3, stride=2, padding=1)
        
        self.res2 = nn.Sequential(
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            Conv3D(256, 256, kernel_size=3, stride=1, padding=1, flag=True),
            Conv3D(256, 256, kernel_size=3, stride=1, padding=1, flag=False),
        )
        
        self.bn_relu = nn.Sequential(
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        res1 = self.res1(x)
        skip1 = self.skip1(x)
        out = res1 + skip1  # [128, 16, 28, 28] -> [256, 8, 14, 14]

        res2 = self.res2(out)
        skip2 = out
        out = res2 + skip2

        return out

class Resnet3D_3(nn.Module):
    """
    The third ResNet module in the ECO's 3D Net module. Layer structure is the same as ResNet 3D_2．
    """

    def __init__(self):
        super(Resnet3D_3, self).__init__()

        self.res1 = nn.Sequential(
            Conv3D(256, 512, kernel_size=3, stride=2, padding=1, flag=True),
            Conv3D(512, 512, kernel_size=3, stride=1, padding=1, flag=False),
        )
        self.skip1 = Conv3D(256, 512, kernel_size=3, stride=2, padding=1)
        
        self.res2 = nn.Sequential(
            nn.BatchNorm3d(512),
            nn.ReLU(inplace=True),
            Conv3D(512, 512, kernel_size=3, stride=1, padding=1, flag=True),
            Conv3D(512, 512, kernel_size=3, stride=1, padding=1, flag=False),
        )
        
        self.bn_relu = nn.Sequential(
            nn.BatchNorm3d(512),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        res1 = self.res1(x)
        skip1 = self.skip1(x)
        out = res1 + skip1  # [256, 8, 14, 14] -> [512, 4, 7, 7]

        res2 = self.res2(out)
        skip2 = out
        out = res2 + skip2

        return out

class ECO_3D(nn.Module):
    """ ECO's entire 3D Net module with ResNet3D 1-3 connected"""

    def __init__(self):
        super(ECO_3D, self).__init__()

        # ResNet3D
        self.res3d_1 = Resnet3D_1()
        self.res3d_2 = Resnet3D_2()
        self.res3d_3 = Resnet3D_3()

        # Global Average Pooling
        self.global_avg_pool = nn.AvgPool3d(
            kernel_size=(4, 7, 7), stride=1, padding=0)

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor, torch.Size([batch_size, frames, 96, 28, 28])
            
        """
        
        x = torch.transpose(x, 1, 2)  # [frames, C, H, W] -> [C, frames, H, W]
        out = self.res3d_1(x)
        out = self.res3d_2(out)
        out = self.res3d_3(out)
        out = self.global_avg_pool(out)

        #Resize tensor, [batch_size, 512, 1, 1, 1]-> [batch_size, 512]
        out = out.view(out.size()[0], out.size()[1])
        
        return out

class ECO_Lite(nn.Module):
    def __init__(self):
        super(ECO_Lite, self).__init__()

        self.eco_2d = ECO_2D()  # 2D Net Module
        self.eco_3d = ECO_3D()  # 3D Net Module
        
        # Fully connected layer for classification
        self.fc_final = nn.Linear(512, 400, bias=True)

    def forward(self, x):
        '''
        Inputs
        ----------
        x : torch.tensor, size = [batch_size, num_segments=16, 3, 224, 224]
        '''

        bs, ns, c, h, w = x.shape # Get the size of each dimension of the input
        _x = x.view(-1, c, h, w)  # Convert to input x size [bs * ns, c, h, w]
        out_2d = self.eco_2d(_x)  # [bs*ns, 3, 224, 224] -> [bs*ns, 96, 28, 28]
        
        # Convert 2D image tensor for 3D
        _out_2d = out_2d.view(-1, ns, 96, 28, 28)  
        out_3d = self.eco_3d(_out_2d)  # [bs, ns, 96, 28, 28] -> [bs, 512]

        out = self.fc_final(out_3d)  # [bs, 512] -> [bs, 400]

        return out

if __name__ == '__main__':
    batch_size = 1
    # (2D | 3D) Net input test tensor
    input_tensor_for2d = torch.randn(batch_size, 3, 224, 224)
    input_tensor_for3d = torch.randn(batch_size, 16, 96, 28, 28)
    input_tensor_forLite = torch.randn(batch_size, 16, 3, 224, 224)

    # # Testing the Basic Conv module
    # basic_conv = BasicConv()
    # basic_out = basic_conv(input_tensor_for2d)
    # print('Basic Conv output:', basic_out.shape)

    # #Testing the Inception A module
    # inception_a = Inception_A()
    # inception_a_out = inception_a(basic_out)
    # print('Inception A output:', inception_a_out.shape)

    # # Testing the Inception B module
    # inception_b = Inception_B()
    # inception_b_out = inception_b(inception_a_out)
    # print('Inception B output:', inception_b_out.shape)

    # # Testing the Inception C module
    # inception_c = Inception_C()
    # inception_c_out = inception_c(inception_b_out)
    # print('Inception C output:', inception_c_out.shape)

    # # ECO 2D network testing
    # eco_2d = ECO_2D()
    # eco_2d_out = eco_2d(input_tensor_for2d)
    # print('ECO 2D output:', eco_2d_out.shape)  # [batch_size, 96, 28, 28]

    # #Testing the ResNet_3D_1 module
    # resnet3d_1 = Resnet3D_1()
    # resnet3d_1_out = resnet3d_1(input_tensor_for3d)
    # print('ResNet3D_1 output:', resnet3d_1_out.shape)  # [N, 128, 16, 28, 28]

    # # Testing the ResNet_3D_2 module
    # resnet3d_2 = Resnet3D_2()
    # resnet3d_2_out = resnet3d_2(resnet3d_1_out)
    # print('ResNet3D_2 output:', resnet3d_2_out.shape)  # [N, 256, 8, 14, 14]

    # #Testing the ResNet_3D_3 module
    # resnet3d_3 = Resnet3D_3()
    # resnet3d_3_out = resnet3d_3(resnet3d_2_out)
    # print('ResNet3D_3 output:', resnet3d_3_out.shape)  # [N, 512, 4, 7, 7]

    # # ECO 3D network testing
    # eco_3d = ECO_3D()
    # eco_3d_out = eco_3d(input_tensor_for3d)
    # print('ECO 3D output:', eco_3d_out.shape)  # [batch_size, 512]

    # ECO Lite network testing
    eco_lite = ECO_Lite()
    eco_lite_out = eco_lite(input_tensor_forLite)
    print('ECO 3D output:', eco_lite_out.shape)  # [batch_size, 400]
