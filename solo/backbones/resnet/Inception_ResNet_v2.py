import torch
import torch.nn as nn
import torch.nn.functional as F


def Conv1(in_channel,out_channel,kernel_size,stride,padding):
    return nn.Sequential(nn.Conv2d(in_channel,out_channel,kernel_size,stride,padding),
                         nn.BatchNorm2d(out_channel),
                         nn.ReLU())

class Stem(nn.Module):
    def __init__(self, inch=32):
        super(Stem,self).__init__()
        self.conv1=Conv1(in_channel=3,out_channel=inch,kernel_size=3,stride=2,padding=0)
        self.conv2=Conv1(in_channel=inch,out_channel=inch,kernel_size=3,stride=1,padding=0)
        self.conv3=Conv1(in_channel=inch,out_channel=inch*2,kernel_size=3,stride=1,padding=1)
        self.branch1_1=nn.MaxPool2d(kernel_size=3,stride=2)
        self.branch1_2=Conv1(in_channel=inch*2,out_channel=inch*3,kernel_size=3,stride=2,padding=0)
        self.branch2_1_1=Conv1(in_channel=inch*5,out_channel=inch*2,kernel_size=1,stride=1,padding=0)
        self.branch2_1_2=Conv1(in_channel=inch*2,out_channel=inch*3,kernel_size=3,stride=1,padding=0)
        self.branch2_2_1=Conv1(in_channel=inch*5,out_channel=inch*2,kernel_size=1,stride=1,padding=0)
        self.branch2_2_2=Conv1(in_channel=inch*2,out_channel=inch*2,kernel_size=(7,1),stride=1,padding=(3,0))
        self.branch2_2_3=Conv1(in_channel=inch*2,out_channel=inch*2,kernel_size=(1,7),stride=1,padding=(0,3))
        self.branch2_2_4=Conv1(in_channel=inch*2,out_channel=inch*3,kernel_size=3,stride=1,padding=0)
        self.branch3_1=Conv1(in_channel=inch*6,out_channel=inch*6,kernel_size=3,stride=2,padding=0)
        self.branch3_2=nn.MaxPool2d(kernel_size=3,stride=2)

    def forward(self,x):
        out1=self.conv1(x)
        out2=self.conv2(out1)
        out3=self.conv3(out2)
        out4_1=self.branch1_1(out3)
        out4_2=self.branch1_2(out3)
        out4=torch.cat((out4_1,out4_2),dim=1)
        out5_1=self.branch2_1_2(self.branch2_1_1(out4))
        out5_2=self.branch2_2_4(self.branch2_2_3(self.branch2_2_2(self.branch2_2_1(out4))))
        out5=torch.cat((out5_1,out5_2),dim=1)
        out6_1=self.branch3_1(out5)
        out6_2=self.branch3_2(out5)
        out=torch.cat((out6_1,out6_2),dim=1)
        return out

class InceptionResNetA(nn.Module):
    def __init__(self,in_channel,inch=32,scale=0.1):
        super(InceptionResNetA, self).__init__()
        self.branch1=Conv1(in_channel=in_channel,out_channel=inch,kernel_size=1,stride=1,padding=0)
        self.branch2_1=Conv1(in_channel=in_channel,out_channel=inch,kernel_size=1,stride=1,padding=0)
        self.branch2_2=Conv1(in_channel=inch,out_channel=inch,kernel_size=3,stride=1,padding=1)
        self.branch3_1=Conv1(in_channel=in_channel,out_channel=inch,kernel_size=1,stride=1,padding=0)
        self.branch3_2=Conv1(in_channel=inch,out_channel=int(inch*1.5),kernel_size=3,stride=1,padding=1)
        self.branch3_3=Conv1(in_channel=int(inch*1.5),out_channel=inch*2,kernel_size=3,stride=1,padding=1)
        self.linear=Conv1(in_channel=inch*4,out_channel=inch*12,kernel_size=1,stride=1,padding=0)
        self.out_channel=inch*12
        self.scale=scale

        self.shortcut=nn.Sequential()
        if in_channel != self.out_channel:
            self.shortcut=nn.Sequential(
                nn.Conv2d(in_channels=in_channel,out_channels=self.out_channel,kernel_size=1,stride=1,padding=0),
            )

    def forward(self,x):
        output1=self.branch1(x)
        output2=self.branch2_2(self.branch2_1(x))
        output3=self.branch3_3(self.branch3_2(self.branch3_1(x)))
        out=torch.cat((output1,output2,output3),dim=1)
        out=self.linear(out)
        x=self.shortcut(x)
        out=x+self.scale*out
        out=F.relu(out)
        return out

class InceptionResNetB(nn.Module):
    def __init__(self,in_channel,inch=32,scale=0.1):
        super(InceptionResNetB, self).__init__()
        self.branch1=Conv1(in_channel=in_channel,out_channel=inch*6,kernel_size=1,stride=1,padding=0)
        self.branch2_1=Conv1(in_channel=in_channel,out_channel=inch*4,kernel_size=1,stride=1,padding=0)
        self.branch2_2=Conv1(in_channel=inch*4,out_channel=inch*5,kernel_size=(1,7),stride=1,padding=(0,3))
        self.branch2_3=Conv1(in_channel=inch*5,out_channel=inch*6,kernel_size=(7,1),stride=1,padding=(3,0))
        self.linear=Conv1(in_channel=inch*12,out_channel=inch*36,kernel_size=1,stride=1,padding=0)
        self.out_channel=inch*36
        self.scale=scale

        self.shortcut=nn.Sequential()
        if in_channel != self.out_channel:
            self.shortcut=nn.Sequential(
                nn.Conv2d(in_channels=in_channel,out_channels=self.out_channel,kernel_size=1,stride=1,padding=0)
            )

    def forward(self,x):
        output1=self.branch1(x)
        output2=self.branch2_3(self.branch2_2(self.branch2_1(x)))
        out=torch.cat((output1,output2),dim=1)
        out=self.linear(out)
        x=self.shortcut(x)
        out=x+out*self.scale
        out=F.relu(out)
        return out

class InceptionResNetC(nn.Module):
    def __init__(self,in_channel,inch=32,scale=0.1):
        super(InceptionResNetC, self).__init__()
        self.branch1 = Conv1(in_channel=in_channel, out_channel=inch*6, kernel_size=1, stride=1, padding=0)
        self.branch2_1 = Conv1(in_channel=in_channel, out_channel=inch*6, kernel_size=1, stride=1, padding=0)
        self.branch2_2 = Conv1(in_channel=inch*6, out_channel=inch*7, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.branch2_3 = Conv1(in_channel=inch*7, out_channel=inch*8, kernel_size=(3, 1), stride=1, padding=(1, 0))
        self.linear = Conv1(in_channel=inch*14, out_channel=inch*67, kernel_size=1, stride=1, padding=0)
        self.out_channel = inch*67
        self.scale = scale

        self.shortcut = nn.Sequential()
        if in_channel != self.out_channel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels=in_channel, out_channels=self.out_channel, kernel_size=1, stride=1, padding=0)
            )

    def forward(self, x):
        output1 = self.branch1(x)
        output2 = self.branch2_3(self.branch2_2(self.branch2_1(x)))
        out = torch.cat((output1, output2), dim=1)
        out = self.linear(out)
        x = self.shortcut(x)
        out=x + out * self.scale
        out=F.relu(out)
        return out

class ReductionA(nn.Module):
    def __init__(self,in_channel, inch=32):
        super(ReductionA, self).__init__()
        self.branch1=nn.MaxPool2d(kernel_size=3,stride=2,padding=0)
        self.branch2=Conv1(in_channel=in_channel,out_channel=inch*12,kernel_size=3,stride=2,padding=0)
        self.branch3_1=Conv1(in_channel=in_channel,out_channel=inch*8,kernel_size=1,stride=1,padding=0)
        self.branch3_2=Conv1(in_channel=inch*8,out_channel=inch*8,kernel_size=3,stride=1,padding=1)
        self.branch3_3=Conv1(in_channel=inch*8,out_channel=inch*12,kernel_size=3,stride=2,padding=0)

    def forward(self,x):
        out1=self.branch1(x)
        out2=self.branch2(x)
        out3=self.branch3_3(self.branch3_2(self.branch3_1(x)))
        return torch.cat((out1,out2,out3),dim=1)

class ReductionB(nn.Module):
    def __init__(self,in_channel, inch=32):
        super(ReductionB, self).__init__()
        self.branch1=nn.MaxPool2d(kernel_size=3,stride=2,padding=0)
        self.branch2_1=Conv1(in_channel=in_channel,out_channel=inch*8,kernel_size=1,stride=1,padding=0)
        self.branch2_2=Conv1(in_channel=inch*8,out_channel=inch*12,kernel_size=3,stride=2,padding=0)
        self.branch3_1=Conv1(in_channel=in_channel,out_channel=inch*8,kernel_size=1,stride=1,padding=0)
        self.branch3_2=Conv1(in_channel=inch*8,out_channel=inch*9,kernel_size=3,stride=2,padding=0)
        self.branch4_1=Conv1(in_channel=in_channel,out_channel=inch*8,kernel_size=1,stride=1,padding=0)
        self.branch4_2=Conv1(in_channel=inch*8,out_channel=inch*9,kernel_size=3,stride=1,padding=1)
        self.branch4_3=Conv1(in_channel=inch*9,out_channel=inch*10,kernel_size=3,stride=2,padding=0)

    def forward(self,x):
        out1=self.branch1(x)
        out2=self.branch2_2(self.branch2_1(x))
        out3=self.branch3_2(self.branch3_1(x))
        out4=self.branch4_3(self.branch4_2(self.branch4_1(x)))
        return torch.cat((out1,out2,out3,out4),dim=1)

class InceptionResNetV2(nn.Module):
    def __init__(self,num_classes=1000):
        super(InceptionResNetV2, self).__init__()
        blocks,inch=[],16
        blocks.append(Stem(inch))
        for _ in range(5):
            blocks.append(InceptionResNetA(inch*12,inch))
        blocks.append(ReductionA(inch*12,inch))
        for _ in range(10):
            blocks.append(InceptionResNetB(inch*36,inch))
        blocks.append(ReductionB(inch*36,inch))
        for _ in range(5):
            blocks.append(InceptionResNetC(inch*67,inch))
        self.map=nn.Sequential(*blocks)
        self.pool=nn.AdaptiveAvgPool2d((1, 1))
        self.inplanes = inch*67
        self.fc = nn.Linear(inch*67, num_classes)

    def forward(self,x):
        out=self.map(x)
        out=self.pool(out)
        out=out.view(out.size(0), -1)
        out=self.fc(out)
        return out

# if __name__=="__main__":
# #     net=InceptionResNetV2()
# # #     data = torch.randn(1,3,448,448)
# # #     otp = net(data)
# #     print(net)
# #     # # stat(net, (3, 224, 224))
# #     #
#     import torchvision.models as models
#     import thop
#
#     model = models.resnet18()
#     x = torch.randn(1, 3, 224, 224)
#     flops, params = thop.profile(model, inputs=(x,))
#     print('flops',flops/1000000)
#     print('params', params/1000000)
#     net = InceptionResNetV2()
#     flops, params = thop.profile(net, inputs=(x,))
#     print('flops',flops/1000000)
#     print('params', params/1000000)