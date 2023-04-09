import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt
import numpy as np
import time

########################################
# You can define whatever classes if needed
########################################

class IdentityResNet(nn.Module):
    
    # __init__ takes 4 parameters
    # nblk_stage1: number of blocks in stage 1, nblk_stage2.. similar
    def __init__(self, nblk_stage1, nblk_stage2, nblk_stage3, nblk_stage4):
        super(IdentityResNet, self).__init__()
    ########################################
    # Implement the network
    # You can declare whatever variables
    ########################################
        self.nblk_stage1 = nblk_stage1
        self.nblk_stage2 = nblk_stage2
        self.nblk_stage3 = nblk_stage3
        self.nblk_stage4 = nblk_stage4

        self.conv0 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv2_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.conv2_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv3_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1)
        self.conv3_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv4_1 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1)
        self.conv4_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        
        self.bn64 = nn.BatchNorm2d(num_features=64)
        self.bn128 = nn.BatchNorm2d(num_features=128)
        self.bn256 = nn.BatchNorm2d(num_features=256)
        self.bn512 = nn.BatchNorm2d(num_features=512)

        self.shortcut2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1, stride=2)
        self.shortcut3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=1, stride=2)
        self.shortcut4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=1, stride=2)

        self.stage1_block = nn.Sequential(self.bn64, nn.ReLU(), self.conv1, self.bn64, nn.ReLU(), self.conv1,
        self.bn64, nn.ReLU(), self.conv1, self.bn64, nn.ReLU(), self.conv1)
        
        self.stage2_first_before = nn.Sequential(self.bn64, nn.ReLU())
        self.stage2_first_after = nn.Sequential(self.conv2_1, self.bn128, nn.ReLU(), self.conv2_2)
        self.stage2_else = nn.Sequential(self.bn128, nn.ReLU(), self.conv2_2, self.bn128, nn.ReLU(), self.conv2_2)

        self.stage3_first_before = nn.Sequential(self.bn128, nn.ReLU())
        self.stage3_first_after = nn.Sequential(self.conv3_1, self.bn256, nn.ReLU(), self.conv3_2)
        self.stage3_else = nn.Sequential(self.bn256, nn.ReLU(), self.conv3_2, self.bn256, nn.ReLU(), self.conv3_2)

        self.stage4_first_before = nn.Sequential(self.bn256, nn.ReLU())
        self.stage4_first_after = nn.Sequential(self.conv4_1, self.bn512, nn.ReLU(), self.conv4_2)
        self.stage4_else = nn.Sequential(self.bn512, nn.ReLU(), self.conv4_2, self.bn512, nn.ReLU(), self.conv4_2)

        self.linear = nn.Linear(512, 10)

    ########################################
    # You can define whatever methods
    ########################################

    def stage1(self, x):
        s1_bn = self.bn64(x)
        s1_relu = F.relu(s1_bn)
        s1_conv = self.conv1(s1_relu)
        s1_bn2 = self.bn64(s1_conv)
        s1_relu2 = F.relu(s1_bn2)
        s1_conv2 = self.conv1(s1_relu2)

        return s1_conv2

    def forward(self, x):
        ########################################
        # Implement the network
        # You can declare or define whatever variables or methods
        ########################################

        conv33 = self.conv0(x)


        s1_out = self.stage1_block(conv33) + conv33
        for i in range (self.nblk_stage1 - 1):
          s1_out = self.stage1_block(s1_out) + s1_out
        '''s1_out = self.stage1(conv33) + conv33
        for i in range (self.nblk_stage1 - 1):
          s1_out = self.stage1(s1_out) + s1_out'''
        
        s2_x = self.stage2_first_before(s1_out)
        s2_out = self.stage2_first_after(s2_x) + self.shortcut2(s2_x)
        for i in range (self.nblk_stage2 - 1):
          s2_out = self.stage2_else(s2_out) + s2_out
        
        s3_x = self.stage3_first_before(s2_out)
        s3_out = self.stage3_first_after(s3_x) + self.shortcut3(s3_x)
        for i in range (self.nblk_stage3 - 1):
          s3_out = self.stage3_else(s3_out) + s3_out
        
        s4_x = self.stage4_first_before(s3_out)
        s4_out = self.stage4_first_after(s4_x) + self.shortcut4(s4_x)
        for i in range(self.nblk_stage4 - 1):
          s4_out = self.stage4_else(s4_out) + s4_out

        avg_pool = F.avg_pool2d(s4_out, kernel_size=4, stride=4)
        avg_pool = avg_pool.view(-1, 512)
        out = self.linear(avg_pool)

        return out

########################################
# Q1. set device
# First, check availability of GPU.
# If available, set dev to "cuda:0";
# otherwise set dev to "cpu"
########################################
dev = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
print('current device: ', dev)

########################################
# data preparation: CIFAR10
########################################

########################################
# Q2. set batch size
# set batch size for training data
########################################
batch_size = 4

# preprocessing
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# load training data
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True)

# load test data
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# define network
net = IdentityResNet(nblk_stage1=2, nblk_stage2=2,
                     nblk_stage3=2, nblk_stage4=2)

########################################
# Q3. load model to GPU
# Complete below to load model to GPU
########################################
net = net.to(dev)


# set loss function
criterion = nn.CrossEntropyLoss()

########################################
# Q4. optimizer
# Complete below to use SGD with momentum (alpha= 0.9)
# set proper learning rate
########################################
optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

# start training
t_start = time.time()

for epoch in range(5):  # loop over the dataset multiple times
    
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(dev), data[1].to(dev)
        
        ########################################
        # Q5. make sure gradients are zero!
        # zero the parameter gradients
        ########################################
        net.zero_grad()
        
        ########################################
        # Q6. perform forward pass
        ########################################
        outputs = net(inputs)
        
        # set loss
        loss = criterion(outputs, labels)
        
        ########################################
        # Q7. perform backprop
        ########################################
        loss.backward()
        
        ########################################
        # Q8. take a SGD step
        ########################################
        optimizer.step()
        
        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0
            t_end = time.time()
            print('elapsed:', t_end-t_start, ' sec')
            t_start = t_end

print('Finished Training')


# now testing
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))

########################################
# Q9. complete below
# when testing, computation is done without building graphs
########################################
with torch.no_grad():
    for data in testloader:
        images, labels = data[0].to(dev), data[1].to(dev)
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

# per-class accuracy
for i in range(10):
    print('Accuracy of %5s' %(classes[i]), ': ',
          100 * class_correct[i] / class_total[i],'%')

# overall accuracy
print('Overall Accurracy: ', (sum(class_correct)/sum(class_total))*100, '%')

