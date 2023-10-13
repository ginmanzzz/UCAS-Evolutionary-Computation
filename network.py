import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

class network(nn.Module):
    def __init__(self, hyParaList = None):
        super(network, self).__init__()
        self.gene = hyParaList
        self.conv1 = nn.Conv2d(3, 64, 5, padding=2)
        self.norm1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, 3, padding = 1)
        self.norm2 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(32*32*128, 500)
        self.fc2 = nn.Linear(500, 10)
    def forward(self, x):
        batch_size = x.size(0)
        out = self.conv1(x)
        out = F.relu(out)
        out = self.conv2(out)
        out = F.relu(out)
        out = out.view(batch_size, -1)
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        out = F.log_softmax(out, dim=1)
        return out

class vggNet(nn.Module):
    def __init__(self, hyParaList = None):
        super(vggNet, self).__init__()
        self.gene = hyParaList
        self.conv1_1 = nn.Conv2d(3, 64, 3, padding = 1) # 32*32*64
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding = 1)
        self.pool1 = nn.MaxPool2d(2, 2) # 16*16*64
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding = 1)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding = 1) # 16*16*128
        self.pool2 = nn.MaxPool2d(2, 2) # 8*8*128   
        self.conv3_1 = nn.Conv2d(128, 256, 3, padding = 1)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding = 1)
        self.pool3 = nn.MaxPool2d(2, 2) # 4*4*256
        self.fc1 = nn.Linear(4*4*256, 500)
        self.fc2 = nn.Linear(500, 10)
        self.sequen = nn.Sequential(
                self.conv1_1,
                nn.BatchNorm2d(64),
                nn.ReLU(),
                self.conv1_2,
                nn.BatchNorm2d(64),
                nn.ReLU(),
                self.pool1,
                self.conv2_1,
                nn.BatchNorm2d(128),
                nn.ReLU(),
                self.conv2_2,
                nn.BatchNorm2d(128),
                nn.ReLU(),
                self.pool2,
                self.conv3_1,
                nn.BatchNorm2d(256),
                nn.ReLU(),
                self.conv3_2,
                nn.BatchNorm2d(256),
                nn.ReLU(),
                self.pool3
        )

    def forward(self, x):
        batch_size = x.size(0)
        out = self.sequen(x)
        out = out.view(batch_size, -1)
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        out = F.relu(out)
        out = F.log_softmax(out, dim=1)
        return out
class netIndividual(nn.Module):
    def __init__(self, hyParaList):
        """
            hyParaList(Dict):
                number_nodes(list) = [layer1(conv), layer2(conv), lyaer3(fc)]
                learn_rate(float)  = 0.1 or other
                batch_size(int) = 512 or other
                kernel_size(list) = [layer1, layer2] 3 or 5
                fitness
        """
        super(netIndividual, self).__init__()
        self.gene = hyParaList
        self.gene['fitness'] = 0.0
        nb_1 = hyParaList['nb_node'][0]
        nb_2 = hyParaList['nb_node'][1]
        nb_3 = hyParaList['nb_node'][2]
        kernel_1 = hyParaList['kernels'][0]
        kernel_2 = hyParaList['kernels'][1]
        padding1 = int((kernel_1 - 1) / 2)
        padding2 = int((kernel_2 - 1) / 2)
        self.conv1_1 = nn.Conv2d(3, nb_1, kernel_1, padding = padding1) # 32*32*64
        self.conv1_2 = nn.Conv2d(nb_1, nb_1, kernel_1, padding = padding1)
        self.pool1 = nn.MaxPool2d(2, 2) # 16*16*64
        self.conv2_1 = nn.Conv2d(nb_1, nb_2, kernel_2, padding = padding2)
        self.conv2_2 = nn.Conv2d(nb_2, nb_2, kernel_2, padding = padding2) # 16*16*128
        self.pool2 = nn.MaxPool2d(2, 2) # 8*8*128   
        self.fc1 = nn.Linear(8*8*nb_2, nb_3)
        self.fc2 = nn.Linear(nb_3, 10)
        self.sequen = nn.Sequential(
                self.conv1_1,
                nn.BatchNorm2d(nb_1),
                nn.ReLU(),
                self.conv1_2,
                nn.BatchNorm2d(nb_1),
                nn.ReLU(),
                self.pool1,
                self.conv2_1,
                nn.BatchNorm2d(nb_2),
                nn.ReLU(),
                self.conv2_2,
                nn.BatchNorm2d(nb_2),
                nn.ReLU(),
                self.pool2)

    def forward(self, x):
        batch_size = x.size(0)
        out = self.sequen(x)
        out = out.view(batch_size, -1)
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        out = F.relu(out)
        out = F.log_softmax(out, dim=1)
        return out
