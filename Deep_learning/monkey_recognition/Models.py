import torch.nn as nn

class FC64(nn.Module):
    def __init__(self):
        super(FC64, self).__init__()

        self.fc1 = nn.Linear(64*64*3, 1024) 
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 10)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.fc5(x)
        return x


class ConvNet(nn.Module):
    def __init__(self, num_classes):
        super(ConvNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16,
                               kernel_size=3, padding=1)

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32,
                               kernel_size=3, padding=1)

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64,
                               kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64,
                               kernel_size=3, padding=1)

        self.maxpool = nn.MaxPool2d(kernel_size=2)

        self.dropout = nn.Dropout()

        self.fc1 = nn.Linear(in_features=1024, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=num_classes)

        self.relu = nn.ReLU()


    def forward(self, x):
        #64x64x3
        x = self.relu(self.conv1(x))
        #64x64x16
        x = self.maxpool(x)
        #32x32x16
        x = self.relu(self.conv2(x))
        #32x32x32
        x = self.maxpool(x)
        #16x16x32
        x = self.relu(self.conv3(x))
        #16x16x64
        x = self.maxpool(x)
        #8x8x64
        x = self.relu(self.conv4(x))
        #8x8x64
        x = self.maxpool(x)
        #4x4x64
        x = x.view(-1, 1024)

        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x 

    import torch.nn as nn


class AlexNet(nn.Module):
    def __init__(self, num_classes):
        super(AlexNet, self).__init__()

        self.convA = nn.Conv2d(in_channels=3, out_channels=96,
                               kernel_size=11, stride=4)
        self.convB = nn.Conv2d(in_channels=96, out_channels=256,
                               kernel_size=5, padding=2)
        self.convC1 = nn.Conv2d(in_channels=256, out_channels=384,
                                kernel_size=3, padding=1)
        self.convC2 = nn.Conv2d(in_channels=384, out_channels=384,
                                kernel_size=3, padding=1)
        self.convC3 = nn.Conv2d(in_channels=384, out_channels=256,
                                kernel_size=3, padding=1)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)

        self.fc1 = nn.Linear(in_features=9216, out_features=4096)
        self.fc2 = nn.Linear(in_features=4096, out_features=4096)
        self.fc3 = nn.Linear(in_features=4096, out_features=num_classes)

        self.relu = nn.ReLU()


    def forward(self, x):
        #227x227x3
        x = self.relu(self.convA(x)) # k=11, s=4, c=96
        #55x55x96
        x = self.maxpool(x) # k=3, s=2
        #27x27x96
        x = self.relu(self.convB(x)) # k=5, s=1, p="same"=2, c=256
        #27x27x256
        x = self.maxpool(x) # k=3, s=2
        #13x13x256
        x = self.relu(self.convC1(x))# k=5, s=1, p="same"=2, c=384
        #13x13x384
        x = self.relu(self.convC2(x))# k=5, s=1, p="same"=2, c=384
        #13x13x384
        x = self.relu(self.convC3(x))# k=5, s=1, p="same"=2, c=256
        #13x13x256
        x = self.maxpool(x)
        #6x6x256
        #Flatening
        x = x.view(-1, 6*6*256)

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)

        return x 



