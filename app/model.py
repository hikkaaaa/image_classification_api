import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        
        #Convolutional Block 1
        #inputL (3, 32, 32) -> output: (16, 32, 32)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        #pooling: reduces spatial dimensions by 2 -> (16, 16, 16)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        #Convolutional Block 2
        #input: (16, 16, 16) -> output: (32, 16, 16)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        #maxpool -> (32, 8, 8)

        #convolutional block 6
        #input: (32, 8, 8) -> output: (64, 8, 8)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        #maxpool -> (64, 4, 4)

        #Fully Connected Block
        #64 cgannels * 4 * 4 spatial dimenstion = 1024 inputs
        self.fc1 = nn.Linear(in_features=1024, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=num_classes)

        self.dropout = nn.Dropout(p=0.25)

    def forward(self, x):
        #Convolutional Block 1
        x = self.pool(F.relu(self.conv1(x)))
        #Convolutional Block 2
        x = self.pool(F.relu(self.conv2(x)))
        #Convolutional Block 3
        x = self.pool(F.relu(self.conv3(x)))
        #Fully Connected Block
        x = x.view(-1, 64 * 4 * 4)
        
        #classifiction head
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    #quick test to verify dimensions
if __name__ == "__main__":
    model = SimpleCNN()
    dummy_input = torch.randn(1, 3, 32, 32) #batch size = 1,rgb, 32 x 32
    output = model(dummy_input)
    print(output.shape)