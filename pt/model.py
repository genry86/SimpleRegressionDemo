import torch
import torch.nn as nn

class DetectCenterCNNModel(nn.Module):
    def __init__(self, in_channels, out):
        super(DetectCenterCNNModel, self).__init__()
        self.conv = nn.Sequential(
            # nn.Conv2d(in_channels, 32, kernel_size=3, padding=0, bias=False),  #  (b, 1, 64, 64) -> (b, 32, 62, 62)
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),  # (b, 1, 64, 64) -> (b, 32, 64, 64)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (b, 32, 64, 64) -> (b, 32, 32, 32)

            # nn.Conv2d(32, 64, kernel_size=3, padding=0, bias=False),  # (b, 32, 62, 62) -> (b, 64, 60, 60)
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),  # (b, 32, 32, 32) -> (b, 64, 32, 32)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # (b, 64, 32, 32) -> (b, 64, 16, 16)
        )

        self.flatten = nn.Flatten()     # (b, 64, 60, 60) -> (b, 64*60*60)  or (b, 64, 16, 16) -> (b, 64*16*16)

        self.fn = nn.Sequential(
            nn.Linear(64*16*16, 128),
            # nn.Linear(64*60*60, 128),
            # nn.Linear(64*64, 128),

            nn.ReLU(),
            # nn.Dropout(0.1),
            nn.Linear(128, out)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.flatten(x)
        x = self.fn(x)
        return x

if __name__ == '__main__':
    BATCH_SIZE = 128
    model = DetectCenterCNNModel(1, out=2)
    print(model)

    input = torch.rand(BATCH_SIZE, 1, 64, 64, dtype=torch.float32)
    output = model(input)
    print("output - ", output)
    print("shape - ", output.shape)