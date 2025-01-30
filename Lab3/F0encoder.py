import torch.nn as nn

class F0_encoder(nn.Module):
    def __init__(self,d_model):
        super(F0_encoder, self).__init__()
        self.conv1 = nn.Conv1d(1, d_model, 3)
        self.conv2 = nn.Conv1d(d_model, d_model, 3)
        self.conv3 = nn.Conv1d(d_model, d_model, 3)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = self.conv3(x)
        x = nn.functional.relu(x)
        return x