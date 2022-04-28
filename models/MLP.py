import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(
      self, 
      input_size: int = 784, 
      output_size: int = 10
    ):
        super(MLP, self).__init__()
        self.fc_1 = nn.Linear(input_size, 128)
        self.fc_2 = nn.Linear(128, 128)
        self.fc_3 = nn.Linear(128, output_size)

    def forward(self, x):
        out = x.view(x.size(0), -1)
        out = F.relu(self.fc_1(out))
        out = F.relu(self.fc_2(out))
        out = self.fc_3(out)
        return out