
import torch
from torch import nn, optim
import torch.nn.functional as F
import Consts

############
#Embedder
############
#MODELS adopted from github
class LinearNorm(nn.Module):
    def __init__(self):
        super(LinearNorm, self).__init__()
        self.linear_layer = nn.Linear(768, 256)

    def forward(self, x):
        return self.linear_layer(x)

class Embedder(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM( # embedder inputsize = 40 = number of mel filterbanks
            input_size=40,
            hidden_size=768,
            num_layers=3, 
            batch_first=True
        )
        
        self.proj = LinearNorm()

    def forward(self, mel): # mel is the mel-filterbank energy
        #input is (num_mels x Time=T) (num_mels = 40)
        
        #create sliding window with size = 80(?) and 50% overlap
        mels = mel.unfold(1, 80, 40) #(num_mels x (T/40=T') x 80)
        mels = mels.permute(1, 2, 0) #(T' x 80 x num_mels)
        x = self.lstm(mels)[0] #(T' x 80 x lstm_hidden)
        #get last window from x
        x = x[:, -1, :] #(T' x 1 x lstm_hidden)
        x = self.proj(x) #(T' x emb_dim=256)
        #L2 norm all vectors
        x = x / torch.norm(x, p=2, dim=1, keepdim=True) #(T' x emb_dim)
        #average pooling
        x = x.sum(dim=0) / x.size(0) #(emb_dim)
        return x


############
#Extractor
############
class Extractor(nn.Module):
  def __init__(self):
    super().__init__()

    # 8 convs, 1 lstm , 2 fc
    self.conv = nn.Sequential(
      # cnn1
      nn.ZeroPad2d((3, 3, 0, 0)),
      nn.Conv2d(1, 64, kernel_size=(1, 7), dilation=(1, 1)),
      nn.BatchNorm2d(64), nn.ReLU(),

      # cnn2
      nn.ZeroPad2d((0, 0, 3, 3)),
      nn.Conv2d(64, 64, kernel_size=(7, 1), dilation=(1, 1)),
      nn.BatchNorm2d(64), nn.ReLU(),

      # cnn3
      nn.ZeroPad2d(2),
      nn.Conv2d(64, 64, kernel_size=(5, 5), dilation=(1, 1)),
      nn.BatchNorm2d(64), nn.ReLU(),

      # cnn4
      nn.ZeroPad2d((2, 2, 4, 4)),
      nn.Conv2d(64, 64, kernel_size=(5, 5), dilation=(2, 1)), # (9, 5)
      nn.BatchNorm2d(64), nn.ReLU(),

      # cnn5
      nn.ZeroPad2d((2, 2, 8, 8)),
      nn.Conv2d(64, 64, kernel_size=(5, 5), dilation=(4, 1)), # (17, 5)
      nn.BatchNorm2d(64), nn.ReLU(),

      # cnn6
      nn.ZeroPad2d((2, 2, 16, 16)),
      nn.Conv2d(64, 64, kernel_size=(5, 5), dilation=(8, 1)), # (33, 5)
      nn.BatchNorm2d(64), nn.ReLU(),

      # cnn7
      nn.ZeroPad2d((2, 2, 32, 32)),
      nn.Conv2d(64, 64, kernel_size=(5, 5), dilation=(16, 1)), # (65, 5)
      nn.BatchNorm2d(64), nn.ReLU(),

      # cnn8
      nn.Conv2d(64, 8, kernel_size=(1, 1), dilation=(1, 1)), 
      nn.BatchNorm2d(8), nn.ReLU(),
      
    )

    self.lstm = nn.LSTM( 5064, 400,
                        batch_first=True, bidirectional=True)
    
    self.fc1 = nn.Linear(2*400, 600)
    self.fc2 = nn.Linear(600, 601)

  def forward(self, x, dvec):
    # x: [B, T, num_freq]
    x = x.unsqueeze(1)
    # x: [B, 1, T, num_freq]
    x = self.conv(x)
    # x: [B, 8, T, num_freq]
    x = x.transpose(1, 2).contiguous()
    # x: [B, T, 8, num_freq]
    x = x.view(x.size(0), x.size(1), -1)
    # x: [B, T, 8*num_freq]

    # dvec: [B, emb_dim]
    dvec = dvec.unsqueeze(1)
    dvec = dvec.repeat(1, x.size(1), 1)
    # dvec: [B, T, emb_dim]

    x = torch.cat((x, dvec), dim=2) # [B, T, 8*num_freq + emb_dim]

    x, _ = self.lstm(x) # [B, T, 2*lstm_dim]
    x = F.relu(x)
    x = self.fc1(x) # x: [B, T, fc1_dim]
    x = F.relu(x)
    x = self.fc2(x) # x: [B, T, fc2_dim], fc2_dim == num_freq
    x = torch.sigmoid(x)
    return x