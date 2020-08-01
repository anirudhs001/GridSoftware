# THESE models are experimental and may not work better(if at all) than the
# architectures in models.py

# Aim: try to add skip connections in extractor

import torch
from torch import nn
import torch.nn.functional as F

############
# Extractor
############
class Extractor(nn.Module):
    def __init__(self):
        super().__init__()

        # 10 convs, 1 lstm , 2 fc

        # bs x 301 x 601
        self.conv1_resblock = nn.Sequential(
            # cnn1
            nn.ZeroPad2d((3, 3, 0, 0)),
            nn.Conv2d(1, 8, kernel_size=(1, 7), dilation=(1, 1)),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            # bs x 301 x 600 x 8
            # cnn2
            nn.ZeroPad2d((0, 0, 3, 3)),
            nn.Conv2d(8, 8, kernel_size=(7, 1), dilation=(1, 1)),
            # nn.BatchNorm2d(8), nn.ReLU(),
        )  # bs x 300 x 600 x 8

        self.conv2_resblock = nn.Sequential(
            # cnn3
            nn.ZeroPad2d(2),
            nn.Conv2d(8, 64, kernel_size=(5, 5), dilation=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # bs x 300 x 600 x 64
            # cnn4
            nn.ZeroPad2d((2, 2, 4, 4)),
            nn.Conv2d(64, 64, kernel_size=(5, 5), dilation=(2, 1)),
            # nn.BatchNorm2d(64), nn.ReLU(),
        )  # bs x 300 x 600 x 64

        self.conv3_resblock = nn.Sequential(
            # cnn5
            nn.ZeroPad2d((2, 2, 8, 8)),
            nn.Conv2d(64, 64, kernel_size=(5, 5), dilation=(4, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # bs x 300 x 600 x 64
            # cnn6
            nn.ZeroPad2d(2),
            nn.Conv2d(64, 64, kernel_size=(5, 5), dilation=(8, 1)),
            # nn.BatchNorm2d(64), nn.ReLU()
        )  # bs x 300 x 600 x 64

        self.conv4_resblock = nn.Sequential(
            # cnn7
            nn.ZeroPad2d((2, 2, 8, 8)),
            nn.Conv2d(64, 64, kernel_size=(5, 5), dilation=(4, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # bs x 300 x 600 x 64
            # cnn8
            nn.ZeroPad2d(2),
            nn.Conv2d(64, 64, kernel_size=(5, 5), dilation=(8, 1)),
            # nn.BatchNorm2d(64), nn.ReLU()
        )  # bs x 300 x 600 x 64

        self.conv5_resblock = nn.Sequential(
            # cnn9
            nn.ZeroPad2d(2),
            nn.Conv2d(64, 64, kernel_size=(5, 5), dilation=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # bs x 300 x 600 x 64
            # cnn10
            nn.ZeroPad2d(2),
            nn.Conv2d(64, 64, kernel_size=(5, 5), dilation=(32, 1)),
            # nn.BatchNorm2d(64), nn.ReLU()
        )  # bs x 300 x 600 x 64

        # EXTRAs: batch norm and relu sequential for resblocks
        self.batch_relu_64 = nn.Sequential(nn.BatchNorm2d(64), nn.ReLU(),)
        self.batch_relu_8 = nn.Sequential(nn.BatchNorm2d(8), nn.ReLU(),)
        #droupout layer for dvec(NOT using F.dropout since it is not disabled in eval mode)
        self.dropout = nn.Dropout(p=0.1)

        # LSTMs and FC layers ( same os old model )
        # inp = bs x T x 8*num_freq + emb_dim
        self.lstm = nn.LSTM(5064, 400, batch_first=True, bidirectional=True)

        self.fc1 = nn.Linear(2 * 400, 600)
        self.fc2 = nn.Linear(600, 601)

    def forward(self, x, dvec):
        # x: [B, T, num_freq]
        x = x.unsqueeze(1)
        # x: [B, 1, T, num_freq]
        x = self.conv1_resblock(x)
        x = self.batch_relu_8(x)
        # x: [B, 8, T-1, num_freq-1]

        x = self.conv2_resblock(x)
        x = self.batch_relu_64(x)
        # x: [B, 64, T-1, num_freq-1]

        y = self.conv3_resblock(x)
        # y: [B, 64, T-1, num_freq-1]
        # RESIDUAL connection
        x = y + x
        del y  # free memory

        # batchnorm and relu
        x = self.batch_relu_64(x)
        # x: [B, 64, T-1, num_freq-1]

        y = self.conv4_resblock(x)
        # y: [B, 64, T-1, num_freq-1]
        # RESIDUAL connection
        x = y + x
        del y  # free memory
        # batchnorm and relu
        x = self.batch_relu_64(x)
        # x: [B, 64, T-1, num_freq-1]

        x = self.conv5_resblock(x)
        x = self.batch_relu_8(x)
        # x: [B, 8, T, num_freq]

        x = x.transpose(1, 2).contiguous()
        # x: [B, T, 8, num_freq]
        x = x.view(x.size(0), x.size(1), -1)
        # x: [B, T, 8*num_freq]

        # dvec: [B, emb_dim]
        dvec = dvec.unsqueeze(1)
        dvec = dvec.repeat(1, x.size(1), 1)
        # dvec: [B, T, emb_dim]
        dvec = self.dropout(dvec) #adding dropout so model does not become too dependent on the dvec
        # dvec: [B, T, emb_dim]

        x = torch.cat((x, dvec), dim=2)  # [B, T, 8*num_freq + emb_dim]

        x, _ = self.lstm(x)  # [B, T, 2*lstm_dim]
        x = F.relu(x)
        x = self.fc1(x)  # x: [B, T, fc1_dim]
        x = F.relu(x)
        x = self.fc2(x)  # x: [B, T, fc2_dim], fc2_dim == num_freq
        x = torch.sigmoid(x)
        return x
