import torch
import torch.nn as nn


class BasicBlock(nn.Module):
  """
    Basic residual block used in ResNet-18.

    Structure:
        Conv 3x3 → BatchNorm → ReLU → Conv 3x3 → BatchNorm
                            ↓
                     + shortcut
                            ↓
                          ReLU

    The shortcut is an identity mapping when input/output dimensions
    are the same. Otherwise, a 1x1 convolution is used to match
    the number of channels and/or spatial dimensions.
  """

  def __init__(self, in_channels, out_channels, stride=1):
      super(BasicBlock, self).__init__() 
      self.conv1 = nn.Conv2d(
                                in_channels, 
                                out_channels, 
                                kernel_size=(3,3), 
                                stride=stride, 
                                padding=1, 
                                bias=False
                                )
      self.bn1 = nn.BatchNorm2d(out_channels)
      self.relu = nn.ReLU(inplace=True)
      self.conv2 = nn.Conv2d(
                                out_channels, 
                                out_channels, 
                                kernel_size=(3,3), 
                                stride=(1,1), 
                                padding=1, 
                                bias=False
                                )
      self.bn2 = nn.BatchNorm2d(out_channels)
      
      self.shortcut = nn.Sequential()
      if stride != 1 or in_channels != out_channels:
          self.shortcut = nn.Sequential(
              nn.Conv2d(
                        in_channels, 
                        out_channels, 
                        kernel_size=(1,1), 
                        stride=(stride,stride), 
                        bias=False
                        ),
              nn.BatchNorm2d(out_channels)
          )

  def forward(self, x):
      out = self.conv1(x)
      out = self.bn1(out)
      out = self.relu(out)

      out = self.conv2(out)
      out = self.bn2(out)

      out += self.shortcut(x)
      out = self.relu(out)

      return out


class ResNet18(nn.Module):
    """
    ResNet-18 for grayscale image classification.

    Input:
        (batch_size, 1, height, width)

    Architecture:
        7x7 Conv → MaxPool
        → 2 BasicBlocks (64 channels)
        → 2 BasicBlocks (128 channels)
        → 2 BasicBlocks (256 channels)
        → 2 BasicBlocks (512 channels)
        → Global Average Pooling
        → Fully Connected layer

    Output:
        (batch_size, num_classes)

    The model returns raw logits. CrossEntropyLoss should be applied
    externally during training.
    """

    def __init__(self, num_classes=6):
        super(ResNet18, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(
                               in_channels=1, 
                               out_channels=64, 
                               kernel_size=(7,7), 
                               stride=(2,2), 
                               padding=(3,3), 
                               bias=False
                               )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(
                                    kernel_size=3, 
                                    stride=2, 
                                    padding=1
                                    )
        
        self.layer1 = self._make_layer(BasicBlock, 64,  2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(
                            in_features=512, 
                            out_features=num_classes
                            )


    def _make_layer(self, block, out_channels, num_blocks, stride):
        """
        Create a group of residual blocks.

        The first block can change the number of channels and/or
        reduce spatial resolution. The remaining blocks keep
        the same dimensions.
        """
        
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)


    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)
        
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
                
