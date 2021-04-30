# problems about CV 


1. BatchNorm, LayerNorm, InstanceNorm和GroupNorm  
  - Batch Norm  
    - 输入[B,N,W,H] 在[B,W,H] 做归一化
    - **参数**：gamma[N] beta[N] mean[N] var[N]  
    - **输出**：gamma*((input-mean)/sqrt(var)) + beta  
    - **作用**：解决"Internal Covariate Shift" 问题(网络训练输入输出分布出现偏移，如sigmoid产生的值较小导致梯度消失现象，难以收敛)，但受batchsize 影响较大  
    - **注意**：
          - Batch Normalization不能在小数据集上进行，因为均值和方差的估计会不准确。
          - 在batch normalization的时候，我们在train和test的时候进行的操作是不同的。这是由于在test的时候, 输入数据可能只有一个data, 故不能计算均值和标准差;所以, 在test的时候, 会使用之前训练计算得到的均值和标准差做标准化。
          - RNN变长序列计算不友好
    - **作用**：解决"Internal Covariate Shift" 问题(网络训练输入输出分布出现偏移，如sigmoid产生的值较小导致梯度消失现象，难以收敛)，但受batchsize 影响较大  
    - **代码**：
      ```
      class BatchNorm2d(nn.Module):
          def __init__(self, dim, momentum=0.1):
              super(BatchNorm2d, self).__init__()
              self.gamma = nn.Parameter(torch.ones([dim]))
              self.beta = nn.Parameter(torch.zeros([dim]))
              self.register_buffer("moving_mean", torch.zeros([dim]))
              self.register_buffer("moving_var", torch.ones([dim]))
              self.register_buffer("momentum",torch.tensor(momentum))

          def forward(self, x):
              if self.training:
                  mean = x.mean(dim=[0,2,3])
                  var = x.var(dim=[0,2,3])
                  self.moving_mean = self.moving_mean * self.momentum + mean * (1 - self.momentum)
                  self.moving_var  = self.moving_var * self.momentum + var * (1 - self.momentum)
              x = (x - self.moving_mean) / (torch.sqrt(self.moving_var + 1e-8))
              return x * self.gamma + self.beta
      ```
  - LayerNorm
    - 输入[B,N,W,H] 在[N,W,H] 做归一化,如果不指定dim时候，默认为后两个维度，即在[W,H]做归一化
    - **参数**：gamma[B] beta[B] mean[B] var[B]  
    - **输出**：gamma*((input-mean)/sqrt(var)) + beta  
    - **作用**：解耦batchsize，不同样本拥有各自的平均值和标准差,在特征维度进行归一化，对每个Batch有一个均值和方差，因此不依赖于batch大小，即使batch为1也能使用。
    - **代码**：
      ```
      import torch
      import torch.nn as nn
      class LayerNorm2d(nn.Module):
          def __init__(self, dim, momentum=0.1):
              super(LayerNorm2d, self).__init__()
              self.gamma = nn.Parameter(torch.ones([1,dim,1,1]))
              self.beta = nn.Parameter(torch.zeros([1,dim,1,1]))

          def forward(self, x):
              mean = x.mean(dim=[1,2,3],keepdim=True)
              var = x.var(dim=[1,2,3],keepdim=True)
              x = (x - mean) / (torch.sqrt(var + 1e-8))
              return x * self.gamma + self.beta
      ```
  - InstanceNorm
    - 输入[B,N,W,H] 在[W, H] 做归一化
    - **参数**：gamma[B*N] beta[B*N] mean[B*N] var[B*N]  
    - **输出**：gamma*((input-mean)/sqrt(var)) + beta  
    - **作用**：
    - **代码**：
  - GroupNorm
    - 输入[B,N,W,H] 在[N/g, W, H] 做归一化
    - **参数**：gamma[B*g] beta[B*g] mean[B*g] var[B]  
    - **输出**：gamma*((input-mean)/sqrt(var)) + beta  
    - **作用**：对 Layer Norm 和 Instance Norm 的折中
    - **代码**：
      ```
      import torch
      import torch.nn as nn
      class GroupNorm2d(nn.Module):
          def __init__(self, dim, g, momentum=0.1):
              super(GroupNorm2d, self).__init__()
              self.gamma = nn.Parameter(torch.ones([1,dim,1,1]))
              self.beta = nn.Parameter(torch.zeros([1,dim,1,1]))
              self.register_buffer("g",torch.tensor(g))

          def forward(self, x):
              shape = x.shape
              x = x.view(shape[0], self.g, -1, shape[2], shape[3])
              mean = x.mean(dim=[2,3,4],keepdim=True)
              var = x.var(dim=[2,3,4],keepdim=True)
              x = (x - mean) / (torch.sqrt(var + 1e-8))
              x = x.view(shape)
              print(x.shape)
              return x * self.gamma + self.beta
      ```
## Reference
  1. [BatchNorm LayerNorm 实现](https://zhuanlan.zhihu.com/p/172185048)
  2. [Batch Normalization技术介绍](https://mathpretty.com/10335.html)
  3. [BatchNorm, LayerNorm, InstanceNorm和GroupNorm总结](https://mathpretty.com/11223.html)
