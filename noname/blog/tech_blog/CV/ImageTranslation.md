# 图像翻译

## Preface

## Introduction

### pix2pix
  paper: [Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/pdf/1611.07004.pdf)
  
  code:  [https://github.com/phillipi/pix2pix](https://github.com/phillipi/pix2pix)
  
  本文最大的贡献在于提出了一个统一的框架解决了图像翻译问题。在这篇paper里面，作者提出的框架十分简洁优雅（好用的算法总是简洁优雅的）。相比以往算法的大量专家知识，手工复杂的loss。这篇paper非常粗暴，使用CGAN处理了一系列的转换问题。
  
### pix2pixHD
  paper: [High-Resolution Image Synthesis and Semantic Manipulation with Conditional GANs](https://arxiv.org/pdf/1711.11585v1.pdf)
  
  code:  [https://github.com/NVIDIA/pix2pixHD](https://github.com/NVIDIA/pix2pixHD)
  
  这篇paper作为pix2pix的改进版本，如其名字一样，主要是可以产生高分辨率的图像。具体来说，作者的贡献主要在以下两个方面：

  * 使用多尺度的生成器以及判别器等方式从而生成高分辨率图像。
    - 模型结构
    
      生成器由两部分组成，G1和G2，其中G2又被割裂成两个部分。G1和pix2pix的生成器没有差别，就是一个end2end的U-Net结构。G2的左半部分提取特征，并和G1的输出层的前一层特征进行相加融合信息，把融合后的信息送入G2的后半部分输出高分辨率图像。
判别器使用多尺度判别器，在三个不同的尺度上进行判别并对结果取平均。判别的三个尺度为：原图，原图的1/2降采样，原图的1/4降采样（实际做法为在不同尺度的特征图上进行判别，而非对原图进行降采样）。显然，越粗糙的尺度感受野越大，越关注全局一致性。生成器和判别器均使用多尺度结构实现高分辨率重建，思路和PGGAN类似，但实际做法差别比较大。
    - Loss设计
      + GAN loss：和pix2pix一样，使用PatchGAN。
      + Feature matching loss：将生成的样本和Ground truth分别送入判别器提取特征，然后对特征做Element-wise loss
      + Content loss：将生成的样本和Ground truth分别送入VGG16提取特征，然后对特征做Element-wise loss
      
      使用Feature matching loss和Content loss计算特征的loss，而不是计算生成样本和Ground truth的MSE，主要在于MSE会造成生成的图像过度平滑，缺乏细节。Feature matching loss和Content loss只保证内容一致，细节则由GAN去学习。
    - 使用Instance-map的图像进行训练。
  * 使用了一种非常巧妙的方式，实现了对于同一个输入，产生不同的输出。并且实现了交互式的语义编辑方式，这一点不同于pix2pix中使用dropout保证输出的多样性。
  
  作者主要的贡献在于：
  * 提出了生成高分辨率图像的多尺度网络结构，包括生成器，判别器
  * 提出了Feature loss和VGG loss提升图像的分辨率 - 通过学习隐变量达到控制图像颜色，纹理风格信息
  * 通过Boundary map提升重叠物体的清晰度

## Result

## Reference
