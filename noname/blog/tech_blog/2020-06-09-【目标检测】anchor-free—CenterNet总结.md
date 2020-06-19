# centernet
*【文前】本文所讲述的 CenterNet 为 [**Objects as Points**] (http://arxiv.org/abs/1904.07850))
有两篇 paper 都为 centernet（论文撞车，另一篇文章为CenterNet: Keypoint Triplets for Object Detection，基本思路异曲同工之妙，都是 anchor-free）*
![image.png](https://upload-images.jianshu.io/upload_images/9730793-cf08a34e723d52ba.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

【论文】[https://arxiv.org/pdf/1904.07850.pdf](https://arxiv.org/pdf/1904.07850.pdf)

【代码】[https://github.com/xingyizhou/CenterNet](https://github.com/xingyizhou/CenterNet)


## 【result】
1、CenterNet 属于 anchor-free 系列的目标检测，相比于 CornerNet 做出了改进，使得检测速度和精度相比于 one-stage 和 two-stage 的框架都有不小的提高，尤其是与 YOLOv3 作比较，在相同速度的条件下，CenterNet 的精度比 YOLOv3 提高了 4 个左右的点。
![各个模型效果对比](https://upload-images.jianshu.io/upload_images/9730793-28437e5a19979fc9.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/340)

2、不仅可以做 2D 目标检测, 只需要少量扩展就可以迁移到 3D 目标检测和人体关键点检测上。



### 【anchor-base 方法痛点】
1、fastRCNN 依赖计算量高的 region proposal 方法，fasterRCNN RPN 网络需要单独训练 RPN生成anchor
2、featuremap 分辨率较低，后NMS 处理（正：IOU>0.7,负：IOU<0.3）计算量不小

### 【centernet 处理办法】
1、objects as points。所有的检测物体用中心点+长宽表示。这一点从数学来思考：物体都是呈现几何形状，几何形状的中心点代表物体的关键点信息，大部分物体几何中心在物体本身上（特殊情况以及杠精除外）。
而且目前目标检测标注数据都是矩形框而不是像素级别，这样其实对于网络来说，模型傻傻的，当然认定人喂给网络的就是矩形框里的信息都是正信息，不会聪明到人眼睛看到那个真实物体的轮廓，在机器模型眼中，标注的矩形框就是模型认定的轮廓，模型再从所有同一类别下的矩形框自己学习共性，输出包含某一个类别共性的矩形框。所以这样的假定在数学上是可以理解的。（后续可以发展多个几何关键点作为代表，论文 idea！！！）
![image.png](https://upload-images.jianshu.io/upload_images/9730793-d4428398d4179ba4.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
关键点如何得到？采用 heatmap 极值点来表示。
网络训练完成后在 inference 阶段，输入图（WxHx3）经过网络在 feature map 上（![image.png](https://upload-images.jianshu.io/upload_images/9730793-726b8516ac15c830.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
）其中R为相对于原图来说的 stride，C为最终分类数目，论文中 R=4，C=80（coco 数据集），在每个 channel 类别通道上取像素峰值点（论文中100），峰值点选取规则：比周围八个点都大的点，作为检测目标的中心点，再经过三个head：分类网络，中心点 offset 网络以及 W，H 回归网络。offset 说明：因为原图W/R 可能为小数，这样在 featuremap 位置对应会有偏差，用 offset 网络来弥补这个偏差（和 maskRCNN 的 roi align机制异曲同工之妙）


2、分辨率较低问题，因为 center 中心点在分辨率低的 feature map 上偏差很大，感受野较大，因此论文中网络采用卷积+上采样的网络结构，论文中涉及三种网络
Resnet-18 with up-convolutional layers : 28.1% coco and 142 FPS
DLA-34 : 37.4% COCOAP and 52 FPS
Hourglass-104 : 45.1% COCOAP and 1.4 FPS
![image.png](https://upload-images.jianshu.io/upload_images/9730793-4e9f238de732b3e9.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
![hourglass network architecture](https://upload-images.jianshu.io/upload_images/9730793-61ccb375f4485831.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
![yolo v3 network architecture](https://upload-images.jianshu.io/upload_images/9730793-f0ecfa6869d55c67.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

最终输出分辨率为：【W/R, H/R, C】(R=4)，这样高分辨率下对小目标的检测效果较好，
NMS 问题：文中在 heatmap（和 feature map一致， 只是看问题重点不同，heatmap 着重像素值6高低而行程的像热力图那样的）进行取峰值点可类比于 anchor-base 方法中的 NMS 操作，都是取极值操作，只不过centernet在一维点 point 信息下，传统方法在二维ROI区域来说，属于降维打击，这样子计算量就大大降低（但是负样本数量就多了）

## 【训练过程】
先看 loss 设计
![Loss function](https://upload-images.jianshu.io/upload_images/9730793-4b853c3dbf70facf.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/440)
1、分类采用 focal loss![focal loss](https://upload-images.jianshu.io/upload_images/9730793-a7f99e03af1f90c6.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/440)
在 heatmap 上每个峰值点为预测为 Yba，真是 ground-truth 为 Y，非 Y 其他点都为负样本（因一个点为正，其他为负，这样子在相比于 roi 二维区域中，负样本数量大大升高），直接计算 focal loss ，1-Y=0占大多数，模型学习会负样受本影响，怎么办呢？
文中在loss 计算时，ground-truth 映射到 W/R heatmap，以 ground-truth 中心点为中心使用高斯延展，中心点为1，其他地方不直接为0，而时比1小的数，具体延展陡峭程度由高斯核sigma 决定
![image.png](https://upload-images.jianshu.io/upload_images/9730793-312512f1aa1bebbf.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/440)

2、offset loss![offset loss](https://upload-images.jianshu.io/upload_images/9730793-9ed8d710a6f53033.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/240)

3、size regression
![image.png](https://upload-images.jianshu.io/upload_images/9730793-7c24d885818002d5.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/240)

![size regression](https://upload-images.jianshu.io/upload_images/9730793-72ae5276a815a6db.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/240)

## 【实验结果】
centernet 不同网络测试精确度以及速度比对，精度最高的时 hourglass 网络，但是其速度慢，在实际业务应用中采用的是 DLA34网络，可以达到精度与速度的均衡
![](https://upload-images.jianshu.io/upload_images/9730793-3053bca783386774.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
![](https://upload-images.jianshu.io/upload_images/9730793-08a42e29b2904b38.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
![](https://upload-images.jianshu.io/upload_images/9730793-6318a9939d9d16fe.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

其他3D 检测和 pose 检测此处略过。

【reference】
1、[真 Anchor Free 目标检测：CenterNet 详解](https://www.infoq.cn/article/XUDiNPviWhHhvr6x_oMv)
