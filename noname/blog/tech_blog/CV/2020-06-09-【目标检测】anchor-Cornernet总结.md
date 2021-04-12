# cornernet mini 总结

原文链接：https://blog.csdn.net/u014380165/article/details/83032273

论文：CornerNet: Detecting Objects as Paired Keypoints
论文链接：[https://arxiv.org/abs/1808.01244](https://arxiv.org/abs/1808.01244)
代码链接：[https://github.com/umich-vl/CornerNet](https://github.com/umich-vl/CornerNet)

这篇发表在ECCV2018上的目标检测文章给人一种眼前一亮的感觉，简单说一下几个比较吸引我的点：1、将目标检测问题当作关键点检测问题来解决，也就是通过检测目标框的左上角和右下角两个关键点得到预测框，因此CornerNet算法中没有anchor的概念，这种做法在目标检测领域是比较创新的而且能够取得不错效果是很难的。2、整个检测网络的训练是从头开始的，并不基于预训练的分类模型，这使得用户能够自由设计特征提取网络，不用受预训练模型的限制。

我们知道目前大部分常用的目标检测算法都是基于anchor的，比如Faster RCNN系列，SSD，YOLO（v2、v3）等，引入anchor后检测效果提升确实比较明显（比如YOLO v1和YOLO v2），但是引入anchor的缺点在于：1、正负样本不均衡。大部分检测算法的anchor数量都成千上万，但是一张图中的目标数量并没有那么多，这就导致正样本数量会远远小于负样本，因此有了对负样本做欠采样以及focal loss等算法来解决这个问题。2、引入更多的超参数，比如anchor的数量、大小和宽高比等。因此这篇不采用anchor却能有不错效果的CornerNet就省去了这几个额外的操作，确实是非常有意思的作品。

CornerNet算法整体结构如Figure4所示。
首先1个7×7的卷积层将输入图像尺寸缩小为原来的1/4（论文中输入图像大小是511×511，缩小后得到128×128大小的输出）。
然后经过特征提取网络（backbone network）提取特征，该网络采用hourglass network，该网络通过串联多个hourglass module组成（Figure4中的hourglass network由2个hourglass module组成），每个hourglass module都是先通过一系列的降采样操作缩小输入的大小，然后通过上采样恢复到输入图像大小，因此该部分的输出特征图大小还是128×128，整个hourglass network的深度是104层。
hourglass module后会有两个输出分支模块，分别表示左上角点预测分支和右下角点预测分支，每个分支模块包含一个corner pooling层和3个输出：heatmaps、embeddings和offsets。heatmaps是输出预测角点信息，可以用维度为CHW的特征图表示，其中C表示目标的类别（注意：没有背景类），这个特征图的每个通道都是一个mask，mask的每个值（范围为0到1，论文中写的该mask是binary mask，也就是0或1，个人感觉是笔误，预测值应该是0到1，否则后面公式1计算损失函数时就没有意思了）表示该点是角点的分数；embeddings用来对预测的corner点做group，也就是找到属于同一个目标的左上角角点和右下角角点；offsets用来对预测框做微调，这是因为从输入图像中的点映射到特征图时有量化误差，offsets就是用来输出这些误差信息。
![image.png](https://upload-images.jianshu.io/upload_images/9730793-8f14d7b7ed3eda95.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
![image.png](https://upload-images.jianshu.io/upload_images/9730793-ec880ff46c01051c.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


接下来分别看看这3个输出和corner pooling的含义。

CornerNet的第一个输出是headmaps，也就是预测角点的位置。
公式1是针对角点预测（headmaps）的损失函数，整体上是改良版的focal loss。几个参数的含义：pcij表示预测的heatmaps在第c个通道（类别c）的(i,j)位置的值，ycij表示对应位置的ground truth，N表示目标的数量。ycij=1时候的损失函数容易理解，就是focal loss，α参数用来控制难易分类样本的损失权重；ycij等于其他值时表示(i,j)点不是类别c的目标角点，照理说此时ycij应该是0（大部分算法都是这样处理的），但是这里ycij不是0，而是用基于ground truth角点的高斯分布计算得到，因此距离ground truth比较近的(i,j)点的ycij值接近1，这部分通过β参数控制权重，这是和focal loss的差别。为什么对不同的负样本点用不同权重的损失函数呢？这是因为靠近ground truth的误检角点组成的预测框仍会和ground truth有较大的重叠面积，如Figure5所示。
![image.png](https://upload-images.jianshu.io/upload_images/9730793-0dd4b1e83f35e125.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

Figure5是关于对不同负样本点的损失函数采取不同权重值的原因。红色实线框是ground truth；橘色圆圈是根据ground truth的左上角角点、右下角角点和设定的半径值画出来的，半径是根据圆圈内的角点组成的框和ground truth的IOU值大于0.7而设定的，圆圈内的点的数值是以圆心往外呈二维的高斯分布；白色虚线是一个预测框，可以看出这个预测框的两个角点和ground truth并不重合，但是该预测框基本框住了目标，因此是有用的预测框，所以要有一定权重的损失返回，这就是为什么要对不同负样本点的损失函数采取不同权重值的原因。
![image.png](https://upload-images.jianshu.io/upload_images/9730793-5c3c2cccb5a20558.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


CornerNet的第二个输出是offset，这个值和目标检测算法中预测的offset类似却完全不一样，说类似是因为都是偏置信息，说不一样是因为在目标检测算法中预测的offset是表示预测框和anchor之间的偏置，而这里的offset是表示在取整计算时丢失的精度信息，也就是公式2所表达的内容。
![image.png](https://upload-images.jianshu.io/upload_images/9730793-7bfbe0607fc29c5c.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

我们知道从输入图像到特征图之间会有尺寸缩小，假设缩小倍数是n，那么输入图像上的(x,y)点对应到特征图上就如下式子

式子中的符号是向下取整，取整会带来精度丢失，这尤其影响小尺寸目标的回归，Faster RCNN中的 ROI Pooling也是有类似的精度丢失问题。所以通过公式2计算offset，然后通过公式3的smooth L1损失函数监督学习该参数，和常见的目标检测算法中的回归支路类似。


CornerNet的第三个输出是embedding，对应文章中group corner的内容。前面介绍了关于角点的检测，在那部分中对角点的预测都是独立的，不涉及一个目标的一对角点的概念，因此如何找到一个目标的两个角点就是第三个输出embedding做的工作。这部分是受associative embedding那篇文章的启发，简而言之就是基于不同角点的embedding vector之间的距离找到每个目标的一对角点，如果一个左上角角点和一个右下角角点属于同一个目标，那么二者的embedding vector之间的距离应该很小。

embedding这部分的训练是通过两个损失函数实现的，etk表示第k个目标的左上角角点的embedding vector，ebk表示第k个目标的右下角角点的embedding vector，ek表示etk和ebk的均值。公式4用来缩小属于同一个目标（第k个目标）的两个角点的embedding vector（etk和ebk）距离。公式5用来扩大不属于同一个目标的两个角点的embedding vector距离。
![image.png](https://upload-images.jianshu.io/upload_images/9730793-2808571031e2def5.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


最后一部分是corner pooling。首先看看为什么要引入corner pooling，如图Figure2所示。因为CornerNet是预测左上角和右下角两个角点，但是这两个角点在不同目标上没有相同规律可循，如果采用普通池化操作，那么在训练预测角点支路时会比较困难。考虑到左上角角点的右边有目标顶端的特征信息（第一张图的头顶），左上角角点的下边有目标左侧的特征信息（第一张图的手），因此如果左上角角点经过池化操作后能有这两个信息，那么就有利于该点的预测，这就有了corner pooling。
![image.png](https://upload-images.jianshu.io/upload_images/9730793-f9a71deec1ec93ad.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


Figure3是针对左上角点做corner pooling的示意图，该层有2个输入特征图，特征图的宽高分别用W和H表示，假设接下来要对图中红色点（坐标假设是(i,j)）做corner pooling，那么就计算(i,j)到(i,H)的最大值（对应Figure3上面第二个图），类似于找到Figure2中第一张图的左侧手信息；同时计算(i,j)到(W,j)的最大值（对应Figure3下面第二个图），类似于找到Figure2中第一张图的头顶信息，然后将这两个最大值相加得到(i,j)点的值（对应Figure3最后一个图的蓝色点）。右下角点的corner pooling操作类似，只不过计算最大值变成从(0,j)到(i,j)和从(i,0)到(i,j)。
![image.png](https://upload-images.jianshu.io/upload_images/9730793-b7efc1154549049c.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


Figure6也是针对左上角点做corner pooling的示意图，是Figure3的具体数值计算例子，该图一共计算了4个点的corner pooling结果。第二列的数值计算和Figure3介绍的一样，比如第一行第一个图中的0值点，计算该点的最大值时是计算该点和其右侧的值为2的点的最大值，因此得到的就是2。
![image.png](https://upload-images.jianshu.io/upload_images/9730793-0c7573f714e4e964.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


Figure7是Figure4中预测模块的详细结构，该结构包括corner pooling模块和预测输出模块两部分，corner pooling模块采用了类似residual block的形式，有一个skip connection，虚线框部分执行的就是corner pooling操作，也就是Figure6的操作，这样整个corner pooling操作就介绍完了。
![image.png](https://upload-images.jianshu.io/upload_images/9730793-b056baf66764f29d.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


模型测试时候的几个细节：1、在得到预测角点后，会对这些角点做NMS操作，选择前100个左上角角点和100个右下角角点。2、计算左上角和右下角角点的embedding vector的距离时采用L1范数，距离大于0.5或者两个点来自不同类别的目标的都不能构成一对。3、测试图像采用0值填充方式得到指定大小作为网络的输入，而不是采用resize，另外同时测试图像的水平翻转图并融合二者的结果。4、最后通过soft-nms操作去除冗余框，只保留前100个预测框。

## 实验结果：
Table1是关于conner pooling的对比实验，可以看出添加conner pooling（第二行）对效果的提升比较明显，这种提升尤其在目标尺度比较大的数据中表现明显。
![image.png](https://upload-images.jianshu.io/upload_images/9730793-44f8b7a68c027d47.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


Table2是关于对不同位置负样本采取不同权重的损失函数的效果。第一行是不采用这种策略的效果；第二行是采用固定半径值的效果，可以看出提升比较明显；第三行是采用基于目标计算得到的半径值的效果（这篇文章所采用的），效果得到进一步提升。
![image.png](https://upload-images.jianshu.io/upload_images/9730793-419efc2292d2385e.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


Table3是关于这篇文章的几个创新点效果。第一行是这篇文章的算法结果；第二行是将角点预测结果用ground truth代替，可以看到提升非常大；第三行是进一步将偏置用ground truth代替，相比之下提升会小很多。这说明目前该算法的瓶颈主要在于角点预测。
![image.png](https://upload-images.jianshu.io/upload_images/9730793-730ee5053191fd55.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

Table4是connerNet和其他目标检测算法的效果对比，可以看出模型效果还是很不错的。
![image.png](https://upload-images.jianshu.io/upload_images/9730793-af10be9f33ce63c1.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


