# image enhancement algorithm based on gan

# paper

Deep Photo Enhancer: Unpaired Learning for Image Enhancement From Photographs With GANs  
https://openaccess.thecvf.com/content_cvpr_2018/papers/Chen_Deep_Photo_Enhancer_CVPR_2018_paper.pdf

# 图像质量评价

美学评价 ref：https://arxiv.org/pdf/1709.05424.pdf
NIMA: Neural Image Assessment

# 空间位移不变

# 解决的问题？

超透镜和超构透镜 超透镜为什么没有大规模应用？ 分辨率+成像质量 超构透镜没有聚焦整个可见光谱 薄膜透镜的优势： 传统透镜组合的目的是去除色差

# 损失函数设计

1.G网络损失 恒等映射  
MSE(x,y1)+MSE(y,x1)  
2.G网络损失 一致性损失  
MSE(x,x2)+MSE(y,y2)  
3.D网络损失  
E(D(x))-E(D(x1))  
E(D(y))-E(D(y1))  
正则项1-Lipschitz约束  
4.G网络损失 D(x1)  
-G(D(y1))

loss可能为负数

# 常用的损失函数

均方误差：提升psnr利器，使用L2损失  
L1损失：较好的保持亮度和颜色，可以先做高斯模糊消除纹理和内容的影响  
内容损失：特征图的欧式距离作为内容损失，特征图可以使用其他独立的网络，如vgg，有时候作为感知损失的组成部分  
Lipschitz约束：输出对输入，即y对x的导数要小于1，使得y不随x的变化而产生较大的波动  
SSIM loss：结构相似，包含了亮度、对比度和结构信息  
MS-SSIM loss：多尺度结构相似性，保留高频信息，容易导致亮度的改变和颜色的偏差  
style loss：Gram matrix，计算两两特征的相关性  
Total Variation Regularization：全变分，和Lipschitz约束的区别是，全变分是y内部的导数，Lipschitz是y对x的导数    
感知损失：高阶特征的相似性，例如两张完全不同梵高的画，高阶特征可能基本一致 eg EnlightenGAN使用预训练的vgg特征向量平方和作为误差,原文出自Perceptual loss for Real time Style
Transfer and Super-Resolution 循环一致性损失C： d-gan损失改进：

# 图像增强的场景

噪声、模糊、对比度、纹理细节、色彩、视觉伪影

# 训练过程中存在的问题

如何避免gan放大噪声？ 不收敛：模型参数振荡，不稳定，永不收敛，D过早的训练好，real和generator的数据没有重叠，G不再学习,原因是D(x) = 1 x属于正样本，否则D(x) = 0,阶跃函数导数为零  
模式崩溃：发生器坍塌，产生有限的样品种类，损失函数没有关注数据分布导致  
衰减梯度：鉴别器过早的训练成功，real和generator的数据没有重叠，G不再学习  
发电机和鉴别器之间的不平衡导致过度拟合，和 对超参数选择非常敏感。

# gan训练的问题

TOWARDS PRINCIPLED METHODS FOR TRAINING GENERATIVE ADVERSARIAL NETWORKS

# name_scope和variable_scope的用法

tf.variable_scope可以让变量有相同的命名，包括tf.get_variable得到的变量，还有tf.Variable的变量  
tf.name_scope可以让变量有相同的命名，只是限于tf.Variable的变量 即tf.get_variable忽略name_scope

# tf.get_variable和tf.Variable的区别

tf.get_variable获取变量，如果没有就创建  
tf.Variable创建变量

# keras compile

用于配置训练模型

# keras model train_on_batch

model定义张量，train_on_batch传入张量值

# sigmoid和tanh

sigmoid范围0~1 tanh范围-1~1