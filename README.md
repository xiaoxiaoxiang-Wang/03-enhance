# enhance

1.有少量成对训练集，大量未标注output   
弱监督，可以通过gan学习如何生成input，这样就得到了大量的label input，参考Deblurring by Realistic Blurring  
2.有少量成对训练集，大量未标注input，或者说只有人工构造的pix2pix训练集，以及一些实际场景input  
弱监督，有标签和无标签input通过unet编码结构生成雨、雾、噪声自身的分布，参考Syn2Real Transfer Learning for Image Deraining using Gaussian Processes  
3.训练集非pix2pix  
cyclegan  
4.知道模糊核，非盲去模糊

# 图像增强领域的研究方向

1.超分辨率 2.图像去雨 3.图像去雾 4.去模糊 5.去噪 6.图像恢复 7.图像增强 8.图像修复
2/3/5 可以归为加性模型，4归为卷积模型

# 研究方向

现有技术无法真实模拟下雨、起雾、模糊，导致算法训练中使用的合成数据集和真实图像降质有差异，即存在domain shift解决此差异是当前研究重点， Syn2Real Transfer Learning for Image Deraining
using Gaussian Processes和Domain Adaptation for Image Dehazing都是解决这类问题的