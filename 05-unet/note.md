#ref
https://github.com/1024210879/unet-denoising-dirty-documents
#为什么unet的分割任务使用交叉熵
可能存在多类
#unet的特点是什么
u型结构(编码+解码)+skip-connection
结合了低分辨率信息(类别特征)和高分辨率信息(精准分割)，组合后适用于分割
#训练过程中发生loss突然为nan 重新训练后必现
减小学习率后还是会出现
tf.clip 约束值在一定范围后解决
#关于unet++
#关于relu的dead relu problem(神经元坏死)
Xavier初始化，目标是每一层输出的方差应该尽量相等
ref：https://www.cnblogs.com/shine-lee/p/11908610.html
#验证集的损失远远小于训练集损失
