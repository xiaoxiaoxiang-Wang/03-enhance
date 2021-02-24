#ref
https://github.com/1024210879/unet-denoising-dirty-documents
#为什么unet的分割任务使用交叉熵
可能存在多类
#unet的特点是什么
u型结构(编码+解码)+skip-connection
结合了低分辨率信息(类别特征)和高分辨率信息(精准分割)，组合后适用于分割
#训练过程中发生loss突然为nan 重新训练后必现
减小学习率后不再出现
tf.clip 约束值在一定范围
#关于unet++