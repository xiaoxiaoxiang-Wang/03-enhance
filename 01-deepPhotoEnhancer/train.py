from tensorflow import keras

from network import D, G

x, y = 1, 1
model_g_xy = G()
model_g_yx = G()
model_d_x = D()
model_d_y = D()

gy1 = model_g_xy(x)
gx2 = model_g_yx(gy1)

gx1 = model_g_yx(y)
gy2 = model_g_xy(gx1)

dx = model_d_x(x)
dx1 = model_d_x(gx1)
dx2 = model_d_x(gx2)

dy = model_d_y(y)
dy1 = model_d_y(gy1)
dy2 = model_d_y(gy2)

# 固定d训练g
# gy1, gx1 恒等映射 L1 loss
# dy1, dy2 g网络损失 wasserstein_loss
# gx2, gy2 一致性损失 L2 loss
model_combine_g = keras.Model(inputs=[x, y],
                              outputs=[gy1, gx1,
                                       dy1, dy2,
                                       gx2, gy2])

# 学习率衰减设置
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-4,
    decay_steps=1000,
    decay_rate=0.9)
optimizer = keras.optimizers.SGD(learning_rate=lr_schedule)

# 模型配置
model_combine_g.compile(loss=['mae', 'mae',
                              'wasserstein_loss', 'wasserstein_loss',
                              'mse', 'mse'],
                        loss_weights=[1, 1, 1, 1, 10, 10],
                        optimizer=optimizer)

# 固定g训练d
# d网络损失
# dx, dy1, dy, dx1 wasserstein_loss
model_combine_d = keras.Model(inputs=[x, y],
                              outputs=[dx, dy1, dy, dx1])

# 模型配置
model_combine_d.compile(loss=['wasserstein_loss', 'wasserstein_loss',
                              'wasserstein_loss', 'wasserstein_loss'
                              ],
                        loss_weights=[1, 1, 1, 1],
                        optimizer=optimizer)


def train():
    model_combine_g.train_on_batch()
    model_combine_d.train_on_batch()
    pass


def wasserstein_loss(y_true, y_pred):
    return keras.backend.mean(y_true * y_pred)
