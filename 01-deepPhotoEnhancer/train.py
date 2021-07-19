from tensorflow import keras
import tensorflow.keras.backend as K

from network import D, G

import data_prepare
import numpy as np
import datetime

from functools import partial

from tensorflow.python.framework.ops import disable_eager_execution


class DpeGan():
    def __init__(self):
        self.img_rows = 128
        self.img_cols = 128
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.batch_size = 45
        self.file_per_batch = 1
        self.epochs = 1

        self.model_g_xy = G()
        self.model_g_yx = G()
        self.model_d_x = D()
        self.model_d_y = D()

        self.d_init()
        self.g_init()
        print("model_combine_g.summary")
        self.model_combine_g.summary()
        print("model_combine_d.summary")
        self.model_d_x.summary()

    def g_init(self):
        # Input images from both domains
        x = keras.Input(shape=self.img_shape)
        y = keras.Input(shape=self.img_shape)

        # 固定d网络
        # self.model_d_x.trainable = False
        # self.model_d_y.trainable = False

        gy1 = self.model_g_xy(x)
        gx2 = self.model_g_yx(gy1)

        gx1 = self.model_g_yx(y)
        gy2 = self.model_g_xy(gx1)

        dx1 = self.model_d_x(gx1)
        dy1 = self.model_d_y(gy1)

        # 固定d训练g
        # gy1, gx1 恒等映射 L1 loss
        # dy1, dy2 g网络损失 wasserstein_loss
        # gx2, gy2 一致性损失 L2 loss
        self.model_combine_g = keras.Model(inputs=[x, y],
                                           outputs=[gy1, gx1,
                                                    dx1, dy1,
                                                    gx2, gy2])

        # 学习率衰减设置
        lr_schedule_g = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=1e-4,
            decay_steps=1000,
            decay_rate=0.9)
        optimizer_g = keras.optimizers.SGD(learning_rate=lr_schedule_g)

        # 模型编译
        self.model_combine_g.compile(loss=['mae', 'mae',
                                           self.wasserstein_loss, self.wasserstein_loss,
                                           'mse', 'mse'],
                                     loss_weights=[1, 1, 1, 1, 1, 1],
                                     optimizer=optimizer_g)
        print("model_combine_g.summary")
        self.model_combine_g.summary()

    def d_init(self):
        x = keras.Input(shape=self.img_shape)
        y = keras.Input(shape=self.img_shape)

        gx1 = keras.Input(shape=self.img_shape)
        gy1 = keras.Input(shape=self.img_shape)
        # 避免训练g
        dx = self.model_d_x(x)
        dy = self.model_d_y(y)

        dx1 = self.model_d_x(gx1)
        dy1 = self.model_d_y(gy1)

        # 学习率衰减设置
        lr_schedule_d = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=1e-4,
            decay_steps=1000,
            decay_rate=0.9)
        optimizer_d = keras.optimizers.SGD(learning_rate=lr_schedule_d)

        self.model_d_x.compile(loss=self.wasserstein_loss, optimizer=optimizer_d)
        self.model_d_y.compile(loss=self.wasserstein_loss, optimizer=optimizer_d)

        # Use Python partial to provide loss function with additional
        # 'averaged_samples' argument
        # xy1 = self.interpolate([x, gy1])
        # d_xy1 = self.model_d_y(xy1)
        # gp_d_xy1 = partial(self.gradient_penalty_loss,
        #                    averaged_samples=xy1)
        # gp_d_xy1.__name__ = 'gp_d_xy1'  # Keras requires function names
        #
        # # Use Python partial to provide loss function with additional
        # # 'averaged_samples' argument
        # yx1 = self.interpolate([y, gx1])
        # d_yx1 = self.model_d_x(yx1)
        #
        # gp_d_yx1 = partial(self.gradient_penalty_loss,
        #                    averaged_samples=yx1)
        # gp_d_yx1.__name__ = 'gp_d_yx1'  # Keras requires function names

        # 固定g训练d
        # d网络损失
        # dx, dy1, dy, dx1 wasserstein_loss
        # self.model_combine_d = keras.Model(inputs=[x, y, gy1, gx1],
        #                                    outputs=[dx, dy, dy1, dx1, d_yx1, d_xy1])
        #
        # # 模型编译
        # self.model_combine_d.compile(loss=[self.wasserstein_loss, self.wasserstein_loss,
        #                                    self.wasserstein_loss, self.wasserstein_loss,
        #                                    gp_d_yx1, gp_d_xy1
        #                                    ],
        #                              loss_weights=[1, 1, 1, 1, 1, 1],
        #                              optimizer=optimizer_d)
        print("model_combine_d.summary")
        self.model_d_x.summary()

    def train(self):
        start_time = datetime.datetime.now()
        for epoch in range(self.epochs):
            for batch_i, (x, y, batch_size) in enumerate(data_prepare.loadData(self.file_per_batch)):
                valid = -np.ones((x.shape[0], 1, 1, 1))
                fake = np.ones((x.shape[0], 1, 1, 1))
                dummy = np.ones((x.shape[0], 1, 1, 1))
                # 先计算x1和y1，避免训练g网络
                gy1 = self.model_g_xy(x)
                gx1 = self.model_g_yx(y)

                # 固定g训练d
                # d网络损失
                # dx, dy1, dy, dx1 wasserstein_loss
                # d_loss_x = self.model_combine_d.train_on_batch(x=[x, y, gy1, gx1],
                #                                              y=[valid, valid, fake, fake, dummy, dummy])
                d_loss_x = self.model_d_x.train_on_batch(x, valid)
                d_loss_x1 = self.model_d_x.train_on_batch(gx1, fake)
                d_loss_y = self.model_d_y.train_on_batch(y, valid)
                d_loss_y1 = self.model_d_y.train_on_batch(gy1, valid)
                # 固定d训练g
                # gy1, gx1 恒等映射 L1 loss
                # dy1, dx1 g网络损失 wasserstein_loss
                # gx2, gy2 一致性损失 L2 loss
                g_loss = self.model_combine_g.train_on_batch(x=[x, y],
                                                             y=[x, y,
                                                                valid, valid,
                                                                x, y])
                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss:total loss=%f, dx=%f, dy1=%f, dy=%f, dx1=%f,it_xy1=%f, it_yx1=%f] [G loss:total loss=%f, x-y1 mae=%f, y-x1 mae=%f, dy1=%f, dx1=%f, x-x2 mse=%f, y-y2 mse=%f] time: %s " \
                    % (epoch, self.epochs,
                       batch_i, batch_size,
                       d_loss_x, d_loss_x, d_loss_x1, d_loss_x1, d_loss_y, d_loss_y, d_loss_y1,
                       g_loss[0], g_loss[1], g_loss[2], g_loss[3], g_loss[4], g_loss[5], g_loss[6],
                       datetime.datetime.now() - start_time))
        self.model_d_x.save("./models/model_d_x.h5")
        self.model_d_y.save("./models/model_d_y.h5")
        self.model_g_xy.save("./models/model_g_xy.h5")
        self.model_g_yx.save("./models/model_g_yx.h5")

    def wasserstein_loss(self, y_true, y_pred):
        print("xxxxxxxxxxxx", y_true.shape, y_pred.shape, keras.backend.mean(y_true * y_pred).shape)
        return keras.backend.mean(y_true * y_pred)

    def gradient_penalty_loss(self, y_true, y_pred, averaged_samples):
        """
        Computes gradient penalty based on prediction and weighted real / fake samples
        """
        gradients = K.gradients(y_pred, averaged_samples)
        # compute the euclidean norm by squaring ...
        gradients_sqr = K.square(gradients)
        #   ... summing over the rows ...
        gradients_sqr_sum = K.sum(gradients_sqr,
                                  axis=np.arange(1, len(gradients_sqr.shape)))
        #   ... and sqrt
        gradient_l2_norm = K.sqrt(gradients_sqr_sum)
        # compute lambda * (1 - ||grad||)^2 still for each single sample
        gradient_penalty = K.square(1 - gradient_l2_norm)
        # return the mean as loss over all the batch samples
        return K.mean(gradient_penalty)

    def interpolate(self, inputs):
        alpha = K.random_uniform((self.batch_size, 1, 1, 1))
        return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])


if __name__ == '__main__':
    disable_eager_execution()
    dpe = DpeGan()
    dpe.train()
