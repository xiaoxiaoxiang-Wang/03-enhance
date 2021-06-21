from tensorflow import keras


def G(channel=1):
    input = keras.layers.Input(shape=(None, None, channel), name='input')
    conv0 = conv_layer(input, 48)
    conv1 = conv_layer(conv0, 48)
    pool1 = pool(conv1)
    conv2 = conv_layer(pool1, 48)
    pool2 = pool(conv2)
    conv3 = conv_layer(pool2, 48)
    pool3 = pool(conv3)
    conv4 = conv_layer(pool3, 48)
    pool4 = pool(conv4)
    conv5 = conv_layer(pool4, 48)
    pool5 = pool(conv5)
    conv6 = conv_layer(pool5, 48)

    deconv5 = deconv_layer(conv6, pool4, 96)
    deconv4 = deconv_layer(deconv5, pool3, 96)
    deconv3 = deconv_layer(deconv4, pool2, 96)
    deconv2 = deconv_layer(deconv3, pool1, 96)

    deconv1 = keras.layers.concatenate([keras.layers.UpSampling2D(size=(2, 2))(deconv2), input], axis=-1)
    deconv1A = conv_layer(deconv1, 64)
    deconv1B = conv_layer(deconv1A, 32)
    out = conv_layer(deconv1B, channel, bn=False, ac=False)

    out = keras.layers.Subtract()([input, out])
    model = keras.models.Model(inputs=input, outputs=out)
    return model


def D(channel=1):
    input = keras.layers.Input(shape=(256, 256, channel), name='input')
    conv0 = conv_layer(input, 16, (3, 3), (2, 2))  # (batch_size,128,128,16)
    conv1 = conv_layer(conv0, 32, (5, 5), (2, 2))  # (batch_size,64,64,32)
    conv2 = conv_layer(conv1, 64, (5, 5), (2, 2))  # (batch_size,32,32,64)
    conv3 = conv_layer(conv2, 128, (5, 5), (2, 2))  # (batch_size,16,16,128)
    out = conv_layer(conv3, 1, (16, 16), (1, 1), 'valid')  # (batch_size,128,128,16)
    model = keras.models.Model(inputs=input, outputs=out)
    return model


def double_conv(input, filters):
    conv1 = conv_layer(input, filters)
    conv2 = conv_layer(conv1, filters)
    return conv2


def pool(input):
    return keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="valid")(input)


def conv_layer(input, filters, kernel_size=(3, 3), strides=(1, 1), padding='same', bn=True, ac=True):
    out = keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides,
                              kernel_initializer=keras.initializers.VarianceScaling(scale=1.0, mode='fan_in',
                                                                                    distribution='normal'),
                              bias_initializer='zeros',
                              padding=padding, use_bias=False)(
        input)
    if ac:
        out = keras.layers.Activation(activation='relu')(out)
    if bn:
        out = keras.layers.BatchNormalization(axis=-1, momentum=0.0, epsilon=0.0001)(out)
    return out


def deconv_layer(input, conv_prev, filter):
    up1 = keras.layers.UpSampling2D(size=(2, 2))(input)
    add1 = keras.layers.concatenate([up1, conv_prev], axis=-1)
    conv1 = keras.layers.Conv2D(filters=filter, kernel_size=(3, 3), strides=(1, 1),
                                kernel_initializer=keras.initializers.VarianceScaling(scale=1.0, mode='fan_in',
                                                                                      distribution='normal'),
                                bias_initializer='zeros',
                                padding='same')(add1)
    conv2 = keras.layers.Conv2D(filters=filter, kernel_size=(3, 3), strides=(1, 1),
                                kernel_initializer=keras.initializers.VarianceScaling(scale=1.0, mode='fan_in',
                                                                                      distribution='normal'),
                                bias_initializer='zeros',
                                padding='same')(conv1)
    return conv2


if __name__ == "__main__":
    model = D()
    model.summary()
    keras.utils.plot_model(model, to_file='model.png', show_shapes=True)
