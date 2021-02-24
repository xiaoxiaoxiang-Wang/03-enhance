from tensorflow import keras


def unet(channel):
    input = keras.layers.Input(shape=(None, None, channel), name='input')
    conv1 = conv_layer(input, 64)
    pool1 = pool(conv1)
    conv2 = conv_layer(pool1, 128)
    pool2 = pool(conv2)
    conv3 = conv_layer(pool2, 256)

    deconv4 = deconv_layer(conv3, conv2, 128)
    deconv5 = deconv_layer(deconv4, conv1, 64)
    conv6 = conv_layer(deconv5, 2, False, False)

    out = keras.layers.Activation(activation="sigmoid")(conv6)
    model = keras.models.Model(inputs=input, outputs=out)
    return model


def double_conv(input, filters):
    conv1 = conv_layer(input, filters)
    conv2 = conv_layer(conv1, filters)
    return conv2


def pool(input):
    return keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="valid")(input)


def conv_layer(input, filters, bn=True, ac=True):
    out = keras.layers.Conv2D(filters=filters, kernel_size=(3, 3), strides=(1, 1), kernel_initializer='Orthogonal',
                              padding='same', use_bias=False)(
        input)
    if bn:
        out = keras.layers.BatchNormalization(axis=-1, momentum=0.0, epsilon=0.0001)(out)
    if ac:
        out = keras.layers.Activation(activation='relu')(out)
    return out


def deconv_layer(input, conv_prev, filter):
    up1 = keras.layers.UpSampling2D(size=(2, 2))(input)
    conv1 = keras.layers.Conv2D(filters=filter, kernel_size=(3, 3), strides=(1, 1), kernel_initializer='Orthogonal',
                                padding='same')(up1)
    concat1 = keras.layers.concatenate([conv_prev, conv1])
    conv2 = double_conv(concat1, filter)
    return conv2


if __name__ == "__main__":
    model = unet(1)
    model.summary()
    keras.utils.plot_model(model, to_file='model.png', show_shapes=True)
