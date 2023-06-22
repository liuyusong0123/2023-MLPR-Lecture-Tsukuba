import tensorflow as tf
import tensorflow.keras as keras

def max_pool(input):
    maxpool = keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='same')
    return maxpool(input)

def conv(input, filter, kernel):
    con = keras.layers.Conv2D(filters=filter, kernel_size=kernel, padding='same')
    bn = keras.layers.BatchNormalization()
    af = keras.layers.ReLU()
    x1 = bn(con(input))
    return af(x1)
def conv_woaf(input, filter, kernel):
    con = keras.layers.Conv2D(filters=filter, kernel_size=kernel, padding='same')
    bn = keras.layers.BatchNormalization()
    x1 = bn(con(input))
    return x1
def double_conv(x, filter1, kernel1, filter2, kernel2):
    x1 = conv(x, filter1, kernel1)
    x2 = conv(x1, filter2, kernel2)
    return x2

def double_conv_woaf(x, filter1, kernel1, filter2, kernel2):
    x1 = conv_woaf(x, filter1, kernel1)
    x2 = conv_woaf(x1, filter2, kernel2)
    return x2

def up (x, y, filter1, filter2, kernel2, filter3, kernel3):
    dconv = keras.layers.Conv2DTranspose(filter1, (2, 2), strides=2)
    y1 = dconv(y)
    xy = tf.concat([x, y1], axis=3)
    return double_conv(xy, filter2, kernel2, filter3, kernel3)







