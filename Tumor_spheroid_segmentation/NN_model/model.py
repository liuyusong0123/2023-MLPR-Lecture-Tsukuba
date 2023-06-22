


from part import *




def UNet_structure(input, kernelsize):
    x1 = double_conv(input, 64,  kernelsize, 64, kernelsize)
    x2_1 = max_pool(x1)
    x2 = double_conv(x2_1, 128, kernelsize, 128, kernelsize)
    x3_1 = max_pool(x2)
    x3 = double_conv(x3_1, 256, kernelsize, 256, kernelsize)
    x4_1 = max_pool(x3)
    x4 = double_conv(x4_1, 512, kernelsize, 512, kernelsize)
    x5_1 = max_pool(x4)
    x5 = double_conv(x5_1, 1024, kernelsize, 1024, kernelsize)
    y4 = up(x4, x5, 512, 512, kernelsize, 512, kernelsize)
    y3 = up(x3, y4, 256, 256, kernelsize, 256, kernelsize)
    y2 = up(x2, y3, 128, 128, kernelsize, 128, kernelsize)
    y1 = up(x1, y2, 64, 64,  kernelsize, 32,  kernelsize)
    y0 = double_conv_woaf(y1, 1,  kernelsize, 1, (1, 1))
    y1 = keras.activations.sigmoid(y0)
    return y1



def model_():
    input_ = keras.layers.Input((None, None, 1))
    output_ = UNet_structure(input_, (3, 3))
    return keras.Model(inputs=[input_], outputs=[output_])