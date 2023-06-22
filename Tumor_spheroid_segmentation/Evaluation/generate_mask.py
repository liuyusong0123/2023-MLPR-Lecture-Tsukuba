import numpy as np
import tifffile
import tensorflow.keras as keras



def select_frame(index):
    out = np.zeros([32])
    block = int(index/16)
    index2 = index % 16
    for i in range(32):
        out[i] = 512*block + index2 + 16 * i
    out2 = [int(out[0])]
    return out2

trained_model_path = r"E:\Segmentation\Lecture\Dataset\NNmodel\model.h5"
model = keras.models.load_model(trained_model_path)
save_basic_path = r"E:\Segmentation\Lecture\Dataset\NNmodel\mask"
basic_data_path = r"E:\Ibrahim_data_copy\total_volume_data_32frames"
data_IDs = ["MCF7_Spheroid_20210707_030"]
OCT_type = "_OCTIntensityPDavg.tiff"
mask_type = "_NNmask.tif"
for data_ID in data_IDs:
    print(data_ID)
    root = save_basic_path + "\\" + data_ID
    save_path = save_basic_path + "\\" + data_ID + mask_type
    OCT_path = basic_data_path + "\\" + data_ID + OCT_type
    OCT = tifffile.TiffFile(OCT_path).asarray()[:, :384, :]
    OCT_frame = np.zeros([128, 384, 512])
    for index in range(128):
        OCT_frame[index] = OCT[select_frame(index)]
    OCT_frame = 10*np.log10(OCT_frame)
    mask = np.zeros([128, 384, 512])
    for x in range(128):
        OCT_train = OCT_frame[x]
        OCT_train = np.expand_dims(OCT_train, axis=0)
        mask[x] = np.squeeze(model(OCT_train, training=False).numpy())
    tifffile.imsave(save_path, mask.astype(dtype='f4'))



