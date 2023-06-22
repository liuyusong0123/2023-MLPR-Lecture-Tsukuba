import numpy as np

import tifffile


def select_frame(index):
    out = np.zeros([32])
    block = int(index/16)
    index2 = index % 16
    for i in range(32):
        out[i] = 512*block + index2 + 16 * i
    out2 = [int(out[0])]
    return out2


id_path = r"E:\Dataset\data_description\IDs\train_ID.txt"
with open(id_path, mode="r") as data_IDs:
    data_ID_ = data_IDs.readlines()
    data_ID = [x.strip("\n") for x in data_ID_]
basic_path = r"E:\Ibrahim_data_copy\total_volume_data_32frames"
i = 0
save_data = np.zeros([3975, 2, 384, 512])


for id in data_ID:
    print(id)
    OCT = tifffile.TiffFile(basic_path + "\\" + id + "_OCTIntensityPDavg.tiff").asarray()[:, :384, :]
    mask = tifffile.TiffFile(basic_path + "\\" + id + "_OCTIntPDavg_view_normal_volume_Findconnected region_glassplateremoved.tif").asarray()
    mask_ = np.where(mask > 0, 1, 0)
    mask_ = mask_[:, :384, :]
    print(mask.shape)
    for index in range(mask_.shape[0]):
        if np.sum(mask_[index]) > 1000:
            save_data[i, 0] = OCT[select_frame(index)]
            save_data[i, 1] = mask_[index]
            i = i + 1
np.save(r"E:\Segmentation\Lecture\Dataset\train_data.npy", save_data)
print(save_data[-1, 0])
