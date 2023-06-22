import numpy as np
import tifffile





id_path = r"E:\Dataset\data_description\IDs\validation_ID.txt"
with open(id_path, mode="r") as data_IDs:
    data_ID_ = data_IDs.readlines()
    data_ID = [x.strip("\n") for x in data_ID_]
basic_path = r"E:\Ibrahim_data_copy\total_volume_data_32frames"
count = 0
for id in data_ID:
    print(id)
    mask = tifffile.TiffFile(basic_path + "\\" + id + "_OCTIntPDavg_view_normal_volume_Findconnected region_glassplateremoved.tif").asarray()
    mask_ = np.where(mask>0, 1, 0)
    for index in range(mask_.shape[0]):
        if np.sum(mask_[index]) > 1000:
            count = count + 1
print(count)
