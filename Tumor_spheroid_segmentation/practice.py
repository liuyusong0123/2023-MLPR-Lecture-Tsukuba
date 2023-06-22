import numpy as np
import tifffile
train_data = np.load(r"E:\Segmentation\Lecture\Dataset\train_data_500.npy")
tifffile.imsave(r"E:\Segmentation\Lecture\Dataset\OCT_500.tif", 10*np.log10(train_data[:,0]).astype('f4'))
tifffile.imsave(r"E:\Segmentation\Lecture\Dataset\mask_500.tif", train_data[:,1].astype('f4'))