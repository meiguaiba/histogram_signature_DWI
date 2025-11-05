import numpy as np
import os
import SimpleITK as sitk
import pickle
import pandas as pd

src_root = '/DWI'
labelfilename = '/DWI/label.xlsx'

src_root_vec = src_root.split(os.path.sep)
APEN_len=0
MAX=0
MIN=10000
BINS=70

Patient_num = 0
aug_NUM = 100

Xdata = np.zeros([191, BINS*10, aug_NUM])

label = pd.read_excel(labelfilename)
Ydata = label.values[:,-1]


for patient_root, subdirs1, files in os.walk(src_root):
    patient_root_vec = patient_root.split(os.path.sep)
    if len(patient_root_vec) != len(src_root_vec)+1:
        continue
    ID = patient_root_vec[-1]


    if APEN_len==0:
        ID=ID
    else:
        ID = ID[:-APEN_len]
    nii_number = 0
    print(ID)


    for file in files:
        if 'ROI' in file:
            MaskImage = sitk.ReadImage(os.path.join(patient_root,file))
            MaskArray0 = sitk.GetArrayFromImage(MaskImage)
            ROIval = np.unique(MaskArray0)
            for RV in ROIval:
                if RV != 0:
                    for aug in range(aug_NUM):
                        START = 0

                        np.random.seed(aug)
                        randMask = np.random.rand(MaskArray0.shape[0],MaskArray0.shape[1],MaskArray0.shape[2])

                        MaskArray = (MaskArray0 == RV)*randMask
                        MaskArray = (MaskArray>0.8)

                        for file in files:

                            if file.endswith('.nii') and 'ROI' not in file:

                                Image = sitk.ReadImage(os.path.join(patient_root,file))
                                Array = sitk.GetArrayFromImage(Image)
                                Array = np.log(Array * MaskArray + 1)
                                arr = Array.flatten()
                                arr = arr[arr>0]
                                nii_number += 1
                                n,bins = np.histogram(arr, bins=BINS, range=(3.0, 6.5))
                                Xdata[Patient_num, START:START + BINS, aug] = n
                                START += BINS
                    Patient_num += 1

with open (os.path.join(src_root, 'histogram_signature_augmentation100_30_65_bin70.pkl'), 'wb') as pickle_file:
    pickle.dump((Xdata, Ydata), pickle_file)