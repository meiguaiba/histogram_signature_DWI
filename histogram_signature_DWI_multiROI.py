import numpy as np
import os
import SimpleITK as sitk
import pickle
import xlwt
import pandas as pd

workbook = xlwt.Workbook(encoding='utf-8')
worksheet = workbook.add_sheet('ID')

src_root = '/DWI'
labelfilename = '/DWI/label.xlsx'
src_root_vec = src_root.split(os.path.sep)
APEN_len=0
MAX=0
MIN=10000
BINS=70
Xdata = np.zeros([191, BINS*10])
Patient_num = 0

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

    print(ID)

    for file in files:
        if file.endswith('.nii') and 'ROI' in file:
            MaskImage = sitk.ReadImage(os.path.join(patient_root,file))
            MaskArray = sitk.GetArrayFromImage(MaskImage)


            ROIval = np.unique(MaskArray)

            for RV in ROIval:
                if RV != 0:
                    START = 0
                    for file in files:

                        if file.endswith('.nii') and 'ROI' not in file:

                            Image = sitk.ReadImage(os.path.join(patient_root,file))
                            Array = sitk.GetArrayFromImage(Image)
                            Array = np.log(Array*(MaskArray == RV)+1)
                            arr = Array.flatten()
                            arr = arr[arr>0]

                            n,bins = np.histogram(arr, bins=BINS, range=(3.0,6.5))
                            Xdata[Patient_num, START:START + BINS] = n

                            START += BINS
                    worksheet.write(Patient_num, 0, float(ID))
                    Patient_num += 1


with open (os.path.join(src_root, 'histogram_signature_original_30_65_bin70.pkl'), 'wb') as pickle_file:
    pickle.dump((Xdata, Ydata), pickle_file)
workbook.save(os.path.join(src_root, 'ID.xls'))