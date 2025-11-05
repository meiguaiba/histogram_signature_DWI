import pickle
import numpy as np
import xlwt
from sklearn.model_selection import train_test_split
NUM_aug = 100
LEN = 700
random_div = 8888
augment_datafile = '/DWI/histogram_signature_augmentation100_30_65_bin70.pkl'
with open (augment_datafile, 'rb') as pickle_file:
    (Xdata_aug, Ydata_aug)=pickle.load(pickle_file,encoding="latin-1")


original_datafile = '/DWI/histogram_signature_original_30_65_bin70.pkl'
with open (original_datafile, 'rb') as pickle_file:
    (Xdata, Ydata)=pickle.load(pickle_file,encoding="latin-1")

workbook = xlwt.Workbook(encoding='utf-8')
worksheet = workbook.add_sheet('train')

ODER = np.arange(0, 191)

TRAIN_VALID = ODER[:90]
TEST = ODER[91:]
TRAIN, VALID = train_test_split(TRAIN_VALID, test_size=0.2, random_state=random_div, stratify=Ydata[TRAIN_VALID])



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Y_train = Ydata[TRAIN]
NUM_pos = Y_train.sum()
NUM_neg = Y_train.shape[0]-NUM_pos

NUM_aug_pos = NUM_aug
NUM_aug_neg = np.round(NUM_aug * NUM_pos / NUM_neg)
NUM_aug_neg = NUM_aug_neg.astype('int64')


NUM_total = NUM_aug_pos * NUM_pos + NUM_aug_neg * NUM_neg
NUM_total = NUM_total.astype('int64')

X_train = np.zeros([NUM_total, LEN])
Y_train = np.zeros(NUM_total)

ROW = 0
LINE = 0
for ii in TRAIN:
    worksheet.write(ROW, 0, float(ii))
    worksheet.write(ROW, 1, float(Ydata[ii]))
    ROW += 1
    if Ydata[ii] == 1:
        for jj in range(NUM_aug_pos):
            X_train[LINE,:] = Xdata_aug[ii,:,jj]
            Y_train[LINE] = Ydata[ii]
            LINE += 1
    else:
        for jj in range(NUM_aug_neg):
            X_train[LINE,:] = Xdata_aug[ii,:,jj]
            Y_train[LINE] = Ydata[ii]
            LINE += 1
print('Train:')
print(NUM_total)

X_train = np.concatenate((X_train, Xdata[TRAIN,:]),axis=0)
Y_train = np.concatenate((Y_train, Ydata[TRAIN]),axis=0)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
worksheet = workbook.add_sheet('validation')

Y_valid = Ydata[VALID]
NUM_posV = Y_valid.sum()
NUM_negV = Y_valid.shape[0]-NUM_posV

NUM_aug_posV = NUM_aug
NUM_aug_negV = np.round(NUM_aug * NUM_posV / NUM_negV)
NUM_aug_negV = NUM_aug_negV.astype('int64')


NUM_totalV = NUM_aug_posV * NUM_posV + NUM_aug_negV * NUM_negV
NUM_totalV = NUM_totalV.astype('int64')

X_valid = np.zeros([NUM_totalV, LEN])
Y_valid = np.zeros(NUM_totalV)

ROW = 0
LINE = 0
for ii in VALID:
    worksheet.write(ROW, 0, float(ii))
    worksheet.write(ROW, 1, float(Ydata[ii]))
    ROW += 1
    if Ydata[ii] == 1:
        for jj in range(NUM_aug_posV):
            X_valid[LINE,:] = Xdata_aug[ii,:,jj]
            Y_valid[LINE] = Ydata[ii]
            LINE += 1
    else:
        for jj in range(NUM_aug_negV):
            X_valid[LINE,:] = Xdata_aug[ii,:,jj]
            Y_valid[LINE] = Ydata[ii]
            LINE += 1
print('VALID:')
print(NUM_totalV)

X_valid = np.concatenate((X_valid, Xdata[VALID,:]),axis=0)
Y_valid = np.concatenate((Y_valid, Ydata[VALID]),axis=0)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

X_test = Xdata[TEST,:]
Y_test = Ydata[TEST]

with open ('/DWI/spine_n191_30_65_bin70.pkl', 'wb') as pickle_file:
    pickle.dump(((X_train, Y_train),(X_valid, Y_valid),(X_test, Y_test)), pickle_file)

workbook.save('/DWI/trainvalidtest_split.xls')