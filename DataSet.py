import SimpleITK as sitk
from matplotlib import pyplot as plt


def showNii(img):
    for i in range(img.shape[0]):
        plt.imshow(img[i, :, :], cmap='gray')
        plt.show()


itk_img = sitk.ReadImage('./data/ct_train_1001_label.nii.gz')
img = sitk.GetArrayFromImage(itk_img)
print(img.shape)  # (88, 132, 175)表示各个维度切片的数量
showNii(img)
