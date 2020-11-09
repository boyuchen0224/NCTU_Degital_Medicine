import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os

def get_origin_img_list():
    img = pd.read_csv('..\\Train_png_all\\train_img.csv', header=None)
    label = pd.read_csv('..\\Train_png_all\\train_label.csv', header=None)
    return np.squeeze(img.values), np.squeeze(label.values)


if __name__ == "__main__":
    img_root = "..\\Train_png_all\\photo\\"
    new_img_file = "..\\Train_png_all\\aug_train_img.csv"
    new_label_file = "..\\Train_png_all\\aug_train_label.csv"

    images, labels = get_origin_img_list()

    N = len(images)
    new_images = images.copy()
    new_labels = labels.copy()
    for idx in range(N):

        if labels[idx] >= 0:
            img_name = os.path.join(img_root, images[idx])
            
            raw_img = Image.open(img_name)
            aug_img_1 = raw_img.rotate( 90, Image.BILINEAR )
            aug_img_2 = raw_img.rotate( 180, Image.BILINEAR )
            aug_img_3 = raw_img.rotate( 270, Image.BILINEAR )

            # *** for test ***
            # plt.figure()
            # plt.subplot(121)
            # plt.imshow(raw_img)
            # plt.subplot(122)
            # plt.imshow(aug_img_1)
            # plt.show()

            aug_img_1.save( img_root + 'aug1_' + images[idx] )
            new_images = np.append(new_images,'aug1_'+images[idx])
            new_labels = np.append(new_labels, labels[idx])

            aug_img_2.save( img_root + 'aug2_' + images[idx] )
            new_images = np.append(new_images,'aug2_'+images[idx])
            new_labels = np.append(new_labels, labels[idx])

            aug_img_3.save( img_root + 'aug3_' + images[idx] )
            new_images = np.append(new_images,'aug3_'+images[idx])
            new_labels = np.append(new_labels, labels[idx])

            print(images[idx] + "  " + str(labels[idx]))

        # if idx >20:
        #     break

    np.savetxt(new_img_file, new_images, fmt='%s', delimiter=",")
    np.savetxt(new_label_file, new_labels, fmt='%d', delimiter=",")