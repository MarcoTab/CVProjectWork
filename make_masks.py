import matplotlib.pyplot as plt
import numpy as np
import os


filedirs = os.path.join("yolo_outs")
newoutdir = "yolo_masks"

os.makedirs(newoutdir, exist_ok=True)

for dir in os.listdir(filedirs):

    for file in os.listdir(os.path.join(filedirs, dir)):
        arr = np.loadtxt(os.path.join(filedirs, dir, file), dtype=int)

        img = np.zeros((900,1600))

        img[arr[0]:arr[2], arr[1]:arr[2]] = 255

        plt.imsave(os.path.join(newoutdir, f"{dir}.png"), img, cmap="gray")

        # exit()