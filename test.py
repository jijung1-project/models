import numpy as np, array
from PIL import Image
import os

path = '/Users/jaewan/Desktop/food_img'
dir = os.listdir(path)
wholelist = []
print(dir)
count = 0

for d in range(5):
    folderdir = path + "/"+ dir[d] + "/"
    print(folderdir)
    for item in os.listdir(folderdir):
        imgpath = folderdir + item
        img = Image.open(imgpath)
        arr = np.array(img)
        wholelist.append(arr)
        count += 1
        print(count)

print(count)
wholeArr = np.array(wholelist)
print(wholeArr.shape)