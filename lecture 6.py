import numpy as np
from matplotlib import pyplot as plt
import cv2
import pandas as pd

x = np.arange(100)
print(x)
y = np.sin(x)
print(y)

'''plt.scatter(x, y)
plt.show()
plt.plot(x, y)
plt.show()'''

# Loading and showing images in Open cv
# For these libraries the data should be present in the working directory or the path should be mentioned
img = cv2.imread('lecture_6_image.jpeg')
'''cv2.imshow('car', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)
plt.show()

# Pandas
dataset = pd.read_csv('lecture_6_dataset.csv')
print(dataset.shape)
X = dataset.iloc[2, 1:20].values
Y = dataset.iloc[3, 1:20].values
print(X)
print(Y)
