import os
import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt
from tensorflow.keras.utils import normalize
import cv2
from PIL import Image
from sklearn.model_selection import train_test_split
from simple_unet_model import simple_unet_model

# Set directory paths
image_dir = 'C:/Users/john_/OneDrive/Desktop/mito/generated_patches/images/'
mask_dir = 'C:/Users/john_/OneDrive/Desktop/mito/generated_patches/masks/'

SIZE = 256
image_dataset = []  
mask_dataset = []
images = os.listdir(image_dir)
for i, image_name in enumerate(images):   
    if (image_name.split('.')[1] == 'tif'):
        image = cv2.imread(os.path.join(image_dir, image_name), 0)
        image = Image.fromarray(image)
        image = image.resize((SIZE, SIZE))
        image_dataset.append(np.array(image))

masks = os.listdir(mask_dir)
for i, image_name in enumerate(masks):
    if (image_name.split('.')[1] == 'tif'):
        image = cv2.imread(os.path.join(mask_dir, image_name), 0)
        image = Image.fromarray(image)
        image = image.resize((SIZE, SIZE))
        mask_dataset.append(np.array(image))

# Normalize images
image_dataset = np.expand_dims(normalize(np.array(image_dataset), axis=1), 3)

# Normalize masks and threshold them
mask_dataset = np.array(mask_dataset) / 255.
mask_dataset[mask_dataset > 0.5] = 1
mask_dataset[mask_dataset <= 0.5] = 0
mask_dataset = np.expand_dims(mask_dataset, 3)

X_train, X_test, y_train, y_test = train_test_split(image_dataset, mask_dataset, test_size=0.10, random_state=0)

import random
import numpy as np
image_number = random.randint(0, len(X_train))
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.imshow(np.reshape(X_train[image_number], (256, 256)), cmap='gray')
plt.subplot(122)
plt.imshow(np.reshape(y_train[image_number], (256, 256)), cmap='gray')
plt.show()

IMG_HEIGHT = image_dataset.shape[1]
IMG_WIDTH  = image_dataset.shape[2]
IMG_CHANNELS = image_dataset.shape[3]

def get_model():
    return simple_unet_model(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

model = get_model()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, batch_size=16, verbose=1, epochs=20, validation_data=(X_test, y_test), shuffle=False)
model.save('mitochondria_test.hdf5')

# plot the training and validation accuracy and loss at each epoch
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
# IOU
y_pred = model.predict(X_test)
y_pred_thresholded = (y_pred > 0.5).astype(int)
intersection = np.logical_and(y_test, y_pred_thresholded)
union = np.logical_or(y_test, y_pred_thresholded)
iou_score = np.sum(intersection) / np.sum(union)
print("IoU score is:", iou_score)