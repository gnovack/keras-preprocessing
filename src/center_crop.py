import matplotlib.pyplot as plt
import tensorflow as tf

BATCH_SIZE = 32
CROP_HEIGHT = 250
CROP_WIDTH = 250

#Load images from the data/image directory
image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
data_generator = image_generator.flow_from_directory(directory=str('/data/image'),
                                                    shuffle=True,
                                                    batch_size=BATCH_SIZE)

#Create a model with a single CenterCrop layer
center_crop_layer = tf.keras.layers.experimental.preprocessing.CenterCrop(CROP_HEIGHT,CROP_WIDTH)
model = tf.keras.models.Sequential([center_crop_layer])

uncropped_batch, _ = next(data_generator)
cropped_images = model.predict(data_generator)

#Show cropped and uncropped images side-by-side
for i in range(0,10,2):
    print(i)
    plt.figure(figsize=(15,15))
    plt.subplot(5,2,i+1)
    plt.imshow(uncropped_batch[i])
    plt.subplot(5,2,i+2)
    plt.imshow(cropped_images[i])

plt.show()