from tensorflow import keras
import numpy as np
import tensorflow as tf
import cv2

tf.enable_eager_execution()

model = keras.models.load_model("underage_detector_no_wrapper.h5")
# model.summary()

layer_dict = dict((layer.name, layer) for layer in model.layers)
# print(layer_dict)
layer_num = int(input('layer no: '))
layer_name = list(layer_dict.keys())[layer_num]


def init_img(path):
    img = cv2.imread(path)
    X =[]
    X.append(img)
    X = np.array(X)
    X = X.astype('float32')
    X /= 255

    return X


# Set up a model that returns the activation values for our target layer
# for layer_name in layer_dict.keys():
layer = model.get_layer(name=layer_name)
feature_extractor_part = keras.Model(inputs=model.inputs, outputs=layer.output)
feature_extractor_part.summary()

X = init_img('Dataset/25-30/25_0_0_20170116200929068.jpg')
feature = feature_extractor_part(X)
print(feature.shape)

filters = int(feature.shape[3])

i = 0
for i in range(0, filters, 3):
    print(i, i+3)
    try:
        display = feature.numpy()[0, :, :, i:i+3]
        keras.preprocessing.image.save_img('display/'+layer_name+'_'+str(i)+".png", display)
    except:
        pass




