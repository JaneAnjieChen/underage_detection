# 将18岁以下的人的数据集合到underage/文件夹中
# 将大致同等数量的成人图片集合到adult/文件夹中
import os
from shutil import copy
import matplotlib.pyplot as plt
import cv2
import numpy as np

from sklearn.model_selection import KFold, cross_val_score
from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers import Dense, Flatten
from keras.utils import to_categorical
from keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.models import Model, load_model
from keras.preprocessing import image


def collect_underage_data():
    base_dir1 = 'Dataset/6-20/'
    base_dir2 = 'face_age/'

    underage_dir = 'underage/'

    if not os.path.exists(underage_dir):
        os.mkdir(underage_dir)

    for img_file in os.listdir(base_dir1):
        age = int(img_file.split('_')[0])
        if age < 18:
            # print(age)
            copy(base_dir1+img_file, underage_dir+img_file)

    for age_file in os.listdir(base_dir2):
        age = int(age_file.split('.')[0].lstrip('0'))
        if age < 18:
            # print(age)
            for img_file in os.listdir(base_dir2+'/'+age_file):
                copy(base_dir2+'/'+age_file+'/'+img_file, underage_dir+img_file)


def age_num_count():
    '''
    统计各年龄的人脸图片
    :return:graph
    '''
    age_num_dict = {}

    # base_dir1 = 'Dataset/'
    base_dir2 = 'face_age/'

    # for age_range_file in os.listdir(base_dir1):
    #     for img_file in os.listdir(base_dir1+'/'+age_range_file):
    #         age = int(img_file.split('_')[0])
    #         if age not in age_num_dict.keys():
    #             age_num_dict[age] = 1
    #         else:
    #             age_num_dict[age] += 1

    for age_file in os.listdir(base_dir2):
        num = len(os.listdir(base_dir2+'/'+age_file))
        age = int(age_file.split('.')[0].lstrip('0'))
        if age not in age_num_dict.keys():
                age_num_dict[age] = num
        else:
            age_num_dict[age] += num

    # print(age_num_dict)
    return age_num_dict

    # # draw
    # ages = []
    # nums = []
    # for age in sorted(age_num_dict):
    #     ages.append(age)
    #     nums.append(age_num_dict[age])
    # plt.bar(ages, nums)
    # plt.show()

    # adult_count = 0
    # for age in age_num_dict.keys():
    #     if age >= 18:
    #         adult_count += age_num_dict[age]
    # print(adult_count)


def collect_adult_data():
    adult_dir = 'adult/'

    if not os.path.exists(adult_dir):
        os.mkdir(adult_dir)

    base_dir = 'face_age/'
    for age_file in os.listdir(base_dir):
        age = int(age_file.split('.')[0].lstrip('0'))
        if age >= 18:
            # print(age)
            for img_file in os.listdir(base_dir+'/'+age_file):
                copy(base_dir+'/'+age_file+'/'+img_file, adult_dir+img_file)

# Settings
# data files
underage_dir = 'underage/'
adult_dir = 'adult/'
# test_p = 0.2
# underage and adults, binary classification
classes = 2
# vgg16 not-freezed layers
train_last_layers = 2

epochs = 9
batch_size = 128
# K-fold
splits = 5

# underage classification
def load_data():
    '''
    shuffle data
    :return:X, y
    '''

    # read data
    X =[]
    y = []
    for underage_imgfile in os.listdir(underage_dir):
        img = cv2.imread(underage_dir+underage_imgfile)
        # img = img.astype('float32')
        # img /= 255
        X.append(img)
        y.append(0)
    for adult_imgfile in os.listdir(adult_dir):
        img = cv2.imread(adult_dir+adult_imgfile)
        # img = img.astype('float32')
        # img /= 255
        X.append(img)
        y.append(1)

    # convert into tensors
    X = np.array(X)
    y = np.array(y)
    # print(X.shape)
    # print(y.shape)
    y = to_categorical(y, 2)
    # print(y.shape)
    X = X.astype('float32')
    X /= 255

    # shuffle data
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]

    return X, y


def build_model():
    base_model = VGG16(weights='vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',
                     include_top=False, input_shape=(200, 200, 3))

    # Freeze the layers except the last `train_last_layers` layers
    for layer in base_model.layers[:-train_last_layers]:
        layer.trainable = False

    # Check the trainable status of the individual layers

    top_model = Flatten()(base_model.output)
    top_model = Dense(classes, activation='softmax')(top_model)
    model = Model(inputs=base_model.inputs, outputs=top_model)
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    return model


def cross_validation(X, y):
    Model = KerasClassifier(build_fn=build_model, epochs=epochs, batch_size=batch_size)
    # use k-fold to get optimal hyperparameters
    kfold = KFold(n_splits=splits)
    accs = cross_val_score(Model, X, y, cv=kfold, verbose=0)
    print(accs)
    acc_mean = np.mean(accs)
    print(acc_mean)
    return acc_mean


def save_model(X, y):
    # Model = KerasClassifier(build_fn=build_model, epochs=epochs, batch_size=batch_size)
    # # save_model
    # # Model.fit(X[:8000, :, :, :], y[:8000, :])
    # # Model.model.save('underage_detection.h5')
    # # acc_mean = Model.score(X[8000:, :, :, :], y[8000:, :])
    # # print(acc_mean)
    # # return acc_mean
    # Model.fit(X, y)
    # Model.model.save('underage_detection_all_data.h5')

    model = build_model()
    model.fit(X, y, epochs=epochs, batch_size=batch_size, validation_split=0.2)
    model.save('underage_detector_02.h5')



def predition(model, image_path):
    img = cv2.imread(image_path)
    X = []
    X.append(img)
    X = np.array(X)
    X = X.astype('float32')
    X /= 255

    pred_y = model.predict(X)

    # print(pred_y)

    # 0 is underage
    if pred_y[0][0] > pred_y[0][1]:
        return 'underage'
    else:
        return 'adult'


    # src = cv2.imread(image_path)
    # cv2.namedWindow(label, cv2.WINDOW_AUTOSIZE)
    # cv2.imshow(label, src)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


if __name__ == '__main__':

    # collect_underage_data()
    # age_num_count()
    # collect_adult_data()

    # X, y = load_data()
    # print(X.shape, y.shape)

    # model = build_model()

    # cross_validation(X, y)
    # save_model(X, y)
    # predition('underage_detection_all_data.h5', 'Dataset/25-30/25_0_0_20170104004153743.jpg')

    error_count = 0
    count = 0
    model = load_model('underage_detector_02.h5')
    dir = 'Dataset/6-20/'
    for i in os.listdir(dir):
        count += 1
        label = predition(model, dir+i)
        # print(label)
        if label != 'underage':
            # print(i)
            error_count += 1
    print(error_count)
    print(error_count/count)



    # BAD PREPROCESSING!
    # 'underage_detection_all_data.h5':
    # 'Dataset/25-30/':
    # 1729
    # 0.6935419173686321

    # 42-48/
    # 867
    # 0.6025017373175816

    # 60-98/
    # 1347
    # 0.5090702947845805

    # adults/
    # 3613
    # 0.6176068376068377

    # underage/
    # 100
    # 0.017271157167530225






