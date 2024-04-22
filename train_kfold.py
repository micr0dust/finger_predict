%cd /content/drive/MyDrive/mid/
from skmultilearn.model_selection import IterativeStratification
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from keras.preprocessing.image import img_to_array, load_img
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D,Dropout, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
import os
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from tensorflow.keras.utils import plot_model



folders = os.listdir('processed_data_all')

data = []
labels = []

for folder in tqdm(folders,desc='load image'):
    if not os.path.isdir(f'processed_data_all/{folder}'):
        continue
    for file in os.listdir(f'processed_data_all/{folder}'):
        image = cv2.imread(f'processed_data_all/{folder}/{file}')
        processed_image = image
        data.append(processed_image)
        labels.append(folder)


# Assume image_list is your list of images and label_list is your list of labels
# Convert the list of images to a NumPy array
X = np.array(data)/255.0

# Convert the list of labels to a NumPy array
labels = [list(map(int, label)) for label in labels]
y = np.array(labels)


# Assume X is your feature matrix and y is your multi-label target matrix
k_fold = IterativeStratification(n_splits=5, order=1)

def partial_correct_accuracy(y_true, y_pred):
    y_true = tf.cast(y_true, 'bool')
    y_pred = tf.cast(tf.round(y_pred), 'bool')
    p = tf.reduce_sum(tf.cast(tf.logical_and(y_true, y_pred), 'float32'), axis=-1)
    q = tf.reduce_sum(tf.cast(tf.logical_or(y_true, y_pred), 'float32'), axis=-1)
    return tf.reduce_mean(tf.where(tf.equal(q, 0), 0.0, p / q))

def history_plot(train_history, i):
    # 繪製訓練和驗證的準確度、損失值和partial_correct_accuracy
    plt.figure(figsize=(18, 4))

    plt.subplot(1, 3, 1)
    plt.plot(train_history.history['accuracy'], label='Train Accuracy')
    plt.plot(train_history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'Fold {i+1} Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(train_history.history['loss'], label='Train Loss')
    plt.plot(train_history.history['val_loss'], label='Validation Loss')
    plt.title(f'Fold {i+1} Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(train_history.history['partial_correct_accuracy'], label='Train Partial Correct Accuracy')
    plt.plot(train_history.history['val_partial_correct_accuracy'], label='Validation Partial Correct Accuracy')
    plt.title(f'Fold {i+1} Partial Correct Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Partial Correct Accuracy')
    plt.legend()

    plt.savefig(f'/content/fold_{i+1}_history.png')
    plt.close()

# 創建一個ImageDataGenerator對象來進行資料擴增
datagen = ImageDataGenerator(
    # width_shift_range=0.1,
    # height_shift_range=0.1,
    # rotation_range=30,
    # fill_mode='nearest',
    horizontal_flip=True,
)



# 在迴圈外部初始化一個列表來保存每次迭代的評估分數
scores = []
history_list = []
for i, (train_index, test_index) in enumerate(k_fold.split(X, y)):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 48, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.3))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.6))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(5, activation='sigmoid'))

    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy', partial_correct_accuracy])



    # 使用ImageDataGenerator對象來擴增訓練數據
    train_gen = datagen.flow(X_train, y_train, batch_size=32)

    # 從訓練數據生成器中獲取一批數據
    data_batch, labels_batch = next(train_gen)

    # 繪製第一個圖片
    plt.figure(figsize=(6, 6))
    plt.imshow((data_batch[0]*255).astype('int'))
    plt.title('Augmented Image')
    # 儲存圖片
    plt.savefig('/content/augmented_image.png')
    plt.close()

    model.summary()
    plot_model(model, to_file='/content/model.png', show_shapes=True)

    train_history = model.fit(
        X_train,
        y_train,
        epochs=80,
        validation_data=(X_test, y_test),
        verbose=0
    )

     # 在每次迭代結束後，評估模型並保存評估分數和訓練歷程
    score = model.evaluate(X_test, y_test)
    scores.append(score)
    history_list.append(train_history)

    history_plot(train_history, i)

# 計算平均分數
average_score = np.mean(scores, axis=0)
print(f'Average loss: {average_score[0]}')
print(f'Average accuracy: {average_score[1]}')
print(f'Average partial correct accuracy: {average_score[2]}')
