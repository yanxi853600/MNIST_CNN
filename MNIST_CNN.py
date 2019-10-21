# 

from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

# 定義梯度下降批量
batch_size = 128
# 定義分類數量
num_classes = 10
# 定義訓練週期
epochs = 5 #為了不等那麼久，將epoch設為5

# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# channels_first: 色彩通道(R/G/B)資料(深度)放在第2維度，第3、4維度放置寬與高
# channels_last: 色彩通道(R/G/B)資料(深度)放在第4維度，第2、3維度放置寬與高
if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

# 轉換色彩 0~255 資料為 0~1
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# one-hot encoding
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
# 建立卷積層
##filter=32
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))

# 建立卷積層
##filter=64,即 output space 的深度, Kernal Size: 3x3, activation function 採用 relu
model.add(Conv2D(64, (3, 3), activation='relu')) 

#池化層(池化大小=2x2，取最大值)
model.add(MaxPooling2D(pool_size=(2, 2))) 
model.add(Dropout(0.25))## Dropout層隨機斷開輸入神經元，用於防止過度擬合，斷開比例:0.25
model.add(Flatten()) ##把多維的輸入一維化，常用在從卷積層到全連接層的過渡。

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
#全連接層(128 output)
model.add(Dense(num_classes, activation='softmax')) 
## 使用 softmax activation function，將結果分類

# 編譯: 選擇損失函數、優化方法及成效衡量方式
model.compile(loss=keras.losses.binary_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

# 進行訓練, 訓練過程會存在 train_history 變數中
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))

# 顯示損失函數、訓練成果(分數)
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])



print("\n調整參數_說明:")
print("1. 增加隱藏層數可以讓誤差變小:\n   NN=2-2-1時，(epoch=1000)loss=1.573\n   NN=2-5-1時，(epoch=1000)loss=1.088\n ")
print("2. Relu (0.9898)vs. Sigmoid(0.9452)\n")
print("3. 損失函數:\n   categorical_crossentropy(0.9904) \n   binary_crossentropy(0.9974)-->最高\n   優化方法:\n   Adadelta(0.9904)\n   SGD (0.9602)\n")
print("4. dropout:\n   全連接改成 0.25 (0.9904)\n   dropout:0.5 (0.9975)")