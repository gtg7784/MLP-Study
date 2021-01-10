import tensorflow as tf
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Activation, Dense
from keras import optimizers

print(tf.__version__)

dataset = load_breast_cancer()

X_data = dataset.data
y_data = dataset.target

X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.3, random_state=7) # 학습 데이터(0.7)와 검증 데이터(0.3)로  전체 데이터 셋을 나눈다

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

model = Sequential()
model.add(Dense(10, input_shape=10))
model.add(Activation('sigmoid'))
model.add(Dense(10))
model.add(Activation('sigmoid'))
model.add(Dense(10))
model.add(Activation('sigmoid'))
model.add(Dense(1))
model.add(Activation('sigmoid'))

sgd = optimizers.SGD(lr=0.01)
model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['acc'])

model.summary()

