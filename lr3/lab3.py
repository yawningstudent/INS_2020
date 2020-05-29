import tensorflow
import numpy as np
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import boston_housing
import matplotlib.pyplot as plt

(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()
mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std
test_data -= mean
test_data /= std
k = 24
num_val_samples = len(train_data) // k
num_epochs = 30
all_scores = []


def build_model():
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(train_data.shape[1],)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mean_absolute_error'])
    return model


def fit_model():
    res = []
    for i in range(k):
        print('processing fold #', i)
        val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
        val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]
        partial_train_data = np.concatenate([train_data[:i * num_val_samples], train_data[(i + 1) * num_val_samples:]],axis=0)
        partial_train_targets = np.concatenate(
            [train_targets[:i * num_val_samples], train_targets[(i + 1) * num_val_samples:]], axis=0)
        model = build_model()
        H = model.fit(partial_train_data, partial_train_targets, epochs=num_epochs, batch_size=1, verbose=0, validation_data=(val_data, val_targets))
        val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
        all_scores.append(val_mae)
        loss = H.history['loss']
        mae = H.history['mean_absolute_error']
        v_loss = H.history['val_loss']
        v_mae = H.history['val_mean_absolute_error']
        x = range(1, num_epochs + 1)
        # plt.plot(x, loss)
        # plt.plot(x, v_loss)
        # plt.title('Model loss')
        # plt.ylabel('loss')
        # plt.xlabel('epochs')
        # plt.legend(['Train data', 'Test data'], loc='upper left')
        # plt.show()
        # plt.plot(x, mae)
        # plt.plot(x, v_mae)
        # plt.title('Model mean absolute error')
        # plt.ylabel('mean absolute error')
        # plt.xlabel('epochs')
        # plt.legend(['Train data', 'Test data'], loc='upper left')
        # plt.show()
    print(np.mean(all_scores))


fit_model()
