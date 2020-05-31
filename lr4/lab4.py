import keras
import matplotlib.pyplot as plt
from keras import Sequential
from keras import optimizers as opt
from keras.layers import Dense
from keras.preprocessing import image
from keras.utils import to_categorical

mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images.reshape(60000, 784)
test_images = test_images.reshape(10000, 784)
train_images = train_images / 255.0
test_images = test_images / 255.0
train_labels = to_categorical(train_labels, 10)
test_labels = to_categorical(test_labels, 10)


def upload_image(path):
    img = image.load_img(path=path, grayscale=True, target_size=(28, 28, 1))
    img = image.img_to_array(img)
    return img.reshape((1, 784))


def build_model():
    model = Sequential()
    model.add(Dense(16, activation='relu', input_shape=(784,)))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    return model


def draw(H, title_name):
    plt.figure(1, figsize=(8, 5))
    plt.title(title_name)
    plt.plot(H.history['accuracy'], 'r', label='train')
    plt.plot(H.history['val_accuracy'], 'b', label='test')
    plt.legend()
    plt.show()
    plt.clf()
    plt.figure(1, figsize=(8, 5))

    plt.title("{} Training and test loss".format(title_name))
    plt.plot(H.history['loss'], 'r', label='train')
    plt.plot(H.history['val_loss'], 'b', label='test')
    plt.legend()
    plt.show()
    plt.clf()

model = build_model()
model.compile(optimizer=opt.Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
H = model.fit(train_images, train_labels, epochs=7, batch_size=100, validation_split=0.1)
test_loss, test_acc = model.evaluate(test_images, test_labels)
draw(H, "Adam")
model = build_model()