from tensorflow.keras import layers, activations
from tensorflow.keras.models import Sequential


def make_lenet5():
    model = Sequential()

    # [DONE] Todo: implement LeNet-5 model
    
    # 1st Layer
    model.add(layers.Conv2D(filters=6, kernel_size=(5, 5), strides=(1, 1), activation=activations.tanh, input_shape=(32, 32, 1)))

    # 2nd Layer
    model.add(layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2)))

    # 3rd Layer
    model.add(layers.Conv2D(filters=16, kernel_size=(5, 5), strides=(1, 1), activation=activations.tanh))

    # 4th Layer
    model.add(layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2)))

    # 5th Layer
    model.add(layers.Flatten())
    #model.add(layers.Conv2D(filters=120, kernel_size=(5, 5), strides=(1, 1), activation=activations.tanh, input_shape=(32, 32, 1)))

    # 6th Layer
    model.add(layers.Dense(units=84, activation=activations.tanh))

    # 7th Layer
    model.add(layers.Dense(units=10, activation=activations.softmax))


    return model
