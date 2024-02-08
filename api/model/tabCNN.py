from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Reshape, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras import optimizers
from keras import backend as K

# CNN architecture
class model:
    def __init__(self,
                 num_classes=21,
                 num_strings=6,
                 input_shape=(192, 9, 1) ,
                 momentum=0.09,
                 weights='weights.h5'
                ):
        
        self.num_classes = num_classes
        self.num_strings = num_strings
        self.input_shape = input_shape
        self.momentum = momentum
        self.weights = weights

    def softmax_by_string(self, t):
        string_sm = []
        for i in range(self.num_strings):
            string_sm.append(K.expand_dims(K.softmax(t[:,i,:]), axis=1))
        return K.concatenate(string_sm, axis=1)

    def catcross_by_string(self, target, output):
        loss = 0
        for i in range(self.num_strings):
            loss += K.categorical_crossentropy(target[:,i,:], output[:,i,:])
        return loss

    def avg_acc(self, y_true, y_pred):
        return K.mean(K.equal(K.argmax(y_true, axis=-1), K.argmax(y_pred, axis=-1)))

    def build_model(self):
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3),
                            activation='relu',
                            input_shape=self.input_shape))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.num_classes * self.num_strings)) # no activation
        model.add(Reshape((self.num_strings, self.num_classes)))
        model.add(Activation(self.softmax_by_string))

        model.compile(loss=self.catcross_by_string,
                    optimizer=optimizers.SGD(momentum=self.momentum),
                    metrics=[self.avg_acc])
        
        model.load_weights(self.weights)

        return model
        