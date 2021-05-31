import tensorflow as tf
import numpy as np
class Model():
    
    def __init__(self, training_data):
        self.X = np.array([i[0] for i in training_data])
        self.y = np.array([i[1] for i in training_data])
        
    def create(self):
        self.model = self.nnm(len(self.X[0]))
        
    def train(self):
        loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        self.model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])
        history = self.model.fit(self.X, self.y, batch_size=1, epochs=5)
        return history
    
    def get_model(self):
        return self.model
    
    def nnm(self,input_size):
        model = tf.keras.Sequential()
        model.add(tf.keras.Input(shape=(input_size,)))
        #model.add(tf.keras.layers.Flatten())
        print(model.output_shape)
        model.add(tf.keras.layers.Dense(128, activation='relu'))
        model.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.Dense(9, activation='softmax'))
        print(model.output_shape)
        model.summary()
        return model