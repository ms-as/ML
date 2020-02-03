import tensorflow as tf
import numpy as np 
from random import random
from sklearn.model_selection import train_test_split

def generate_dataset(number_of_samples, test_size):

    x = np.array([[random()/2 for _ in range(2)] for _ in range(number_of_samples)])
    y = np.array([[i[0] + i[1]] for i in x])
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=test_size)

    return x_train, x_test, y_train, y_test

if __name__ == "__main__":
    x_train, x_test, y_train, y_test = generate_dataset(5000,0.3)
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(5, input_dim=2, activation="sigmoid"),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])

    optimiser = tf.keras.optimizers.SGD(learning_rate=0.1)
    model.compile(optimizer=optimiser, loss="MSE")

    #train
    model.fit(x_train, y_train, epochs = 1000)
    
    #evaluate

    print("\n Model evaluation:")
    model.evaluate(x_test, y_test, verbose=1)

    #predictions

    data = np.array([[0.7, 0.2], [0.3, 0.6]])
    predictions = model.predict(data)

    print("\nPredictions:")
    for d, p in zip(data, predictions):
        print("{} + {} = {}".format(d[0], d[1], p[0]))