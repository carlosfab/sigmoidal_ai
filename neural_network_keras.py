# importar os pacotes necessários
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD
import numpy as np
import matplotlib.pyplot as plt

def main():
    # importar o MNIST
    print("[INFO] importando MNIST...")
    dataset = fetch_mldata("MNIST Original")

    # normalizar todos pixels, de forma que os valores estejam
    # no intervalor [0, 1.0]
    data = dataset.data.astype("float") / 255.0
    labels = dataset.target

    # dividir o dataset entre train (75%) e test (25%)
    (trainX, testX, trainY, testY) = train_test_split(data, dataset.target)

    # converter labels de inteiros para vetores
    lb = LabelBinarizer()
    trainY = lb.fit_transform(trainY)
    testY = lb.transform(testY)

    # definir a arquitetura da Rede Neural usando Keras
    # 784 (input) => 128 (hidden) => 64 (hidden) => 10 (output)
    model = Sequential()
    model.add(Dense(128, input_shape=(784,), activation="sigmoid"))
    model.add(Dense(64, activation="sigmoid"))
    model.add(Dense(10, activation="softmax"))

    # treinar o modelo usando SGD (Stochastic Gradient Descent)
    print("[INFO] treinando a rede neural...")
    model.compile(optimizer=SGD(0.01), loss="categorical_crossentropy",
                 metrics=["accuracy"])
    H = model.fit(trainX, trainY, batch_size=128, epochs=100, verbose=2,
             validation_data=(testX, testY))

    # avaliar a Rede Neural
    print("[INFO] avaliando a rede neural...")
    predictions = model.predict(testX, batch_size=128)
    print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1)))

    # plotar loss e accuracy para os datasets 'train' e 'test'
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0,100), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0,100), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0,100), H.history["acc"], label="train_acc")
    plt.plot(np.arange(0,100), H.history["val_acc"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.show()

    return None

if __name__ == '__main__':
    main()