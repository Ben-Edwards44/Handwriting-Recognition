#Author: Ben-Edwards44


import neural_network
import pickle
import numpy
import imageio


def get_data_training():
    print("Fetching training data...")

    with open("mnist_train.csv", "r") as file:
        data = file.read()

    print("Fetched training data")
    return data.split("\n")


def get_test_data():
    print("Fetching test data...")

    with open("mnist_test.csv", "r") as file:
        data = file.read()

    print("Fetched test data")
    return data.split("\n")


def train(num):
    data = get_data_training()
    print("Training model...")

    for epoch in range(num):
        for i in data:
            line = i.split(",")
            inputs = [int(x) / 255 * 0.99 + 0.01 for x in line[1:len(line)]]

            target_val = int(i[0])
            targets = [0.01 for _ in range(10)]
            targets[target_val] = 0.99

            model.train_network(inputs, targets)

        print(f"Epoch {epoch} complete")

    print("Model trained")


def score():
    data = get_test_data()
    print("Assessing score...")

    correct = 0
    for i in data:
        line = i.split(",")

        ans = int(line[0])
        inputs = numpy.array([int(x) for x in line[1:len(line)]])
        inputs = (inputs / 255.0 * 0.99) + 0.01

        predicted = list(model.predict(list(inputs)))
        predicted_value = predicted.index(max(predicted))

        if predicted_value == ans:
            correct += 1

    print(f"Accuracy: {correct / len(data) * 100}%")


def predict_from_image(fn):
    img_data = list(get_pixel_vals(fn))
    predicted = list(model.predict(img_data))

    certainty = max(predicted)
    predicted_value = predicted.index(certainty)

    print(f"Predicted {predicted_value} with a certainty of {certainty * 100 :.3f}%")


def get_pixel_vals(fn):
    img_array = imageio.imread(fn, as_gray=True)
    img_data  = 255.0 - img_array.reshape(784)
    img_data = (img_data / 255.0 * 0.99) + 0.01

    return img_data


def save_model(file_name):
    with open(file_name, "wb") as file:
        pickle.dump(model, file)

    print("Model saved")


def load_model(file_name):
    global model

    with open(file_name, "rb") as file:
        model = pickle.load(file)


convert_2d = lambda array: [[i] for i in array]


"""
--------------------------------
To load a pre-trained model:
load_model("FILENAME.pickle")
--------------------------------
To train a new model:
model = neural_network.Neural_network(LEARNING RATE, NUMBER OF HIDDEN NODES)
train(NUMBER OF EPOCHS)
save_model("FILENAME.pickle")
--------------------------------
To test a model's accuracy (the neural network must be stored with the identifier "model"):
score()
--------------------------------
To query a model from an image (the neural network must be stored with the identifier "model"):
predict_from_image(FILENAME)
--------------------------------
"""
