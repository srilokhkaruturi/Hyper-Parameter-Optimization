#####################################################################################################################
#   Assignment 2: Neural Network Analysis
#   This is a starter code in Python 3.6 for a neural network.
#   You need to have numpy and pandas installed before running this code.
#   You need to complete all TODO marked sections
#   You are free to modify this code in any way you want, but need to mention it
#       in the README file.
#
#####################################################################################################################
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import warnings
import numpy as np
import pandas as pd
import io
import matplotlib.pyplot as plt
import requests
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor


class NeuralNet:
    def __init__(self, dataFile, header=True):
        # download the file
        dataFile = requests.get(
            dataFile)
        self.raw_input = pd.read_csv(io.StringIO(dataFile.text), sep=";")

    def preprocess(self):
        processed_data = self.raw_input

        # remove null
        processed_data.dropna(inplace=True)

        # drop unneeded columns
        processed_data = processed_data.drop(
            ["school", "Mjob", "Fjob", "reason", "paid", "G1", "G2"], axis=1)

        # categorical to numerical (generally 1 is good)
        # sex
        processed_data['sex'] = processed_data['sex'].map({'F': 0, 'M': 1})

        # address
        processed_data["address"] = processed_data["address"].map({
                                                                  'U': 1, 'R': 0})
        # famsize
        processed_data["famsize"] = processed_data["famsize"].map(
            {'LE3': 0, 'GT3': 1})

        # Pstatus
        processed_data["Pstatus"] = processed_data["Pstatus"].map({
                                                                  'T': 1, 'A': 0})

        # guardian
        processed_data["guardian"] = processed_data["guardian"].map(
            {'mother': 1, 'father': 1, 'other': 0})

        # schoolsup
        processed_data["schoolsup"] = processed_data["schoolsup"].map(
            {'yes': 1, 'no': 0})

        # famsup
        processed_data["famsup"] = processed_data["famsup"].map(
            {'yes': 1, 'no': 0})

        # extra-curricular activities
        processed_data["activities"] = processed_data["activities"].map(
            {'yes': 1, 'no': 0})

        # nursery
        processed_data["nursery"] = processed_data["nursery"].map(
            {'yes': 1, 'no': 0})

        # higher
        processed_data["higher"] = processed_data["higher"].map(
            {'yes': 1, 'no': 0})

        # internet
        processed_data["internet"] = processed_data["internet"].map(
            {'yes': 1, 'no': 0})

        # romantic relationship
        processed_data["romantic"] = processed_data["romantic"].map(
            {'yes': 0, 'no': 1})

        # set processed data
        scaler = StandardScaler()
        self.processed_data = pd.DataFrame(
            scaler.fit_transform(processed_data))
        return 0

    def train_evaluate(self):
        ncols = len(self.processed_data.columns)
        nrows = len(self.processed_data.index)
        X = self.processed_data.iloc[:, 0:(ncols - 1)]
        y = self.processed_data.iloc[:, (ncols-1)]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y)

        # Below are the hyperparameters that you need to use for model
        #   evaluation
        activations = ['logistic', 'tanh', 'relu']
        learning_rate = [0.01, 0.1]
        max_iterations = [100, 200]  # also known as epochs
        num_hidden_layers = [2, 3]

        # go through all possibilites of neural networks
        numIter = 0
        modelList = []
        bestModelNumber = 0
        bestScore = -100
        for activation in activations:
            for rate in learning_rate:
                for epoch in max_iterations:
                    for layers in num_hidden_layers:
                        # increase number of iterations ( model )
                        numIter += 1

                        # instantiate model
                        nn = MLPRegressor(
                            activation=activation, learning_rate_init=rate, learning_rate="constant",  max_iter=epoch)

                        # define num layers
                        nn.n_layers = layers

                        # ignore convergence warnings
                        warnings.filterwarnings(
                            "ignore", category=ConvergenceWarning)

                        # train
                        mse_list = []
                        for i in range(epoch):
                            nn.partial_fit(X_train, y_train)
                            pred = nn.predict(X_test)
                            mse_list.append(mean_squared_error(pred, y_test))

                        # score (mse - test)
                        pred = nn.predict(X_train)
                        mse_train = mean_squared_error(pred, y_train)

                        # score (mse - test)
                        pred = nn.predict(X_test)
                        mse_test = mean_squared_error(pred, y_test)

                        # # print results
                        # print("{:<25}{}".format("Model Number:", numIter))
                        # print("{:<25}{}".format("Activation:", activation))
                        # print("{:<25}{}".format("Learning rate:", rate))
                        # print("{:<25}{}".format("Max iterations:", epoch))
                        # print("{:<25}{}".format(
                        #     "Number of hidden layers:", layers))
                        # print("{:<25}{}".format("MSE Train:", mse_train))
                        # print("{:<25}{}".format("MSE Test:", mse_test))
                        # print("-" * 30)

                        # add to list of model results
                        modelList.append({
                            'model_number': numIter,
                            "Activation": activation,
                            "Learning rate": rate,
                            "Max iterations": epoch,
                            "Number of hidden layers": layers,
                            "MSE Train": mse_train,
                            "MSE Test": mse_test,
                            'mse_list': mse_list
                        })

                        if abs(mse_test) < abs(bestScore):
                            bestScore = mse_test
                            bestModelNumber = numIter

                        # plot the graph of score vs. epoch
                        # plt.plot(mse_list)
                        # plt.xlabel('Epochs')
                        # plt.ylabel('Score (RMSE)')
                        # plt.title('Model %d Performance vs Epoch ' % numIter)
                        # 0.9  # adjust y coordinate
                        # plt.text(
                        #     epoch * 0.55, max(mse_list)*.80, f"Activation: {activation}\nLearning Rate: {rate}\nMax Iterations: {epoch}\nNumber of Hidden Layers: {layers}\nMSE: {mse:.2f}")
                        # plt.show()

        bestModelInfo = modelList[bestModelNumber-1]
        print("** Best Model Number: %d **" % bestModelNumber)
        # print results
        print("{:<25}{}".format("Activation:",
                                bestModelInfo['Activation']))
        print("{:<25}{}".format("Learning rate:",
                                bestModelInfo['Learning rate']))
        print("{:<25}{}".format("Max iterations:",
                                bestModelInfo['Max iterations']))
        print("{:<25}{}".format("Number of hidden layers:",
                                bestModelInfo['Number of hidden layers']))
        print("{:<25}{}".format("MSE Train:", bestModelInfo['MSE Train']))
        print("{:<25}{}".format("MSE Test:", bestModelInfo['MSE Test']))
        print("-" * 30)

        plt.plot(mse_list)
        plt.xlabel('Epochs')
        plt.ylabel('Error (MSE)')
        plt.title('Best Model (%d) Performance vs Epoch ' % bestModelNumber)
        0.9
        plt.text(
            epoch * 0.55, max(mse_list)*.80, f"Activation: {bestModelInfo['Activation']}\nLearning Rate: {bestModelInfo['Learning rate']}\nMax Iterations: {bestModelInfo['Max iterations']}\nNumber of Hidden Layers: {bestModelInfo['Number of hidden layers']}\nTest RMSE: {bestModelInfo['MSE Test']:.2f}")
        plt.show()

        # Print table
        model_data = pd.DataFrame(modelList)
        model_data = model_data.iloc[:, :-1]
        print(model_data.to_markdown(index=False))

        # plot logistic
        for model in modelList:
            if model["Activation"] == "logistic":
                plt.plot(model["mse_list"])

        plt.xlabel('Epochs')
        plt.ylabel('Error (MSE)')
        plt.title('Logistic Models Error vs Epoch ')
        0.9  # adjust y coordinate
        plt.show()

        # plot tanh
        for model in modelList:
            if model["Activation"] == "tanh":
                plt.plot(model["mse_list"])
        plt.xlabel('Epochs')
        plt.ylabel('Error (MSE)')
        plt.title('tanh Models Error vs Epoch ')
        plt.show()

        # plot relu
        for model in modelList:
            if model["Activation"] == "relu":
                plt.plot(model["mse_list"])
        plt.xlabel('Epochs')
        plt.ylabel('Error (MSE)')
        plt.title('relu Models Error vs Epoch ')
        plt.show()


if __name__ == "__main__":
    neural_network = NeuralNet(
        "https://gist.githubusercontent.com/srilokhkaruturi/446faf7c8e042a1130d0dae44530e6c9/raw/1f36e8bafdf5cf8aa2a334c72c0f773b8e721a84/student-math.csv")
    neural_network.preprocess()
    neural_network.train_evaluate()
