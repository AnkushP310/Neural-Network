# Paul, Ankush
# 1001_830_258
# 2021_10_10
# Assignment_02_01

import numpy as np


class LinearAssociator(object):
    def __init__(self, input_dimensions=2, number_of_nodes=4, transfer_function="Hard_limit"):
        """
        Initialize linear associator model
        :param input_dimensions: The number of dimensions of the input data
        :param number_of_nodes: number of neurons in the model
        :param transfer_function: Transfer function for each neuron. Possible values are:
        "Hard_limit", "Linear".
        """
        self.input_dimensions = input_dimensions
        self.number_of_nodes = number_of_nodes
        self.transfer_function = transfer_function
        self.initialize_weights()

    def initialize_weights(self, seed=None):
        """
        Initialize the weights, initalize using random numbers.
        If seed is given, then this function should
        use the seed to initialize the weights to random numbers.
        :param seed: Random number generator seed.
        :return: None
        """
        if seed != None:
            np.random.seed(seed)
        
        self.weights = np.random.randn(self.number_of_nodes, self.input_dimensions)


    def set_weights(self, W):
        """
         This function sets the weight matrix.
         :param W: weight matrix
         :return: None if the input matrix, w, has the correct shape.
         If the weight matrix does not have the correct shape, this function
         should not change the weight matrix and it should return -1.
        """
        if self.weights.shape == W.shape:
            self.weights = W
            return None
        else:
            return -1

    def get_weights(self):
        """
         This function should return the weight matrix(Bias is included in the weight matrix).
         :return: Weight matrix
        """
        return self.weights

    def predict(self, X):
        """
        Make a prediction on an array of inputs
        :param X: Array of input [input_dimensions,n_samples].
        :return: Array of model outputs [number_of_nodes ,n_samples]. This array is a numerical array.
        """
        predictions =  self.weights.dot(X)

        if self.transfer_function == 'Hard_limit' :
            for i in range(predictions.shape[0]):
                for j in range(predictions.shape[1]):
                    predictions[i][j] = 1 if predictions[i][j] >= 0.0 else 0
        return predictions

    def fit_pseudo_inverse(self, X, y):
        """
        Given a batch of data, and the targets,
        this function adjusts the weights using pseudo-inverse rule.
        :param X: Array of input [input_dimensions,n_samples]
        :param y: Array of desired (target) outputs [number_of_nodes,n_samples]
        """
        X_inv = np.linalg.pinv(X)
        self.weights = y.dot(X_inv)

    def train(self, X, y, batch_size=5, num_epochs=10, alpha=0.1, gamma=0.9, learning="Delta"):
        """
        Given a batch of data, and the necessary hyperparameters,
        this function adjusts the weights using the learning rule.
        Training should be repeated num_epochs time.
        :param X: Array of input [input_dimensions,n_samples]
        :param y: Array of desired (target) outputs [number_of_nodes,n_samples].
        :param num_epochs: Number of times training should be repeated over all input data
        :param batch_size: number of samples in a batch
        :param num_epochs: Number of times training should be repeated over all input data
        :param alpha: Learning rate
        :param gamma: Controls the decay
        :param learning: Learning rule. Possible methods are: "Filtered", "Delta", "Unsupervised_hebb"
        :return: None
        """
        #print("X: ", X, '\n')
        #print("y: ", y, '\n\n\n')
        for e in range(num_epochs):
            mini_batches = self.generate_batch(X, y, batch_size)
            for mini_batch in mini_batches:
                ###print("Weights: ", self.weights, '\n')
                X_batch, y_batch = mini_batch
                #print("X_batch: ", X_batch, '\n', "X batch Shape: ", X_batch.shape, '\n')
                #print("y_batch: ", y_batch, '\n', "y batch Shape: ", y_batch.shape, '\n')
                if learning == "Filtered":
                    # print("Filtered", '\n')
                    self.filtered_learning(X_batch, y_batch, alpha, gamma)
                elif learning == "Unsupervised_hebb":
                    # print("Unsupervised_hebb", '\n')
                    self.unsupervised_learning(X_batch, y_batch, alpha)
                else:
                    #print("delta", '\n')
                    self.delta_learning(X_batch, y_batch, alpha)                


    def calculate_mean_squared_error(self, X, y):
        """
        Given a batch of data, and the targets,
        this function calculates the mean squared error (MSE).
        :param X: Array of input [input_dimensions,n_samples]
        :param y: Array of desired (target) outputs [number_of_nodes,n_samples]
        :return mean_squared_error
        """
        prediction = self.predict(X)
        mse_loss = np.square(np.subtract(y, prediction)).mean()
        return mse_loss


    ###Source: https://www.geeksforgeeks.org/ml-mini-batch-gradient-descent-with-python/
    def generate_batch(self, X, y, batch_size=5):
        #print("X: ", X, '\n', "X Shape: ", X.shape, '\n')
        #print("y: ", y, '\n', "y Shape: ", y.shape, '\n')
        mini_batches = []
        #data = np.vstack((X, y))
        #print("data: ", data,'\n', "data Shape: ", data.shape, '\n')
        ##np.random.shuffle(data)
        #data = data.T
        n_minibatches = X.shape[1] // batch_size
  
        for i in range(n_minibatches):
            #mini_batch = data[i * batch_size:(i + 1)*batch_size, :]
            X_mini = X[: ,i * batch_size:(i + 1)*batch_size]
            #print("X_mini", X_mini, '\n')
            y_mini = y[: ,i * batch_size:(i + 1)*batch_size]
            #print("y_mini", y_mini, '\n')
            mini_batches.append((X_mini, y_mini))
        if X.shape[1] % batch_size != 0:
            #mini_batch = data[i * batch_size:data.shape[0]]
            X_mini = X[i * batch_size:X.shape[1], :]
            y_mini = y[i * batch_size:X.shape[1], :]
            mini_batches.append((X_mini, y_mini))
        #print("mini_batches", mini_batches, '\n')
        return mini_batches

    def filtered_learning(self, X, y, alpha=0.1, gamma=0.9):
        self.weights = (1-gamma)*self.weights + alpha*(y.dot(X.T.copy()))

    def delta_learning(self, X, y, alpha=0.1):
        predictions = self.predict(X)
        #print("predictions: ", predictions, '\n')
        error = y - predictions
        #print("error: ", error, '\n')
        self.weights = self.weights + (error.dot(X.T.copy()))*alpha
        #print("weights: ", self.weights, '\n')

    def unsupervised_learning(self, X, y, alpha=0.1):
        predictions = self.predict(X)
        self.weights = self.weights + alpha*(predictions.dot(X.T.copy()))