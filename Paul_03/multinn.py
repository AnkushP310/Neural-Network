# Paul, Ankush
# 1001_830_258
# 2021_10_31
# Assignment_03_01

# %tensorflow_version 2.x
import tensorflow as tf
import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)

class MultiNN(object):
    def __init__(self, input_dimension):
        """
        Initialize multi-layer neural network
        :param input_dimension: The number of dimensions for each input data sample
        """
        self.input_dimension = input_dimension
        self.num_endlayer_nodes = input_dimension
        self.num_layers = 0
        self.weights = []
        self.biases = []
        self.activation = []

    def add_layer(self, num_nodes, transfer_function="Linear"):
        """
         This function adds a dense layer to the neural network
         :param num_nodes: number of nodes in the layer
         :param transfer_function: Activation function for the layer. Possible values are:
        "Linear", "Relu","Sigmoid".
         :return: None
         """
        self.num_layers += 1
        self.activation.append(transfer_function)
        weight = tf.Variable(tf.random.normal(shape=(self.num_endlayer_nodes, num_nodes)), trainable=True)
        bias = tf.Variable(tf.random.normal(shape=(1, num_nodes)), trainable=True)
        self.weights.append(weight)
        self.biases.append(bias)
        self.num_endlayer_nodes = num_nodes


    def get_weights_without_biases(self, layer_number):
        """
        This function should return the weight matrix (without biases) for layer layer_number.
        layer numbers start from zero.
         :param layer_number: Layer number starting from layer 0. This means that the first layer with
          activation function is layer zero
         :return: Weight matrix for the given layer (not including the biases). Note that the shape of the weight matrix should be
          [input_dimensions][number of nodes]
         """
        return self.weights[layer_number].numpy() if layer_number < self.num_layers else -1

    def get_biases(self, layer_number):
        """
        This function should return the biases for layer layer_number.
        layer numbers start from zero.
        This means that the first layer with activation function is layer zero
         :param layer_number: Layer number starting from layer 0
         :return: Weight matrix for the given layer (not including the biases).
         Note that the biases shape should be [1][number_of_nodes]
         """
        return self.biases[layer_number].numpy() if layer_number < self.num_layers else -1

    def set_weights_without_biases(self, weights, layer_number):
        """
        This function sets the weight matrix for layer layer_number.
        layer numbers start from zero.
        This means that the first layer with activation function is layer zero
         :param weights: weight matrix (without biases). Note that the shape of the weight matrix should be
          [input_dimensions][number of nodes]
         :param layer_number: Layer number starting from layer 0
         :return: none
         """
        ## if self.weights[layer_number].numpy().shape == weights.shape :
        weight = tf.Variable(weights, trainable=True)
        self.weights[layer_number] = weight


    def set_biases(self, biases, layer_number):
        """
        This function sets the biases for layer layer_number.
        layer numbers start from zero.
        This means that the first layer with activation function is layer zero
        :param biases: biases. Note that the biases shape should be [1][number_of_nodes]
        :param layer_number: Layer number starting from layer 0
        :return: none
        """
        ## if self.biases[layer_number].numpy().shape == biases.shape :
        bias = tf.Variable(biases, trainable=True)
        self.biases[layer_number] = bias


    def calculate_loss(self, y, y_hat):
        """
        This function calculates the sparse softmax cross entropy loss.
        :param y: Array of desired (target) outputs [n_samples]. This array includes the indexes of
         the desired (true) class.
        :param y_hat: Array of actual output values [n_samples][number_of_classes].
        :return: loss
        """
        softmax_m = tf.nn.softmax(y_hat, axis = 1)
        yht = np.zeros((y.size, y.max()+1))
        yht[np.arange(y.size),y] = 1
        yhtt = tf.Variable(yht)
        return tf.compat.v1.losses.softmax_cross_entropy(yhtt, y_hat)
        # print(y.shape, y_hat.shape)
        # loss = -tf.reduce_sum(y.dot( tf.math.log(softmax_m).numpy()), 1)
        # return loss
        # loss = 0
        # for i in range(len(y)):
        #     loss += (-tf.math.log(softmax_m[i][y[i]]))

        # return tf.convert_to_tensor(loss)


    def predict(self, X):
        """
        Given array of inputs, this function calculates the output of the multi-layer network.
        :param X: Array of input [n_samples,input_dimensions].
        :return: Array of outputs [n_samples,number_of_classes ]. This array is a numerical array.
        """
        y_hat = tf.convert_to_tensor(X, dtype=tf.float32)
        ## print("Weights: ", self.weights)
        ## print("Biases: ", self.biases)
        for i in range(len(self.weights)):
            print("\n\ny shape: ",y_hat.shape, "X shape: ", X.shape, "W shape: ", self.weights[i].shape)
            y_hat = tf.tensordot(y_hat, self.weights[i], axes=1)
            print("\ny1 shape: ",y_hat.shape)
            print("bias shape: ",self.biases[i].shape,"\n\n")
            y_hat = tf.Variable(y_hat + self.biases[i])
            print("\ny2 shape: ",y_hat.shape)
            if self.activation[i] == "Relu":
                y_hat = tf.nn.relu(y_hat)
            elif self.activation[i] == "Sigmoid":
                y_hat = tf.math.sigmoid(y_hat)
        print(type(y_hat))
        return y_hat


    def train(self, X_train, y_train, batch_size, num_epochs, alpha=0.8):
        """
         Given a batch of data, and the necessary hyperparameters,
         this function trains the neural network by adjusting the weights and biases of all the layers.
         :param X: Array of input [n_samples,input_dimensions]
         :param y: Array of desired (target) outputs [n_samples]. This array includes the indexes of
         the desired (true) class.
         :param batch_size: number of samples in a batch
         :param num_epochs: Number of times training should be repeated over all input data
         :param alpha: Learning rate
         :return: None
         """
        mini_batches = self.generate_batch(X_train, y_train, batch_size)
        for e in range(num_epochs):
            for x,y in mini_batches:
                with tf.GradientTape(persistent=True) as tape:
                    tape.watch(self.weights)
                    tape.watch(self.biases)
                    y_hat = self.predict(x)
                    loss = (self.calculate_loss(y, y_hat))
                    ##for i in range(len(self.weights)):
                    dloss_dw, dloss_db = tape.gradient(loss, [self.weights, self.biases])
                print("loss: ", loss, "dloss_dw: ", dloss_dw, "dloss_db: ", dloss_db)
                self.weights.assign_sub(alpha * dloss_dw)
                self.biases.assign_sub(alpha * dloss_db)

    def calculate_percent_error(self, X, y):
        """
        Given input samples and corresponding desired (true) output as indexes,
        this method calculates the percent error.
        For each input sample, if the predicted class output is not the same as the desired class,
        then it is considered one error. Percent error is number_of_errors/ number_of_samples.
        Note that the predicted class is the index of the node with maximum output.
        :param X: Array of input [n_samples,input_dimensions]
        :param y: Array of desired (target) outputs [n_samples]. This array includes the indexes of
        the desired (true) class.
        :return percent_error
        """
        y_hat = self.predict(X)
        y_hat_np = tf.math.argmax(y_hat, 1).numpy()
        print(y_hat_np, "y_hat shape: ", y_hat_np.shape)
        sum_error = 0.0
        for i in range(len(y)):
            if  not (y[i] == y_hat_np[i]):
                sum_error += 1
        return 100* (sum_error/len(y))

    def calculate_confusion_matrix(self, X, y):
        """
        Given input samples and corresponding desired (true) outputs as indexes,
        this method calculates the confusion matrix.
        :param X: Array of input [n_samples,input_dimensions]
        :param y: Array of desired (target) outputs [n_samples]. This array includes the indexes of
        the desired (true) class.
        :return confusion_matrix[number_of_classes,number_of_classes].
        Confusion matrix should be shown as the number of times that
        an image of class n is classified as class m.
        """
        y_hat = self.predict(X)
        # print(y_hat)
        y_hat_np = tf.math.argmax(y_hat, 1).numpy() 
        # print(X, "x shape: ", X.shape)
        # print(y, "y shape: ", y.shape)
        # print(y_hat_np, "y_hat shape: ", y_hat_np.shape)
        cm = tf.math.confusion_matrix( y, y_hat_np)
        print(cm)
        f = open('results.txt','w')
        f.write(str(y_hat.numpy()) + "\n\n\n" + str(y_hat_np) + "\n\n\n" + str(y))
        f.close()
        return cm

    def generate_batch(self, X, y, batch_size=5):
        mini_batches = []
        n_minibatches = X.shape[1] // batch_size
  
        for i in range(n_minibatches):
            X_mini = X[i * batch_size:(i + 1)*batch_size, : ]
            y_mini = y[i * batch_size:(i + 1)*batch_size]
            mini_batches.append((X_mini, y_mini))
        if X.shape[1] % batch_size != 0:
            X_mini = X[i * batch_size:X.shape[1], :]
            y_mini = y[i * batch_size:X.shape[1]]
            mini_batches.append((X_mini, y_mini))
        return mini_batches