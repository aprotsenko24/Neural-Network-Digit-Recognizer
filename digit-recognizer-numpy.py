import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
MODELS=500
BACKPROP=1000
mnist = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())

class Layers():
    """
    A class with global class variables for the CNN layers:
       Attributes:
          COUNTER (int) - counts how many layers do I have in the numpy array
          LAYER_SIZE (np.ndarray) - the np.array of the layer sizes (how many neurons every layer has)
       Methods:
          add(*layers_size):
             Appends new layer(s) to the LAYER_SIZE array
          delete(*indecies):
             Deletes the layer by specifying the index of the layer stored in the LAYER_SIZE array
    """
    COUNTER=0
    LAYER_SIZE=np.array([])
    @classmethod
    def add(cls, *layer_size):
        cls.LAYER_SIZE=list()
        for layer in layer_size:
            cls.LAYER_SIZE.append(layer)
        cls.LAYER_SIZE=np.array(cls.LAYER_SIZE)
    @classmethod
    def delete(cls, *indecies):
        for index in indecies:
            cls.LAYER_SIZE=np.delete(cls.LAYER_SIZE, index, axis=1)
class CNNParams():
    """
    A class that stores the global class parameters for the CNN:
       Attributes:
          weights (np.ndarray) - stores all weights' matrices for the CNN
          biases (np.ndarray) - stores all biases' vectors for the CNN
          gradients (np.ndarray) - stores all gradients' matrcies in one array for backpropogation
          delta (np.ndarray) - stores "deltas" for biases to perform backpropogation
          learning_rate (float) - a rate with which the model trains, gets changed over the learning process to reach global minimum
          decay_rate (float) - a rate with which learning_rate decays
          alpha (float) - used in the cost function and L2 Regularization for punishing large weights. 
                        Also can be used in Leaky_ReLU.func()/Leaky_ReLU.diff()
          epsilon (float) - prevents errors when log() function is not defined, when it is zero, by adding infinitesimal number
       Methods:
          __init__():
             Initializes arrays that do not have the same encapsulated arrays (several-dimensional arrays)
             Attributes:
                weights (np.ndarray)
                biases (np.ndarray)
                gradients (np.ndarray)
                delta (np.ndarray)
          decay_func():
             Decays the learning_rate parameter
          gradient_descend(gradients, delta):
             Performs gradient descend on the provided training data. Gets inputs from the backprop() function (explanation later)
             Attributes:
                gradients (np.ndarray) - gradients that are optimizing weights for the training set
                delta (np.ndarray) - gradients for the biases' optimization during the backpropogation
    """
    epsilon=1e-12
    alpha=1e-1
    learning_rate=0.04
    decay_rate=1e-5
    @classmethod
    def __init__(cls):
        """
        Initializes weights, biases, gradients (gradients for weights) and delta (gradients for biases) with 
        appropriate shapes and sizes
           Attributes:
              weights (np.ndarray)
              biases (np.ndarray)
              gradients (np.ndarray)
              delta (np.ndarray)
        """
        cls.weights = np.empty(len(Layers.LAYER_SIZE)-1, dtype=object)
        cls.biases = np.empty(len(Layers.LAYER_SIZE)-1, dtype=object)
        cls.gradients=np.copy(cls.weights)
        cls.delta=np.copy(cls.biases)
        for i in range(len(Layers.LAYER_SIZE)-1):
            if i == len(Layers.LAYER_SIZE) - 2:
                cls.weights[i] = np.random.randn(Layers.LAYER_SIZE[i+1], Layers.LAYER_SIZE[i]) * (1. / (Layers.LAYER_SIZE[i]))
                cls.gradients[i] = np.zeros((Layers.LAYER_SIZE[i+1], Layers.LAYER_SIZE[i]))
                # He initialization
            else:
                cls.weights[i] = np.random.randn(Layers.LAYER_SIZE[i+1], Layers.LAYER_SIZE[i]) * (2. / (Layers.LAYER_SIZE[i+1] + Layers.LAYER_SIZE[i]))
                cls.gradients[i] = np.zeros((Layers.LAYER_SIZE[i+1], Layers.LAYER_SIZE[i]))
                # Xavier initialization
            cls.biases[i]=np.full(Layers.LAYER_SIZE[i+1], 0)
            cls.delta[i]=np.copy(cls.biases)
    @classmethod
    def decay_func(cls):
        """
        Decays the learing rate (float) exponentially w/r/t decay_rate (float) class variable
        """
        cls.learning_rate*=np.exp(-cls.decay_rate)
    @classmethod
    def gradient_descend(cls, gradient, delta):
        """
        Performs the gradient descend for one epoch w/r/t learing_rate, gradient and delta for a single model
           Attributes:
              gradient (np.ndarray) - gradients array with matrices to parameterize weights during the gradient descend
              delta (np.ndarray) - gradients array with arrays (vectors) to parameterize biases during the gradeint descend
        """
        cls.weights-=cls.learning_rate*(gradient/Train.models.__len__())
        cls.biases-=cls.learning_rate*(delta/Train.models.__len__())
class Train():
    """
    A 'Train' class generates the input number of training models with answers the user wants to train, 
    taking the traing models from Mnist dataset
        Attributes:
            models (np.ndarray) - an array storing the arrays with pixels' brightness as for every training model
            answers (np.ndarray) - an array storing answers for every mnist dataset sample image
        Methods:
            __init__(Models):
                Takes an input as the number of models the user wants to train and stores these samples along 
                with their answers as class variables models and answers respectively        
    """
    @classmethod
    def __init__(cls, Models):
        """
        Initializes the specific number of models, depending on the what number user inputs as for Models attribute
        Attributes:
            models (np.ndarray)
            answers (np.ndarray)
        """
        cls.models=np.empty(Models, dtype=object)
        cls.answers=np.zeros((Models, 10))
        for i in range(Models):
            img_tensor, label = mnist[i]
            cls.models[i]=img_tensor.view(-1).numpy()
            cls.answers[i][label]=1
class CNN():
    """
    The 'CNN' class performs training process and outputs the gradients that are essential for tweaking the weights & biases for a single training model
        Attributes:
            index (int) - the index of the training model
            a (np.ndarray) - the array storing the values of the neurons for every layer
            z (np.ndarray) - the array storing the sum after weights matrix multiplictaion with the previous input neurons' 
            array (L-1) plus the biases vector
            gradients (np.ndarray)
            delta (np.ndarray)
            LAYER_SIZE (np.ndarray)
        Methods:
            __init__(index)
                Initializes zeros arrays for neurons layers and their sums accordingly
                Attributes:
                    index (int)
                    a (np.ndarray)
                    z (np.ndarray)
    """
    def __init__(self, index):
        """
        Initializes zeros arrays for neurons layers and their sums accordingly
        Attributes:
            a (np.ndarray)
            z (np.ndarray)
            index (int)
        """
        self.index=index
        self.a=np.empty(len(Layers.LAYER_SIZE), dtype=object)
        for i in range(len(Layers.LAYER_SIZE)):
            self.a[i]=np.zeros(Layers.LAYER_SIZE[i])
        self.z=np.copy(self.a[1:])
        self.a[0]=Train.models[self.index]
    def feedforward(self):
        """
        Performs feedforward on the Convelutional Neural Network
        Attributes:
            z (np.ndarray)
            a (np.ndarray)
        """
        for i in range(len(Layers.LAYER_SIZE)-1):
            self.z[i]=np.dot(CNNParams.weights[i], self.a[i])+CNNParams.biases[i]
            if i==len(Layers.LAYER_SIZE)-2:
                self.a[i+1]=Softmax.func(self.z[i])
            else:
                self.a[i+1]=Leaky_ReLU.func(self.z[i], CNNParams.alpha)
    def backprop(self):
        """
        Performs backpropogation on the single model from the Training set and returns gradients and 
        delta values used later in the gradient_descend function
        Attributes:
            gradients (np.ndarray)
            delta (np.ndarray)
            biases (np.ndarray)
            weights (np.ndarray)
            z (np.ndarray)
            alpha (float)
            Return values:
                gradients, delta (tuple)
                - used for gradient descend
        """
        CNNParams.gradients=np.zeros_like(CNNParams.gradients)
        CNNParams.biases=np.zeros_like(CNNParams.biases)
        for i in range(len(Layers.LAYER_SIZE)-1, 0, -1):
            if i==len(Layers.LAYER_SIZE)-1:
                CNNParams.delta[i-1]=(self.a[i]-Train.answers[self.index])
            else:
                CNNParams.delta[i-1]=np.dot(CNNParams.weights[i].T, CNNParams.delta[i])*Leaky_ReLU.diff(self.z[i-1], CNNParams.alpha)
            CNNParams.gradients[i-1]=np.outer(CNNParams.delta[i-1],self.a[i-1])+(CNNParams.alpha*CNNParams.weights[i-1])
        return CNNParams.gradients, CNNParams.delta
class Softmax():
    """
    Softmax function used for feedforward and squishing the range of the number to 0-1
        Attributes:
            x (Any) - in my model, an array of latter layer neurons used for computing output layer neurons
            e_x (Any) - exponential value of x
            epsilon (float) - infinitesimal value to prevent NaN or inf values if there is division by 0
        Methods:
            func(x):
                Performs computations on the latter layer neurons and gives an output as the values for the last layer nuerons
    """
    @staticmethod
    def func(x):
        """"
        Performs computations on the latter layer neurons and gives an output as the values for the last layer nuerons
        Attributes:
            x (Any) - in my model, an array of latter layer neurons used for computing output layer neurons
            e_x (Any) - exponential value of x
            epsilon (int) - infinitesimal value to prevent NaN or inf values if there is division by 0
        Return values:
            Computes the output result for the last layer neurons
        """
        if np.any(np.isnan(x)) or np.any(np.isinf(x)):
            print("Wrong computations")
        e_x = np.exp(x - np.max(x))
        return e_x/(e_x.sum()+CNNParams.epsilon)

class ReLU():
    """
    Intemidiate function for the hidden layers, gives output for all layers' neuron except of the final output layer
    Attributes:
        z (Any) - input array of neurons
    Methods:
        func(z):
            Perfoms computations and gives output for ReLU function based on the input (usually, the previous layer neurons)
            Attributes:
                z (Any) - input array of neurons
            Return value:
                returns z if the sum for the specific layer is greater than zero, otherwise return 0
        diff(z):
            Performs differentiation w/r/t ReLU function
            Attributes:
                z (Any) - input array of neurons
            Return value:
                returns 1 if the sum for the specific layer is greater than zero, otherwise return 0
    """
    @staticmethod  
    def func(z):
        """
        func(x):
            Perfoms computations and gives output for ReLU function based on the input (usually, the previous layer neurons)
            Attributes:
                z (Any) - input array of neurons
            Return value:
                returns z if the sum for the specific layer is greater than zero, otherwise return 0
        """
        return np.where(z>0, z, z*0)
    @staticmethod
    def diff(z):
        """
            diff(z):
            Performs differentiation w/r/t ReLU function
            Attributes:
                z (Any) - input array of neurons
            Return value:
                returns 1 if the sum for the specific layer is greater than zero, otherwise return 0
        """
        return np.where(z>0, 1, 0)

class Leaky_ReLU():
    """
    Intemidiate function for the hidden layers, gives output for all layers' neuron except of the final output layer
    Attributes:
        z (Any) - input array of neurons
        alpha (float) - a small float value for retaining the weights values small, not zeroed as it is in ReLU, if they are negative
    Methods:
        func(z):
            Perfoms computations and gives output for ReLU function based on the input (usually, the previous layer neurons)
            Attributes:
                alpha (float) - a small float value for retaining the weights values small, not zeroed as it is in ReLU, if they are negative
                z (Any) - input array of neurons
            Return value:
                returns z if the sum for the specific layer is greater than zero, otherwise returns alpha times z
        diff(z):
            Performs differentiation w/r/t ReLU function
            Attributes:
                z (Any) - input array of neurons
                alpha (float) - a small float value for retaining the weights values small, not zeroed as it is in ReLU, if they are negative
            Return value:
                returns 1 if the sum for the specific layer is greater than zero, otherwise return alpha times z
    """
    @staticmethod
    def func(z, alpha):
        """"
            Perfoms computations and gives output for ReLU function based on the input (usually, the previous layer neurons)
            Attributes:
                alpha (float) - a small float value for retaining the weights values small, not zeroed as it is in ReLU, if they are negative
                z (Any) - input array of neurons
            Return value:
                returns z if the sum for the specific layer is greater than zero, otherwise returns alpha times z

        """
        return np.where(z>0, z, alpha*z)
    @staticmethod
    def diff(z, alpha):
        """
            Performs differentiation w/r/t ReLU function
            Attributes:
                z (Any) - input array of neurons
                alpha (float) - a small float value for retaining the weights values small, not zeroed as it is in ReLU, if they are negative
            Return value:
                returns 1 if the sum for the specific layer is greater than zero, otherwise return alpha times z
        """
        return np.where(z>0, 1, alpha)
    
class Cost():
    """
    Cross-entropy loss function with L2 Regularization combined
    Attributes:
        a (Any) - input array of the predicted outputs
        y (Any) - input array with answers to the specific model
    Methods:
        func(a, y):
            Calculates the value for the cost function
            Attributes:
                a (Any) - input array of the predicted outputs
                y (Any) - input array with answers to the specific model
            Return values:
                the value for the cross-entropy loss function combined with L2 Regularization
    """
    @staticmethod
    def func(a, y):
        """
            Calculates the value for the cost function
            Attributes:
                a (Any) - input array of the predicted outputs
                y (Any) - input array with answers to the specific model
            Return values:
                the value for the cross-entropy loss function combined with L2 Regularization
        """
        entries_sum=sum(np.sum(w**2) for w in CNNParams.weights)
        return -np.sum(y*np.log(a[-1]+CNNParams.epsilon))+(CNNParams.alpha*entries_sum)

Layers.add(784, 16, 10)
CNNParams()
Train(MODELS)   # Train.models[i] - image of the model in pixel brightness representation using array
                # Train.answers[i] - an array representing an answer to the specific model with a single
                # index value identified by 1 for the answer number (answer number=index)
                # - You can change the number of models passed by passing a different parameter instead of const MODELS
training_set=np.empty(Train.models.__len__(), dtype=object)
cost=np.zeros(BACKPROP)
for k in range(BACKPROP):
    correct=0
    total_cost=0
    for i in range(Train.models.__len__()):
        training_set[i]=CNN(i)
        training_set[i].feedforward()
        gradient, delta = training_set[i].backprop()
        CNNParams.gradient_descend(gradient, delta)
        total_cost+=Cost.func(training_set[i].a, Train.answers[i])
        if np.argmax(training_set[i].a[-1])==np.argmax(Train.answers[i]):
            correct+=1
    cost[k]=total_cost/Train.models.__len__()
    CNNParams.decay_func()
print((correct/Train.models.__len__())*100)
plt.plot(cost)
# plt.show()
# By untagging the plt.show() function, you can see the trend of Loss function
