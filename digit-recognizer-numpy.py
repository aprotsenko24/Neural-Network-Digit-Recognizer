import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
TRAIN_MODELS=500
TEST_MODELS=75
VALIDATION_MODELS=75
BACKPROP=1000
mnist = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())

random_indecies=np.random.choice(len(mnist), size=TRAIN_MODELS+TEST_MODELS+VALIDATION_MODELS, replace=False)
train_indecies=random_indecies[:TRAIN_MODELS]
validation_indecies=random_indecies[TRAIN_MODELS:TRAIN_MODELS+VALIDATION_MODELS]
test_indecies=random_indecies[TRAIN_MODELS+VALIDATION_MODELS:]

class Layers():
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
    epsilon=1e-12
    alpha=1e-1
    learning_rate=0.04
    decay_rate=1e-5
    @classmethod
    def __init__(cls):
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
        cls.learning_rate*=np.exp(-cls.decay_rate)
    @classmethod
    def gradient_descend(cls, gradient, delta):
        cls.weights-=cls.learning_rate*(gradient/Train.models.__len__())
        cls.biases-=cls.learning_rate*(delta/Train.models.__len__())
class Train():
    @classmethod
    def __init__(cls, Models, indecies):
        cls.models=np.empty(Models, dtype=object)
        cls.answers=np.zeros((Models, 10))
        for idx, instance in enumerate(indecies):
            img_tensor, label = mnist[instance]
            cls.models[idx]=img_tensor.view(-1).numpy()
            cls.answers[idx][label]=1
class Val(Train):
    @classmethod
    def __init__(cls, Models, indecies):
        super().__init__(Models, indecies)
class Test(Train):
    @classmethod
    def __init__(cls, Models, indecies):
        super().__init__(Models, indecies)
class CNN():
    def __init__(self, index, model):
        self.index=index
        self.a=np.empty(len(Layers.LAYER_SIZE), dtype=object)
        for i in range(len(Layers.LAYER_SIZE)):
            self.a[i]=np.zeros(Layers.LAYER_SIZE[i])
        self.z=np.copy(self.a[1:])
        self.a[0]=model[self.index]
    def feedforward_dropout(self):
        for i in range(len(Layers.LAYER_SIZE)-1):
            self.z[i]=np.dot(CNNParams.weights[i], self.a[i])+CNNParams.biases[i]
            mask=np.random.binomial(1, 0.5, self.a[i+1])
            if i==len(Layers.LAYER_SIZE)-2:
                self.a[i+1]=mask*Softmax.func(self.z[i])
            else:
                self.a[i+1]=mask*Leaky_ReLU.func(self.z[i], CNNParams.alpha)
    def feedforward(self):
        for i in range(len(Layers.LAYER_SIZE)-1):
            self.z[i]=np.dot(CNNParams.weights[i], self.a[i])+CNNParams.biases[i]
            if i==len(Layers.LAYER_SIZE)-2:
                self.a[i+1]=Softmax.func(self.z[i])
            else:
                self.a[i+1]=Leaky_ReLU.func(self.z[i], CNNParams.alpha)
    def backprop(self):
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
    @staticmethod
    def func(x):
        if np.any(np.isnan(x)) or np.any(np.isinf(x)):
            print("Wrong computations")
        e_x = np.exp(x - np.max(x))
        return e_x/(e_x.sum()+CNNParams.epsilon)

class ReLU():
    @staticmethod  
    def func(z):
        return np.where(z>0, z, z*0)
    @staticmethod
    def diff(z):
        return np.where(z>0, 1, 0)

class Leaky_ReLU():
    @staticmethod
    def func(z, alpha):
        return np.where(z>0, z, alpha*z)
    @staticmethod
    def diff(z, alpha):
        return np.where(z>0, 1, alpha)
    
class Cost():
    @staticmethod
    def func(a, y):
        entries_sum=sum(np.sum(w**2) for w in CNNParams.weights)
        return -np.sum(y*np.log(a[-1]+CNNParams.epsilon))+(CNNParams.alpha*entries_sum)

Layers.add(784, 28, 10)
CNNParams()
Val(VALIDATION_MODELS, validation_indecies)
# print(Val.answers[0], Val.models[0])
Train(TRAIN_MODELS, train_indecies)   # Train.models[i] - image of the model in pixel brightness representation using array
                # Train.answers[i] - an array representing an answer to the specific model with a single
                # index value identified by 1 for the answer number (answer number=index)
                # - You can change the number of models passed by passing a different parameter instead of const TRAIN_MODELS
training_set=np.empty(Train.models.__len__(), dtype=object)
val_set=np.empty(Val.models.__len__(), dtype=object)
test_set=np.empty(Test.models.__len__(), dtype=object)
train_cost=np.zeros(BACKPROP)
val_cost=np.copy(train_cost)
test_cost=np.zeros((Test.models.__len__()))
varience=np.copy(val_cost)
weights_storage=np.empty(BACKPROP, dtype=object)
biases_storage=np.empty(BACKPROP, dtype=object)
for k in range(BACKPROP):
    correct_train=0
    correct_val=0
    tr_cost=0
    vl_cost=0
    weights_storage[k]=CNNParams.weights
    biases_storage[k]=CNNParams.biases
    for i in range(Train.models.__len__()):
        training_set[i]=CNN(i, Train.models)
        training_set[i].feedforward()
        gradient, delta = training_set[i].backprop()
        CNNParams.gradient_descend(gradient, delta)
        tr_cost+=Cost.func(training_set[i].a, Train.answers[i])
        if np.argmax(training_set[i].a[-1])==np.argmax(Train.answers[i]):
            correct_train+=1
    train_cost[k]=tr_cost/Train.models.__len__()
    for v in range(Val.models.__len__()):
        val_set[v]=CNN(v, Val.models)
        val_set[v].feedforward()
        vl_cost+=Cost.func(val_set[v].a, Val.answers[v])
        if np.argmax(val_set[v].a[-1])==np.argmax(Val.answers[v]):
            correct_val+=1
    val_cost[k]=vl_cost/Val.models.__len__()
    varience[k]=abs(val_cost[k]-train_cost[k])
    CNNParams.decay_func()
count=0
for i in range(varience.__len__()):
    if varience[i+1]>varience[i]:
        count+=1
        if count==10:
            break
    else:
        count=0
optimal=np.argmin(val_cost[:i+2])
CNNParams.weights=weights_storage[optimal]
CNNParams.biases=biases_storage[optimal]
correct_test=0
Test(TEST_MODELS, test_indecies)
ts_cost=0
for i in range(Test.models.__len__()):
    test_set[i]=CNN(i, Test.models)
    test_set[i].feedforward()
    test_cost[i]=Cost.func(test_set[i].a, Test.answers[i])
    if np.argmax(test_set[i].a[-1])==np.argmax(Test.answers[i]):
        correct_test+=1
print((correct_train/Train.models.__len__())*100)
print((correct_val/Val.models.__len__())*100)
print((correct_test/Test.models.__len__())*100)
plt.subplot(2, 2, 1)
plt.plot(train_cost, label="Training Set")
plt.plot(val_cost, label="Validation Set")
plt.subplot(2, 2, 2)
plt.plot(varience, label="Varience")
plt.subplot(2, 2, 3)
plt.plot(test_cost, label="Test Cost")
plt.show()
# By untagging the plt.show() function, you can see the trend of Loss function
