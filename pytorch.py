import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
MODELS=500
BACKPROP=200
TEST_MODELS=int(MODELS/10)
VALIDATION_SET=TEST_MODELS
SIZE=VALIDATION_SET+MODELS+TEST_MODELS
np.random.seed(42)
mnist = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())
LAYER_SIZE=np.array([0, 784, 816, 832, 842])
SUMS=np.array([0, 32, 48, 58])
class NN():
    def __init__(self):
        #Training Set Init
        self.learning_rate=0.1
        self.decay_rate=0.006
        self.epsilon=1e-12
        self.alpha=1e-1
        self.validation_costs=np.empty((BACKPROP,))
        self.train_costs=np.empty((BACKPROP,))
        self.weights = np.empty(3, dtype=object)
        self.biases = np.empty(3, dtype=object)
        for i in range(0, len(LAYER_SIZE)-2):
            if i == len(LAYER_SIZE) - 3:
                self.weights[i] = np.random.randn(LAYER_SIZE[i+2]-LAYER_SIZE[i+1], LAYER_SIZE[i+1]-LAYER_SIZE[i]) * np.sqrt(1. / (LAYER_SIZE[i+1] - LAYER_SIZE[i]))
            else:
                self.weights[i] = np.random.randn(LAYER_SIZE[i+2]-LAYER_SIZE[i+1], LAYER_SIZE[i+1]-LAYER_SIZE[i]) * np.sqrt(2. / (LAYER_SIZE[i+1] - LAYER_SIZE[i]))
            self.biases[i]=np.zeros(SUMS[i+1]-SUMS[i]) 
        self.x_f=np.empty((MODELS,LAYER_SIZE[-1]))
        self.z=np.empty((MODELS,SUMS[-1]))
        self.z_d=np.empty((MODELS,SUMS[-1]))
        self.y=np.zeros((MODELS,10))
        #Validation Set Init
        self.validation_z=np.empty((MODELS,SUMS[-1]))
        self.validation_examples=np.empty((VALIDATION_SET, LAYER_SIZE[-1]))
        self.validation_answers=np.zeros((VALIDATION_SET, 10))
        #Test Set Init
        self.test_z=np.empty((MODELS,SUMS[-1]))
        self.test_examples=np.empty((TEST_MODELS, LAYER_SIZE[-1]))
        self.test_answers=np.zeros((TEST_MODELS, 10))
        #Assigning Random Examples to the Training, Validation and Test Set
        self.random_indecies=np.random.choice(len(mnist), size=SIZE, replace=False)
        self.train_indices = self.random_indecies[:MODELS]
        self.validation_indices = self.random_indecies[MODELS:MODELS + VALIDATION_SET]
        self.test_indices = self.random_indecies[MODELS + VALIDATION_SET:]
        self.Random_Examples_Generator(self.x_f, self.y, self.train_indices)
        self.Random_Examples_Generator(self.validation_examples, self.validation_answers, self.validation_indices)
        self.Random_Examples_Generator(self.test_examples, self.test_answers, self.test_indices)
    def Random_Examples_Generator(self, input, answers, indecies):
        for idx, instance in enumerate(indecies):
            img_tensor, label = mnist[instance]
            input[idx][:LAYER_SIZE[1]]=img_tensor.view(-1).numpy()
            answers[idx][label]=1
    def summ(self, W, b, x):
        return np.dot(W, x)+b
    def softmax(self, x):
        if np.any(np.isnan(x)) or np.any(np.isinf(x)):
            print("Wrong computations")
        e_x = np.exp(x - np.max(x))
        return e_x/(e_x.sum()+self.epsilon)
    def Leaky_ReLU(self, x): 
        # The usual ReLU caused too many zero entries, so I used Leaky ReLU
        return np.where(x>0, x, x*self.alpha)
    def Leaky_ReLU_d(self, z):
        return np.where(z>0, 1, self.alpha)
    def feedforward_train(self):
            cost=0
            k=0
            for k in range(MODELS):
                """Number of models"""
                i=0
                for i in range(len(LAYER_SIZE)-2):
                    if i == (len(LAYER_SIZE)-3):
                        self.z[k][SUMS[i]:SUMS[i+1]]=self.summ(self.weights[i], self.biases[i], self.x_f[k][LAYER_SIZE[i]:LAYER_SIZE[i+1]])
                        self.x_f[k][LAYER_SIZE[i+1]:LAYER_SIZE[i+2]]=self.softmax(self.z[k][SUMS[i]:SUMS[i+1]])
                    else:
                        self.z[k][SUMS[i]:SUMS[i+1]]=self.summ(self.weights[i], self.biases[i], self.x_f[k][LAYER_SIZE[i]:LAYER_SIZE[i+1]])
                        self.x_f[k][LAYER_SIZE[i+1]:LAYER_SIZE[i+2]]=self.Leaky_ReLU(self.z[k][SUMS[i]:SUMS[i+1]])
                cost+=self.cost_function(self.x_f[k][LAYER_SIZE[-2]:], self.y[k], self.weights)
            return cost/MODELS
    def feedforward_validation(self):
            validation_z=np.zeros_like(self.validation_z)
            validation_examples=np.copy(self.validation_examples)
            validation_cost=0
            k=0
            for k in range(VALIDATION_SET):
                """Number of models"""
                i=0
                for i in range(len(LAYER_SIZE)-2):
                    if i == (len(LAYER_SIZE)-3):
                        validation_z[k][SUMS[i]:SUMS[i+1]]=self.summ(self.weights[i], self.biases[i], validation_examples[k][LAYER_SIZE[i]:LAYER_SIZE[i+1]])
                        validation_examples[k][LAYER_SIZE[i+1]:LAYER_SIZE[i+2]]=self.softmax(validation_z[k][SUMS[i]:SUMS[i+1]])
                    else:
                        validation_z[k][SUMS[i]:SUMS[i+1]]=self.summ(self.weights[i], self.biases[i], validation_examples[k][LAYER_SIZE[i]:LAYER_SIZE[i+1]])
                        validation_examples[k][LAYER_SIZE[i+1]:LAYER_SIZE[i+2]]=self.Leaky_ReLU(validation_z[k][SUMS[i]:SUMS[i+1]])
                validation_cost+=self.cost_function(validation_examples[k][LAYER_SIZE[-2]:], self.validation_answers[k], self.weights)
            return validation_cost/VALIDATION_SET
    def feedforward_test(self, weights):
            k=0
            for k in range(TEST_MODELS):
                """Number of models"""
                i=0
                for i in range(len(LAYER_SIZE)-2):
                    if i == (len(LAYER_SIZE)-3):
                        self.test_z[k][SUMS[i]:SUMS[i+1]]=self.summ(weights[i], self.biases[i], self.test_examples[k][LAYER_SIZE[i]:LAYER_SIZE[i+1]])
                        self.test_examples[k][LAYER_SIZE[i+1]:LAYER_SIZE[i+2]]=self.softmax(self.test_z[k][SUMS[i]:SUMS[i+1]])
                    else:
                        self.test_z[k][SUMS[i]:SUMS[i+1]]=self.summ(weights[i], self.biases[i], self.test_examples[k][LAYER_SIZE[i]:LAYER_SIZE[i+1]])
                        self.test_examples[k][LAYER_SIZE[i+1]:LAYER_SIZE[i+2]]=self.Leaky_ReLU(self.test_z[k][SUMS[i]:SUMS[i+1]])
    def cost_function(self, output, answer, W):
            label_index = np.argmax(answer)
            entries_sum=sum(np.sum(W[l]**2) for l in range(len(LAYER_SIZE)-2))
            return -np.log(output[label_index]+self.epsilon)+((self.alpha)*entries_sum)
    def backprop(self):
        self.grad_w2=np.zeros((10,16))
        self.grad_w1=np.zeros((16,32))
        self.grad_w0=np.zeros((32,784))
        self.grad_b2=np.zeros(10)
        self.grad_b1=np.zeros(16)
        self.grad_b0=np.zeros(32)
        for k in range(MODELS):
            self.delta2=(self.x_f[k][LAYER_SIZE[3]:]-self.y[k])
            self.grad_w2+=np.outer(self.delta2,self.x_f[k][LAYER_SIZE[2]:LAYER_SIZE[3]])+(self.alpha*self.weights[2])
            self.grad_b2+=self.delta2
            self.delta1=(self.weights[2].T@self.delta2)*self.Leaky_ReLU_d(self.z[k][SUMS[1]:SUMS[2]]) 
            self.grad_b1+=self.delta1
            self.grad_w1+=np.outer(self.delta1, self.x_f[k][LAYER_SIZE[1]:LAYER_SIZE[2]])+(self.alpha*self.weights[1])
            self.delta0=(self.weights[1].T@self.delta1)*self.Leaky_ReLU_d(self.z[k][SUMS[0]:SUMS[1]])
            self.grad_b0+=self.delta0
            self.grad_w0+=np.outer(self.delta0, self.x_f[k][LAYER_SIZE[0]:LAYER_SIZE[1]])+(self.alpha*self.weights[0])
        #print(self.delta2.sum())
        if np.any(np.isnan(self.grad_w0)) or np.any(np.isinf(self.grad_w0)) or np.any(np.isnan(self.grad_w1)) or np.any(np.isinf(self.grad_w1)) or np.any(np.isnan(self.grad_w2)) or np.any(np.isinf(self.grad_w2)):
            print("Incorrect weight gradients")
        self.weights[0]-=(self.learning_rate*(self.grad_w0/MODELS))
        self.weights[1]-=(self.learning_rate*(self.grad_w1/MODELS))
        self.weights[2]-=(self.learning_rate*(self.grad_w2/MODELS))
        self.biases[0]-=(self.learning_rate*(self.grad_b0/MODELS))
        self.biases[1]-=(self.learning_rate*(self.grad_b1/MODELS))
        self.biases[2]-=(self.learning_rate*(self.grad_b2/MODELS))
    
network=NN()
for epoch in range(BACKPROP):
      network.learning_rate=network.learning_rate/(1+network.decay_rate)
      network.feedforward_train()
    #   print(f"{network.x_f[1][LAYER_SIZE[-2]:][np.argmax(network.y[1])]}:{network.y[1][np.argmax(network.y[1])]}")
      network.backprop()
varience=np.zeros(BACKPROP,)
i, epoch=0, 0
weights_storage=np.zeros(BACKPROP, dtype=object)
optimal=0
while i!=10 and epoch!=(BACKPROP):
    network.learning_rate*=(1/(1+network.decay_rate*i))
    network.train_costs[epoch]=network.feedforward_train()
    network.validation_costs[epoch]=network.feedforward_validation()
    weights_storage[epoch]=[w.copy() for w in network.weights]
    varience[epoch]=abs(network.validation_costs[epoch]-network.train_costs[epoch])
    if varience[epoch-3]<varience[epoch-2]<varience[epoch-1]<varience[epoch] and epoch>=3 and epoch<=(BACKPROP-4):
        optimal=np.argmin(network.validation_costs[:epoch+1])
        print(optimal)
        break
    if epoch!=(BACKPROP-2):
        network.backprop()
    epoch+=1
print(weights_storage[optimal][2])
network.feedforward_test(weights_storage[optimal])
correct=0
test_accuracy=np.zeros((TEST_MODELS,))
for epoch in range(TEST_MODELS):
    if np.argmax(network.test_examples[epoch][LAYER_SIZE[-2]:])==np.argmax(network.test_answers[epoch]):
        correct+=1
    test_accuracy[epoch]=network.test_examples[epoch][LAYER_SIZE[-2]+np.argmax(network.test_answers[epoch])]
print(f"The models accuracy is: {(correct/TEST_MODELS)}%")
plt.plot(network.train_costs, label="Training Loss")
plt.plot(network.validation_costs, label="Validation Loss")
plt.plot(varience, label="Varience")
plt.plot(test_accuracy, label="Test Models' Accuracy")
plt.legend()
plt.show()

"""self.sum_d=np.ones(self.biases[1].shape)
            self.sum_d=self.ReLU_d(self.z[k][32:48])
            self.sum2_d=np.ones(self.biases[0].shape)
            self.sum2_d=self.ReLU_d(self.z[k][:32])
            self.help2=self.x_f[k][layer_size[-2]:layer_size[-1]]-self.y[k]
            self.w2=np.add(self.w2, np.outer(self.help2,self.x_f[k][layer_size[-3]:layer_size[-2]].T))
            self.b2+=self.help2
            self.help1=(self.weights[2].T@(self.x_f[k][layer_size[-2]:layer_size[-1]]-self.y[k]))*self.sum_d
            self.w1=np.add(np.outer(self.help1,self.x_f[k][layer_size[-4]:layer_size[-3]].T), self.w1)
            self.b1+=self.help1
            self.help0=self.weights[1].T@((self.weights[2].T@(self.x_f[k][layer_size[-2]:layer_size[-1]]-self.y[k]))*self.sum_d)*self.sum2_d
            self.w0=np.add(np.outer(self.help0,self.x_f[k][:layer_size[1]]), self.w0)
            self.b0+=self.help0"""
        
"""self.w2/=MODELS
        self.weights[2]-=(self.learning_rate*self.w2)
        self.b2/=MODELS
        self.biases[2]-=self.learning_rate*self.b2
        self.w1/=MODELS
        self.weights[1]-=self.learning_rate*self.w1
        self.b1/=MODELS
        self.biases[1]-=self.learning_rate*self.b1
        self.w0/=MODELS
        self.weights[0]-=self.learning_rate*self.w0
        self.b0/=MODELS
        self.biases[0]-=self.learning_rate*self.b0"""
"""
test_indices = np.random.randint(0, len(mnist), size=TEST_MODELS)
test_x = []
test_y = np.zeros((TEST_MODELS, 10))
for i, idx in enumerate(test_indices):
    img_tensor, label = mnist[idx]
    test_x.append(img_tensor.view(-1).numpy())
    test_y[i][label] = 1
network.x = np.array(test_x)
network.y = test_y
for i in range(TEST_MODELS):
    network.x_f[i][:layer_size[1]] = network.x[i]
network.feedforward()
correct = 0
for i in range(TEST_MODELS):
    predicted = np.argmax(network.x_f[i][layer_size[3]:])
    actual = np.argmax(network.y[i])
    prob = network.x_f[i][layer_size[3] + predicted]
    if predicted == actual:
        correct += 1
"""

"""testing_indecies=list(np.random.randint(0,10) for _ in range(MODELS))
test_models=np.zeros((MODELS, 784))
test_answers=np.zeros((MODELS,10))
for idx, i in enumerate(testing_indecies):
    img_tensor, label = mnist[i]
    self_x_instance = img_tensor.view(-1).numpy()
    test_models[idx] = self_x_instance"""
"""Initializing the array that will contain the pixels of every training
        instance which are represented as one dimensional arrayeach of them,
      so 10-dimensional array"""
"""    test_answers[idx][label] = 1
    network.x_f[idx][:layer_size[1]]=test_models[idx]
network.y=test_answers
network.feedforward()
for idx, k in enumerate(testing_indecies):
    i=np.argmax(network.y[idx])
    print("Predicted probability:", network.y[idx][i])
    print("Predicted output:", network.x_f[idx][layer_size[3]+i])
    if network.y[idx][i]<1e-12:
        print("Division by zero:(")
    else:
        print(f"The accuracy of the following instance is: {network.x_f[idx][layer_size[3]+i]/network.y[idx][i]}")"""
    
"""for i in range(len(layer_size)-2):
            self.z_d.append(self.ReLU_d(self.z[i]))"""
"""for i in range(layer_size[-1]-layer_size[-2]):
            for l in range(layer_size[-2]-layer_size[-3]):
                self.total_1=0
                for k in range(MODELS):
                    self.total_1+=self.x_f[k][layer_size[-3]+l]*(self.x_f[k][layer_size[-2]+i]-self.y[k][i])
                self.weights[2][i][l]-=self.learning_rate*(self.total_1/MODELS)"""
"""self.n_out, self.n_in = self.weights[1].shape 
        for k in range(MODELS):
            i=np.argmax(self.y[k])
            self.total+=self.weights[2][i].T*np.outer(self.x_f[k][layer_size[1]:layer_size[2]], np.ones(self.n_out))*self.z_d[1]*(self.x_f[k][layer_size[3]+i]-self.y[k][i])
        print(self.total/MODELS)"""