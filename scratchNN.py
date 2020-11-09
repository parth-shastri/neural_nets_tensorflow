import numpy as np
from nnfs.datasets import spiral_data
from matplotlib import pyplot as plt
import random

#x, y = spiral_data(100, 3)
#print(x.shape, y.shape)
def createdata(points, classes):
    x = np.zeros((points*classes, 2))
    y = np.zeros((points*classes))
    for class_number in range(classes):
        ix = range(points*class_number, points*(class_number+1))
        r = np.linspace(0.0, 1, points)
        t = np.linspace(class_number*4, (class_number+1)*4, points) +np.random.randn(points)*0.2
        x[ix] = np.c_[r*np.sin(t*2.5), r*np.cos(t*2.5)]
        y[ix] = class_number
    return x,y
np.random.seed(0)
#X, Y = createdata(100, 2)
#plt.scatter(X[:, 0], X[:, 1], c=Y, s=40, cmap=plt.cm.Spectral)

#plt.show()
#print(X.shape)
#Y = Y.reshape((200, 1))

N = 100 # number of points per class
D = 2 # dimensionality
K = 2 # number of classes
X = np.zeros((N*K,D)) # data matrix (each row = single example)
y = np.zeros(N*K, dtype='uint8') # class labels
for j in range(K):
  ix = range(N*j,N*(j+1))
  r = np.linspace(0.0,1,N) # radius
  t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.2 # theta
  X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
  y[ix] = j
X = X.reshape((200,2))
y = y.reshape((200, 1))


class Dense_layer:
    def __init__(self, n_inputs, n_neurons):
        self.weights = np.random.randn(n_inputs, n_neurons)*0.01
        self.bias = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights)+self.bias
        #print(self.output)

    def backward(self, batchsize, previous_activation, dg, da):
        self.dz = da*dg
        self.dw = np.dot(np.transpose(self.dz), previous_activation) *(1/batchsize)
        self.db = np.sum(np.transpose(self.dz), axis=1, keepdims=True)*(1/ batchsize)
        self.da_1 = np.dot(self.weights, np.transpose(self.dz))
        #print(self.dw.shape)
        #print(self.dz.shape, da.shape, dg.shape, self.da_1.shape)





class Relu_Activation:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

        '''def reluderivation(x):
            x[x<=0] = 0
            x[x>0] = 1
            return x
        self.dg = reluderivation(inputs)'''

        self.dg = np.heaviside(inputs, 1)
        #print(self.dg.shape)
        #print(self.output)


class Sigmoid_activation:
    def forward(self, inputs):
        self.output = 1/(1 + np.exp(inputs))
        self.dg = self.output*(1-self.output)


class tanh_activation:
    def forward(self, inputs):
        self.output = np.tanh(inputs)
        self.dg = 1 - np.power(self.output, 2)



class binary_crossentropy:
    def forward(self, inputs, expected):
        m = expected.shape[0]
        self.output = np.multiply(np.log(inputs), expected)+np.multiply((1-expected), np.log(1-inputs))
        self.dloss = (np.divide(expected, inputs) - np.divide((1-expected), (1-inputs)))
        self.cost = -(1/m)*np.sum(self.output, axis=0)
        #print(self.dloss.shape, inputs.shape, expected.shape)
        print(self.cost)


class optimizer:
    def __init__(self, lr):
        self.lr = lr
    def forward(self, weight, dw):
        weight += -self.lr*dw
        return weight


class Model:
    def __init__(self, **kwargs):
        self.model = kwargs
    def compile(self, optimizer, loss):
        self.loss = loss
        self.optimizer = optimizer
    def fit(self, train, labels, epochs, batchsize):
        for i in range(epochs):
            self.model['layer1'].forward(train)
            self.model['activation1'].forward(self.model['layer1'].output)
            self.model['layer2'].forward(self.model['activation1'].output)
            self.model['activation2'].forward(self.model['layer2'].output)


            self.loss.forward(self.model['activation2'].output, labels)
            if i % 1000 == 0:
                print('epochs : {}/{} ===>  val_loss'.format(i + 1, epochs))
                print(self.loss.cost, self.model['activation2'].output[100:])




            self.model['layer2'].backward(batchsize, self.model['activation1'].output, self.model['activation2'].dg, self.loss.dloss)

            self.model['layer1'].backward(batchsize, train, self.model['activation1'].dg, np.transpose(self.model['layer2'].da_1))
            #print(self.model["layer1"].weights)

            updated_weights1 = self.optimizer.forward(self.model['layer1'].weights, np.transpose(self.model['layer1'].dw))
            updated_bias1 = self.optimizer.forward(self.model['layer1'].bias, np.transpose(self.model['layer1'].db))
            updated_weights2 = self.optimizer.forward(self.model['layer2'].weights, np.transpose(self.model['layer2'].dw))
            updated_bias2 = self.optimizer.forward(self.model['layer2'].bias, np.transpose(self.model['layer2'].db))
            #print(updated_bias1.shape, updated_bias2.shape, updated_weights1.shape, updated_weights2.shape)
            #print(updated_bias2, updated_bias1, np.transpose(self.model['layer1'].db),  np.transpose(self.model['layer2'].db))
            self.model['layer1'].weights = updated_weights1
            self.model['layer1'].bias = updated_bias1
            self.model['layer2'].weights = updated_weights2
            self.model['layer2'].bias = updated_bias2


model = Model(layer1=Dense_layer(X.shape[1], 100),
                        activation1=Relu_Activation(),
                        layer2=Dense_layer(100, 1),
                        activation2=Sigmoid_activation()
                           )
epochs = 10000
model.compile(optimizer=optimizer(1), loss=binary_crossentropy())
model.fit(X, y, epochs=epochs, batchsize=200)








