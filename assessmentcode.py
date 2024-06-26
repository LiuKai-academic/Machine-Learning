#JC3509-Machine Learning

#Assessment1
#Name:Liu Kai
#Student ID: 50079690

import os
import numpy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold

#import the data
Datasets_loc = os.path.join(os.getcwd(),'datasets','Assessment1_Dataset.csv')
Datasets = pd.read_csv(Datasets_loc)

X = Datasets.drop('Producer', axis=1) ##178 #59 class1 #71 class2 # 48 class3
y = Datasets['Producer']

# we use 50% 25% 25% to divide the data into train,val and test datasets and keep the same ratio of three class
Xtrain = pd.concat([X[:30],X[59:96],X[130:154]], axis=0, ignore_index=True)
Xval = pd.concat([X[30:45],X[96:111],X[154:166]], axis=0, ignore_index=True)
Xtest = pd.concat([X[45:59],X[111:130],X[166:]], axis=0, ignore_index=True)
ytrain = pd.concat([y[:30],y[59:96],y[130:154]], axis=0, ignore_index=True)
yval = pd.concat([y[30:45],y[96:111],y[154:166]], axis=0, ignore_index=True)
ytest = pd.concat([y[45:59],y[111:130],y[166:]], axis=0, ignore_index=True)

# After dividing the three data sets, convert them into numpy format
Xtrain, Xval, Xtest, ytrain, yval, ytest = Xtrain.values, Xval.values, Xtest.values, ytrain.values, yval.values, ytest.values

# Normalization of the data set
col_norm = np.linalg.norm(Xtrain,axis=0)
X_train = Xtrain / col_norm[np.newaxis,:]
X_val = Xval / col_norm[np.newaxis,:]
X_test = Xtest / col_norm[np.newaxis,:]

#Keep a format that is not converted to one hot encoding for relevance calculation
ycorrelation = ytrain

#Converting label to one hot code facilitates loss calculation
onehot = np.eye(3)
ytrain= onehot[ytrain - 1]
yforloss = onehot[yval - 1]

#Define activation function
def reLU(x):
    return np.maximum(0,x)

#Define activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

#Define softmax function
def softmax(x):
    expx = np.exp(x - np.max(x, axis=1, keepdims=True))
    return expx / expx.sum(axis=1, keepdims=True)

#Define matrix multiplication
def logits(X,w):
    return np.dot(X,w)

#Compute the loss
def loss(X,Y):
    loss = -np.sum(np.multiply(Y, np.log(X))) / Y.shape[0]
    return loss

#Create the mini batches for trainning
def create_mini_batches(X, y, batch_size):
    mini_batches = []
    m = X.shape[0]
    indices = np.arange(m)
    np.random.shuffle(indices)#The index is scrambled to achieve random data

    #Traverse the entire training set according to the minibatch size
    for i in range(0, m, batch_size):
        batch_indices = indices[i:i + batch_size]
        X_mini = X[batch_indices]
        Y_mini = y[batch_indices]
        mini_batches.append((X_mini, Y_mini))

    return mini_batches

#Propagates the network forward and returns the parameters to be used later
def forward(X,w,w1,bias1,bias2):
    product = logits(X,w.T) + bias1 #X[91,13] w[3,13] product[91,3]
    activate = reLU(product) #activate[91,3]
    output = logits(activate,w1.T) + bias2 #output[91,3]
    prediction = softmax(output)
    return product,activate,prediction

#backpropagation
def backward(X, Y, w, w1, bias1, bias2, product, activate, prediction, learningrate,lambda_):
    m = X.shape[0]

    #The weight gradient of the model is calculated by partial derivation and chain rule to achieve gradient descent
    dZ2 = prediction - Y
    dW1 = (np.dot(dZ2.T, activate ) + lambda_ * w1)/ m #The model is regularized to prevent overfitting
    db2 = np.sum(dZ2, axis=0, keepdims=True) / m

    dA1 = np.dot(dZ2, w1)
    dZ1 = dA1 * (product > 0)
    dW = (np.dot(dZ1.T, X) + lambda_ * w)/ m #The model is regularized to prevent overfitting
    db1 = np.sum(dZ1, axis=0, keepdims=True) / m

    #The weights are updated by the calculated gradient
    w = w - learningrate * dW
    w1 = w1 - learningrate * dW1
    bias1 = bias1 - learningrate * db1
    bias2 = bias2 - learningrate * db2

    return w, w1, bias1, bias2

#
def calculate_accuracy(X, y, w, w1, bias1, bias2):
    m = X.shape[0]
    true = 0
    #Traversing every element of the verification set, propagating forward and predicting, calculating the correct number
    for i in range(m):
        prediction = forward(X[i], w, w1, bias1, bias2)[2]
        prediction = np.argmax(prediction)
        if prediction == y[i] - 1:
            true += 1

    #calculate the accuracy
    accuracy = true / m
    return accuracy

#Making predictions on fixed inputs (later used to calculate confusion matrix)
def predict(X, w, w1, bias1, bias2):
    m = X.shape[0]
    result = np.empty((1,0)) #initialize a empty np array
    for i in range(m):
        prediction = forward(X[i], w, w1, bias1, bias2)[2]
        prediction = np.argmax(prediction) + 1
        result = numpy.append(result,prediction) #After forward propagation, saved the result in the np array
    return result

#This function is not well written, but is simply used to calculate the loss of the verification set in this example to facilitate graphical comparison
def calthevalloss( w, w1, bias1, bias2):
    prediction = forward(X_val, w, w1, bias1, bias2)[2]
    valloss = loss(prediction, yforloss)
    return valloss

#Training code
def train(X_train, ytrain, X_val, yval, epochs, batch_size, learning_rate, nervecell, lambda_):
    features = X_train.shape[1]
    numpy.random.seed(1) #By fixing random seeds, it is possible to reproduce the results and facilitate the comparison of experimental results
    w = np.random.randn(nervecell,features) #According to the incoming data, the number of network neurons and the number of original data features are determined to initialize the weights
    numpy.random.seed(2)
    w1 = np.random.randn(3,nervecell) #initialize the second weight
    bias1 = np.zeros((1, nervecell)) #initialize the bias
    bias2 = np.zeros((1, 3))

    lossforplot = [] #initialize some empty list to store the result for plot
    accuracyforplot = []
    vallossforplot= []

    for i in range(epochs):
        mini_batches = create_mini_batches(X_train,ytrain,batch_size) #The mini batch is divided according to the incoming data before each training round
        lossforepoch = [] #loss for storing each round of minibatch

        for mini_batch in mini_batches: #Iterate over the entire data set
            X_mini,y_mini = mini_batch
            product, activate, prediction = forward(X_mini, w, w1, bias1, bias2) #Forward propagation is performed
            w, w1, bias1, bias2 = backward(X_mini, y_mini, w, w1, bias1, bias2, product, activate, prediction,learning_rate,lambda_) #Backpropagate and update the weights

            lossforepoch.append(loss(prediction,y_mini)/batch_size)
        lossforplot.append(sum(lossforepoch)) #Calculate the average loss for each round and store it in the list

        accuracyforplot.append(calculate_accuracy(X_val, yval, w, w1, bias1, bias2))
        vallossforplot.append(calthevalloss(w, w1, bias1, bias2)) #The weights obtained in this round are used to deduce the verification set and calculate the accuracy

    return w, w1, bias1, bias2, lossforplot,accuracyforplot,vallossforplot #Returns the last trained weights and the data to be plotted

# Initializes the parameters used for training
epochs = 3000
batch_size = 12
learning_rate = 0.015
nervecell = 5
lambda_ = 0.015

#Instantiation training
w_result, w1_result, bias1_result, bias2_result,lossforplot, accuracyforplot, valloss= train(X_train, ytrain, X_val, yval, epochs, batch_size, learning_rate, nervecell, lambda_)

# Draw the trainning and validation loss curve
plt.figure()
plt.plot(lossforplot, label='Training Loss')
plt.plot(valloss, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Across Epochs')
plt.legend()

plt.figure()
plt.plot(accuracyforplot, label='Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training Accuracy Across Epochs')
plt.legend()

plt.show()

#Test sets are used to evaluate model accuracy
acc = calculate_accuracy(X_test, ytest ,w_result, w1_result, bias1_result, bias2_result)

#Calculate the confusion matrix evaluation model
prediction = predict(X_test, w_result, w1_result, bias1_result, bias2_result)
c_matrix = confusion_matrix(ytest,prediction)
print('The confusion matrix:')
print(c_matrix)
print('The accuracy for test dataset:', acc)

#Calculate the impact of each feature in the data on the prediction result
correlations = pd.DataFrame(Xtrain, columns=X.columns).corrwith(pd.Series(ycorrelation))

plt.figure(figsize=(10, 8))
correlations.sort_values().plot(kind='barh')
plt.title('Feature Correlation with Target')
plt.xlabel('Correlation Coefficient')
plt.ylabel('Features')
plt.show()


#cross validation is performed to assess the level of the model
kf = KFold(n_splits=5, random_state=1, shuffle=True) #Define kfold cross validation and divide the data into five pieces
assessment_point = []

Xforcross,yforcross = X.values,y.values

for trainindex, valindex in kf.split(Xforcross):#The model is trained and evaluated by traversing different training and test sets

    x_train,x_val = Xforcross[trainindex],Xforcross[valindex]
    y_train,y_val = yforcross[trainindex],yforcross[valindex]
    x_train = x_train/col_norm
    x_val = x_val/col_norm

    y_train = onehot[y_train - 1]

    w, w1, bias1, bias2, _, _, _= train(x_train, y_train, x_val, y_val, epochs, batch_size, learning_rate, nervecell, lambda_)

    acc = calculate_accuracy(x_val, y_val, w, w1, bias1, bias2)

    assessment_point.append(acc)

model_point = np.mean(assessment_point)
print('Cross validation point(Mean accuracy):' , model_point)