
# this file is create a simple neutral net work with numpy
import numpy as np 
# define the activation function

def sigmoid(x):
    return 1/(1+np.exp(-x))

def initializer_parameter(dim):
    w=np.zeros((dim,1))
    b=0.0
    return w,b

def propagate(w,b,X,y):
    m=X.shape[1]
    A=sigmoid(np.dot(w.T,X)+b)
    cost=-1/m*np.sum(Y*np.log(A)+(1-y)*np.log(1-A))

    dw=np.dot(X,(A-y).T)/m
    db=np.sum(A-y)/m
    cost=np.squeeze(cost)
    assert(cost.shape==())
    grad={'dw':dw,'db':db}

    return grad,cost

def backward_propagation(w,b,X,y,iteration,learnint_rate=0.01,print_cost=False):
    cost=[]
    for i in range(iteration):
        grad,cost=propagate(w,b,X,y)

        dw=grad['dw']
        db=grad['db']
        w=w-learnint_rate*dw
        b=b-learnint_rate*db
        if i % 100==0:
            print("cost after %i :%f"%(i,cost))

        params={"dw":w,"db":db}
        grad={"dw":dw,"db":db}

        return params,grad,cost


def predict(w, b, X):
    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)

    A = sigmoid(np.dot(w.T, X)+b)    
    for i in range(A.shape[1]):        
        if A[:, i] > 0.5:
            Y_prediction[:, i] = 1
        else:
            Y_prediction[:, i] = 0

    assert(Y_prediction.shape == (1, m))    
    return Y_prediction


def model(X_train, Y_train, X_test, Y_test, num_iterations = 2000, learning_rate = 0.5, print_cost = False):
    
    # initialize parameters with zeros (≈ 1 line of code)
    w, b = initializer_parameter(X_train.shape[0])    # Gradient descent (≈ 1 line of code)
    parameters, grads, costs = backward_propagation(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)    # Retrieve parameters w and b from dictionary "parameters"
    w = parameters["w"]
    b = parameters["b"]    # Predict test/train set examples (≈ 2 lines of code)
    Y_prediction_train = predict(w, b, X_train)
    Y_prediction_test = predict(w, b, X_test)    # Print train/test Errors
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    d = {"costs": costs,        
         "Y_prediction_test": Y_prediction_test, 
         "Y_prediction_train" : Y_prediction_train, 
         "w" : w, 
         "b" : b,         
         "learning_rate" : learning_rate,
         "num_iterations": num_iterations}    
    return d