import numpy as np
def train(x, y):
    x = np.array(x)
    y = np.array(y)
    num = (x*y).mean()-x.mean()*y.mean()
    deno = (x**2).mean()-(x.mean())**2
    m = num/deno
    c = y.mean()-m*x.mean()
    return m, c

def predict(x, m, c):
    return (m*x + c)

def score(y_test, y_predicted):
    y_test = np.array(y_test)
    y_predicted = np.array(y_predicted)
    u = ((y_test-y_predicted)**2).sum()
    v = ((y_test - y_predicted.mean())**2).sum()
    return 1-u/v

def cost(x, y, m, c):
    return ((y-m*x+c)**2).mean()

def gd(x,y, learning_rate, m, c):
    m_slope = 0
    c_slope = 0
    x = np.array(x)
    y = np.array(y)
    L = len(x)
    for i in range(L):
        xi = x[i]
        yi = y[i][0]
        m_slope += (-2/L)*(yi-m*xi-c)*xi
        c_slope += (-2/L)*(yi-m*xi-c)
    new_m = m-learning_rate*m_slope
    new_c = c-learning_rate*c_slope    
    return new_m, new_c

def gradient_descent(x_train, y_train, learning_rate, num):    # num---> number of iterations
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    m = 0
    c = 0
    for i in range(num):
        m, c = gd(x_train, y_train, learning_rate, m, c)
    return m, c

def generic_gd(x_train, y_train, learning_rate, num):

    L = np.shape(x_train)[1]
    m = np.zeros(L, float)
    c = np.zeros(L, float)
    for i in range(L):
        for j in range(num):
            m[i],c[i] = gd(x_train[i], y_train, learning_rate, m[i], c[i])

    return m, c
        
def predict_generic(x, m, c):
    return (m*x+c).sum()