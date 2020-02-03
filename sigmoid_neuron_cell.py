import math

def sigmoid(x): 
    y = 1.0 / (1.0 +math.exp(-x))
    return y

def activate(inputs, weights):
    h = 0
    for i, w in zip(inputs, weights): 
        h += i*w
    
    return sigmoid(h)

if __name__ == '__main__': #Testing example
    inputs = [.3,.9,.2]
    weights = [.5,.1,.4]
    output = activate(inputs,weights)
    print(output)