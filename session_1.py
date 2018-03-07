# SESSION 1, EXERCISES
import numpy as np
from keras import backend as K

# EX.1

a = K.placeholder(shape=(5,))
b = K.placeholder(shape=(5,))
c = K.placeholder(shape=(5,))

funcOut1 = a ** 2 + b ** 2 + c ** 2 + 2 * b * c

sumx = K.function(inputs=(a, b, c), outputs=(funcOut1,))

# EX.2

x = K.ones(shape=())

tanh_tensor = (K.exp(x) - K.exp(-x)) / (K.exp(x) + K.exp(-x))

grad_tanh_tensor = K.gradients(loss=tanh_tensor, variables=[x])

tanh_functions = K.function(inputs=[x], outputs=[tanh_tensor, grad_tanh_tensor[0]])

for i in [-100, -1, 0, 1, 100]:
    print(tanh_functions([i]))

# EX.3

w = K.placeholder(shape=(2,))
b = K.placeholder(shape=(1,))
x = K.placeholder(shape=(2,))

funcOut2 = 1 / (1 + (1 / K.exp(w[0] * x[0] + w[1] * x[1] + b)))
multElements = K.function(inputs=(w, b, x), outputs=(funcOut2,))

# EX.4

n = 4 # arbitrary number
x = K.placeholder(shape=(n, 1)) #x = np.ones(n)
coeff = K.placeholder(shape=(n, 1))
n_vec = np.arange(n)
p = K.placeholder(shape=())

monome_vec = np.power(x,n_vec)

if coeff.shape == monome_vec.shape:
    p = coeff * x

grad_polynome = K.gradients(loss=p, variables=(monome_vec))
grad_function = K.function(inputs=(coeff,x), outputs=(grad_polynome)) #why doesnt this work?

print(grad_function(np.array([1,3,5,7]),np.array([3,3,3,3])))
