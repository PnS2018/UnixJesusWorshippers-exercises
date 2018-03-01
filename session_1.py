# Exercise 2

from keras import backend as K

x = K.ones(shape=())

tanh_tensor = (K.exp(x) - K.exp(-x)) / (K.exp(x) + K.exp(-x))

grad_tanh_tensor = K.gradients(loss=tanh_tensor, variables=[x])

tanh_functions = K.function(inputs=[x], outputs=[tanh_tensor, grad_tanh_tensor[0]])

for i in [-100, -1, 0, 1, 100]:
    print(tanh_functions([i]))
