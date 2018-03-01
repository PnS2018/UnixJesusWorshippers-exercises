# SESSION 1, EXERCISES
import numpy as np
from keras import backend as K

# EX.1

a=K.placeholder(shape=(5,))
b=K.placeholder(shape=(5,))
c=K.placeholder(shape=(5,))

funcOut1= a**2 + b**2 + c**2 + 2*b*c

sumx= K.function(inputs=(a, b, c), outputs=(funcOut1,))

# EX.2

# EX.3

w=K.placeholder(shape=(2,))
b=K.placeholder(shape=(1,))
x=K.placeholder(shape=(2,))

funcOut2=1/(1+(1/K.exp(w[0]*x[0]+w[1]*x[1]+b)))
multElements= K.function(inputs=(w, b, x), outputs=(funcOut2,))

# EX.4