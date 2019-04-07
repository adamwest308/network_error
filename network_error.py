from numpy import *
from matplotlib.pyplot import *
from tkinter import *

error_array_0 = []
error_array_1 = []
error_array_2 = []
error_array_3 = []
answer_array_0 = []
answer_array_1 = []
answer_array_2 = []
answer_array_3 = []

# sigmoid function
def sigmoid(x, derivative = False):
    return 1/(1+exp(-x))

#find derivative of sigmoid function
def sigmoid_derivative(x):
    return x * (1.0 - x)
    
# input dataset
input_dataset = array([  [0,0,1],
                            [0,1,1],
                            [1,0,1],
                            [1,1,1] ])
                
# output dataset
output_dataset = array([[0,0,1,1]]).T

# seed random numbers to make calculation
# deterministic (just a good practice)
random.seed(1)

# initialize weights randomly with mean 0
#a 3x1 matrix of weights
random_weights = 2*random.random((3,1)) - 1

for iteration in range(10000):
    
    #forward propagation
    #works out to a 4x1 matrix of propabilities for the answers
    calculated_answers = sigmoid(dot(input_dataset, random_weights))
    
    #calculate error
    #still a 4x1 matrix
    error = output_dataset - calculated_answers
    
    #multiply error by slope of sigmoid at values in weighted_inputs
    #this means that high and low estimates have less of a change,
    #unless there is a high level of error
    weighted_inputs_change = error * sigmoid_derivative(calculated_answers)
    
    #update weights
    #multiplication of the transformed input dataset (3x4 matrix)
    #and the changes 
    random_weights += dot(input_dataset.T, weighted_inputs_change)
    
    if iteration % 1000 == 0:
        error_array_0.append(error[0])
        error_array_1.append(error[1])
        error_array_2.append(error[2])
        error_array_3.append(error[3])
        answer_array_0.append(calculated_answers[0])
        answer_array_1.append(calculated_answers[1])
        answer_array_2.append(calculated_answers[2])
        answer_array_3.append(calculated_answers[3])
    
print ("Output after Training")
print (calculated_answers)

x = arange(0, 10000, 1000)

error_plot = subplot(211)
plot(x,error_array_0)
plot(x,error_array_1)
plot(x,error_array_2)
plot(x,error_array_3)
error_plot.set(xlabel = "number of iterations", 
ylabel = "degree of error (range of 0-1)", 
title = "graph of error approaching zero for all 4 answers")


answer_plot = subplot(212)
plot(x,answer_array_0)
plot(x,answer_array_1)
plot(x,answer_array_2)
plot(x,answer_array_3)

w1 = Tk()
w1.title("Interface")
w1.geometry('450x200')
f1 = Frame(w1)
f1.pack()

grid()
tight_layout()
show()
root.mainloop()