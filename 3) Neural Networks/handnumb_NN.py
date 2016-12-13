#Recognize handwritten digits using a Neural Network
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.special import expit
#Load data
data1 = ('ex3data1.mat')
numdata = scipy.io.loadmat(data1)
X = numdata['X']
y = numdata['y']
X = np.insert(X, 0, 1, axis=1)
#Load weights, check shape of both weights
data2 = ('ex3weights.mat')
weights = scipy.io.loadmat(data2)
theta1 = weights['Theta1']
theta2 = weights['Theta2']
print "Theta1 shape is:", theta1.shape
print "Theta2 shape is:", theta2.shape
#Theta 1 is (25,401) Theta 2 is (10,26)


#Feedforward algorithm
thetatogether = [theta1, theta2]
def ffwd(row, alltheta):
	for itheta in range(len(alltheta)):
		thetanow = alltheta[itheta]
		a = expit(np.dot(thetanow, row))
		if itheta!=len(alltheta)-1:
			a = np.insert(a, 0, 1)
		row = a
	return a 

#Random check to see if its working
s = ffwd(X[0], thetatogether)
print np.argmax(s)
#Passed 3rd row of features to feedforward function and it returned array s of 10 hypothesis'. 
#np.argmax gave the index of largest value in array s which was 9 therefore 10th element which is mapped as 0(to avoid confusion as initially MATLAB was used)

#Prediction algorithm
classes = [10]+range(1,10)
def predictme(PMrow, PMalltheta):
	hyp = ffwd(X[PMrow], PMalltheta)
	ind = int(np.argmax(hyp))
	i = 0 if ind==9 else ind+1
	return i

#Take the row where the handwritten digit belongs from user
user_row = int(raw_input("Enter the row you want to predict: "))
pnum = predictme(user_row, thetatogether)
print "The feedforward algorithm predicts the number is: ", pnum
#Measure accuracy (Predicts 97.52% accuracy)
def mesacc(MAX, MAthetas, MAy):
	correct = []
	wrong = []
	for j in xrange(MAX.shape[0]):
		given = 0 if MAy[j]==10 else MAy[j]
		if predictme(j, MAthetas)==given:
			correct.append(j)
		else: wrong.append(j)
	percent = (float(len(correct))/X.shape[0])*100
	return percent, correct, wrong

acc_perc, cor, wro = mesacc(X, thetatogether, y)
print "Accuracy of neural network is: ", acc_perc, "%"