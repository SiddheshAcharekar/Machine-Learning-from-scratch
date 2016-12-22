import numpy as np 
import matplotlib.pyplot as plt 
from scipy.special import expit
import itertools as it  
import scipy.io 
import random
import matplotlib.cm as cm
#Load data====================================================================================================#
images = ('ex4data1.mat')
img = scipy.io.loadmat(images)
X = img['X']
y = img['y']
X = np.insert(X, 0, 1, axis=1)
wgth = ('ex4weights.mat')
weights = scipy.io.loadmat(wgth)
theta1 = weights['Theta1']
theta2 = weights['Theta2']

print "X shape is", X.shape, "and y shape is ", y.shape
print "Theta1 shape is", theta1.shape, " and Theta2 shape is ", theta2.shape
print "Unique classes from y are: ", np.unique(y)

#Functions to display data====================================================================================#

def reshapetoimage(row):
	im = row[1:].reshape(20,20)
	return im.T

def displaydata():
	w, l = 20, 20
	r, c = 0, 0
	final_img = np.zeros((200,200))
	indices = random.sample(range(X.shape[0]), 100)
	for index in indices:
		if c==10:
			r += 1
			c = 0
		final_img[r*w:r*w+20, c*l: c*l+20] = reshapetoimage(X[index])
		c += 1
	fig = plt.figure(figsize=(6,6))
	img = scipy.misc.toimage(final_img)
	plt.imshow(img, cmap=cm.Greys_r)

#Call function to view how the images actually look like==================================================#
displaydata()
plt.show()

#Store constants==========================================================================================#
#We have a 3 layer neural network here with first layer of 400, hidden layer of 25, output layer of 10. ls is layer_size
ip_ls = 400
hidden_ls = 25
op_ls = 10
training_examples = y.size

#Flatten out parameters first=============================================================================#
alltheta = [theta1, theta2]

def flattheta(FLthetas):
	flatten_theta = [itheta.flatten() for itheta in FLthetas]
	allflattheta = list(it.chain.from_iterable(flatten_theta))
	assert (ip_ls+1)*hidden_ls+(hidden_ls+1)*op_ls == len(allflattheta)
	return np.array(allflattheta).reshape(len(allflattheta), 1)

def reshapetheta(flat_theta):
	theta1 = flat_theta[:hidden_ls*(ip_ls+1)].reshape(hidden_ls,ip_ls+1)
	theta2 = flat_theta[hidden_ls*(ip_ls+1):].reshape(op_ls,hidden_ls+1)
	return [theta1, theta2]

def flattenX(FX):
	return np.array(FX.flatten()).reshape(FX.size,1)

def reshapeX(flat_X):
	return np.array(flat_X).reshape(training_examples, ip_ls+1)

#Cost Function============================================================================================#
def ffwdprop(row, thetas):
	features = row
	za_per_layer = []
	for itheta in xrange(len(thetas)):
		thetanow = thetas[itheta]
		z = np.dot(thetanow, features)
		a = expit(z)
		za_per_layer.append((z, a))
		if itheta == len(thetas)-1:
			return za_per_layer
		a = np.insert(a, 0, 1)
		features = a

zzz = ffwdprop(X[2567], alltheta)

def costfunction(flattenedX, flattenedtheta, CFy, CFlambda):
	X_back = reshapeX(flattenedX)
	theta_back = reshapetheta(flattenedtheta)
	cost_total = 0.0
	m = training_examples
	for irow in xrange(m):
		row = X_back[irow]
		hypo = ffwdprop(row, theta_back)[-1][1]
		temp_y = np.zeros((10,1))
		temp_y[CFy[irow]-1] = 1

		cost_for_row = np.dot(temp_y.T, np.log(hypo)) + np.dot((1-temp_y).T, np.log(1-hypo))
		cost_total += cost_for_row
	cost_total *= (-1.0/m)	
	reg_term = 0.0
	for th in theta_back:
		reg_term += np.sum(th*th) #Make sure this is elementwise multiplication
	reg_term *= (CFlambda/(2*m))

	return cost_total+reg_term

print "Initial cost function with lambda=1.0 is: ", costfunction(flattenX(X), flattheta(alltheta), y, 1.0)

#Backpropogation algorithm=====================================================================================#

def gradofsigmoid(z):
	grad = expit(z)
	return grad*(1-grad)

#Initialise nonsymmetric thetas 
def initrandomthetas():
	init_epsilon = 0.12
	random_thetas = [np.random.rand(hidden_ls, ip_ls+1)*2*init_epsilon - init_epsilon, np.random.rand(op_ls, hidden_ls+1)*2*init_epsilon - init_epsilon]
	return random_thetas

def backprop(BPX, BPthetas, BPy, BPlambda):
	bp_X = reshapeX(BPX)
	bp_thetas = reshapetheta(BPthetas)

	m = training_examples

	#Define deltas, they should have the same shape as thetas
	#Deltas are loss function values corresponding to each weight

	Del_1 = np.zeros((hidden_ls, ip_ls+1))
	Del_2 = np.zeros((op_ls, hidden_ls+1))

	for row in xrange(m):
		irow = bp_X[row].reshape(bp_X.shape[1], 1)
		set_of_za = ffwdprop(bp_X[row], bp_thetas)
		z2 = set_of_za[0][0]
		a2 = set_of_za[0][1]
		z3 = set_of_za[1][0]
		a3 = set_of_za[1][1]
		temp_y = np.zeros((10,1))
		temp_y[BPy[row]-1] = 1
		del3 = a3-temp_y
		del2 = np.dot(bp_thetas[1][1:,:], del3)*gradofsigmoid(z2)
		a2 = np.insert(a2, 0, 1, axis=0)
		Del_1 += np.dot(del2, a1.T)
		Del_2 += np.dot(del3, a2.T)

	Del_1 = Del_1/float(m)
	Del_2 = Del_2/float(m)

	#Regularize deltas
	Del_1[:,1:] = Del_1[:,1:]+(float(BPlambda)/m)*BPthetas[0][:,1:]
	Del_2[:,1:] = Del_2[:,1:]+(float(BPlambda)/m)*BPthetas[1][:,1:]

	return flattheta([Del_1,Del_2])

d1d2 = backprop(flattenX(X), flattheta(alltheta), y, 0)
print d1d2

#Gradient checking=======================================================================================#
def checkGradient(CGthetas,CGDs,CGX,CGy,CGlambda=0.):
    CGeps = 0.0001
    flattened = flattenParams(CGthetas)
    flattenedDs = flattenParams(CGDs)
    CGX_flattened = flattenX(CGX)
    n_elems = len(flattened) 
    #Pick ten random elements, compute numerical gradient, compare to respective D's
    for i in xrange(10):
        x = int(np.random.rand()*n_elems)
        epsvec = np.zeros((n_elems,1))
        epsvec[x] = CGeps
        cost_high = computeCost(flattened + epsvec,CGX_flattened,CGy,CGlambda)
        cost_low  = computeCost(flattened - epsvec,CGX_flattened,CGy,CGlambda)
        CGgrad = (cost_high - cost_low) / float(2*CGeps)
        print "Element: %d. Numerical Gradient = %f. BackProp Gradient = %f."%(x,CGgrad,flattenedDs[x])

checkGradient(alltheta,[D1, D2],X,y)

#Train network=============================================================================================#
def trainNN(mylambda=0.):
    """
    Function that generates random initial theta matrices, optimizes them,
    and returns a list of two re-shaped theta matrices
    """

    randomThetas_unrolled = flattenParams(genRandThetas())
    result = scipy.optimize.fmin_cg(computeCost, x0=randomThetas_unrolled, fprime=backPropagate, \
                               args=(flattenX(X),y,mylambda),maxiter=50,disp=True,full_output=True)
    return reshapeParams(result[0])

#Predict=======================================================================================================#
def predictNN(row,Thetas):
    """
    Function that takes a row of features, propagates them through the
    NN, and returns the predicted integer that was hand written
    """
    classes = range(1,10) + [10]
    output = propagateForward(row,Thetas)
    #-1 means last layer, 1 means "a" instead of "z"
    return classes[np.argmax(output[-1][1])] 

#Compute accuracy================================================================================================#

def computeAccuracy(myX,myThetas,myy):
    """
    Function that loops over all of the rows in X (all of the handwritten images)
    and predicts what digit is written given the thetas. Check if it's correct, and
    compute an efficiency.
    """
    n_correct, n_total = 0, myX.shape[0]
    for irow in xrange(n_total):
        if int(predictNN(myX[irow],myThetas)) == int(myy[irow]): 
            n_correct += 1
    print "Training set accuracy: %0.1f%%"%(100*(float(n_correct)/n_total))

computeAccuracy(X,learned_Thetas,y)