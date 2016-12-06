import numpy as np 
import matplotlib.pyplot as plt 

#Load data=========================================================================================================#
data='ex2data1.txt'
alldata = np.loadtxt(data, delimiter=',', usecols=(0,1,2), unpack=True)
X = np.transpose(np.array(alldata[:-1]))
y = np.transpose(np.array(alldata[-1:]))
m = y.size
X = np.insert(X, 0, 1, axis=1)

#Segregate data into positive and negative=========================================================================#

pos = np.array([X[i] for i in xrange(X.shape[0]) if y[i]==1])
neg = np.array([X[i] for i in xrange(X.shape[0]) if y[i]==0])
print "Got all data", (len(pos)+len(neg)==X.shape[0])

#Visualizing data=================================================================================================#

def plotData():
	plt.figure(figsize=(10,6))
	plt.grid(True)
	plt.plot(pos[:,1], pos[:,2], 'k+', label='Admitted')
	plt.plot(neg[:,1], neg[:,2], 'yo', label='Rejected')
	plt.xlabel('Exam one score')
	plt.ylabel('Exam two score')
	plt.legend()


plotData()
plt.show()
#Use sigmoid function in scipy.special: expit
from scipy.special import expit
 
#Define hypothesis for Logistic Regression z=theta.transpose x X
def h(Htheta, HX):
	return expit(np.dot(HX, Htheta))

#Define Cost function
def CostFunction(CFtheta ,CFy, CFX, CFlambda):
	m = y.size
	term1 = np.dot(CFy.T, np.log(h(CFtheta, CFX)))
	term2 = np.dot((1-y).T, np.log(1-h(CFtheta, CFX)))
	term3 = (CFlambda/(2*m))*(np.sum(np.dot(CFtheta.T, CFtheta)))
	return ((-1./m)*(np.sum(term1 + term2)) + term3)

#Initial Cost Function with theta=0

init_theta = np.zeros((X.shape[1], 1))
ld = 0.0

print "Initial Cost Function is: ", CostFunction(init_theta, y, X, ld)

#Use optimize and fmin from scipy instead of fminunc

from scipy import optimize

def OptTheta(OTy, OTX, OTtheta, OTlambda):
	result = optimize.fmin(CostFunction, x0 = OTtheta, args = (OTy, OTX, OTlambda), maxiter = 400, full_output = True)
	return result[0], result[1]

theta, optimal_CF = OptTheta(y, X, init_theta, ld)
print "Theta is: ", theta
print "Optimized Cost Function is:", optimal_CF

#Plot decision boundary now
#take 2 points from X as b_x
#decision boundary is when h=0
#th0 + th[1]*b_x + th[2]*b_y = 0
b_x = np.array([np.min(X[:,1]), np.max(X[:,1])])
b_y = (-1./theta[2])*(theta[0] + theta[1]*b_x)
plotData()
plt.plot(b_x, b_y, 'b-', label='Decision boundary')
plt.legend()
plt.show()

#Now predicting chances:
print "Enter 2 exam scores to know probability of admission:"
ex1 = float(raw_input("Exam 1 score: "))
ex2 = float(raw_input("Exam 2 score: "))

print "Probability of admission is: ", h(theta, np.array([1, ex1, ex2]))
#If you enter ex1=45, ex2=85, probability should be 0.77629

#===============================================================================================================#
#Predict accuracy of our model
def predict(mytheta, myxx):
	return h(mytheta, myxx) >= 0.5

pos_correct = float(np.sum(predict(theta, pos)))
neg_correct = float(np.sum(np.invert(predict(theta, neg))))
size = len(pos)+len(neg)
print "Model accuracy: ", float((pos_correct+neg_correct)/size)*100, "%"