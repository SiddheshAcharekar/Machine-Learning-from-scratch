#Regularized Logistic regression using second set of data
"""
If you use unregularized logistic regression on ex2data2 then you get the following results:
Optimized CF: 0.6902 so CF doesnt converge to global minimum
Model accuracy:  55.08% really poor accuracy
"""
#This has an almost circular decision boundary hence needs regularization
import numpy as np 
import matplotlib.pyplot as plt 

#Load data=========================================================================================================#
data = 'ex2data2.txt'
alldata = np.loadtxt(data, delimiter=',', usecols=(0,1,2), unpack = True)
X = np.transpose(np.array(alldata[:-1]))
y = np.transpose(np.array(alldata[-1:]))
X = np.insert(X, 0, 1, axis=1)
m = y.size

#Segregate data into positive and negative=========================================================================#
pos = np.array([X[i] for i in xrange(X.shape[0]) if y[i]==1])
neg = np.array([X[i] for i in xrange(X.shape[0]) if y[i]==0])
if len(pos)+len(neg)==X.shape[0] :
	print "Got all data"

#Visualizing data=================================================================================================#
def plotData():
	plt.figure(figsize=(10,10))
	plt.plot(pos[:, 1], pos[:, 2], 'k+', label='Accepted microchip')
	plt.plot(neg[:, 1], neg[:, 2], 'yo', label='Rejected microchip')
	plt.grid(True)
	plt.xlabel("Test 1")
	plt.ylabel("Test 2")
	plt.legend()

plotData()
plt.show()

#Hypothesis, call sigmoid function expit============================================================================#
from scipy.special import expit
def h(Htheta, HX):
	return expit(np.dot(HX, Htheta))

# Run it on initial theta of all zeros to get h=0.5
checkh = raw_input("Check if initial hypothesis is 0.5? Y/N: ")
init_theta = np.zeros((X.shape[1],1))
if checkh=='Y' or checkh=='y':
	print h(init_theta, X)

#Cost function with CFlambda set to something
def CostFunction(CFtheta ,CFy, CFX, CFlambda):
	m = y.size
	term1 = np.dot(CFy.T, np.log(h(CFtheta, CFX)))
	term2 = np.dot((1-y).T, np.log(1-h(CFtheta, CFX)))
	term3 = (CFlambda/(2*m))*(np.sum(np.dot(CFtheta.T, CFtheta)))
	return ((-1./m)*(np.sum(term1 + term2)) + term3)

#Check cost function with theta all 0's it should be 0.693 for this dataset
checkcf = raw_input("Check initial cost function? Y/N: ")
if checkcf=='Y' or checkcf=='y':
	print "Initial cost function without regularizing is: ", CostFunction(init_theta, y, X, 0)

#Feature Mapping to introduce more features to increase accuracy
#Didnt understand this first, code borrowed from user Kaleko
def mapFeature( x1col, x2col ):
    """ 
    Function that takes in a column of n- x1's, a column of n- x2s, and builds
    a n- x 28-dim matrix of featuers as described in the homework assignment
    """
    degrees = 6
    out = np.ones( (x1col.shape[0], 1) )

    for i in range(1, degrees+1):
        for j in range(0, i+1):
            term1 = x1col ** (i-j)
            term2 = x2col ** (j)
            term  = (term1 * term2).reshape( term1.shape[0], 1 ) 
            out   = np.hstack(( out, term ))
    return out

#Create feature-mapped X matrix
mappedX = mapFeature(X[:,1],X[:,2])

#Now check cost function with mappedX
init_theta = np.zeros((mappedX.shape[1], 1))
ld = 0.0
print "Initial Cost function after adding more features and no regularization: ", CostFunction(init_theta, y, mappedX, 0)

#Import optimize from scipy
from scipy import optimize

def OptTheta(OTy, OTX, OTtheta, OTlambda):
	result1 = optimize.fmin(CostFunction, x0=OTtheta, args=(OTy, OTX, OTlambda), maxiter=400, full_output=True)
	return result1[0], result1[1]

theta, optimizedCF = OptTheta(y, mappedX, init_theta, ld)
print theta
print optimizedCF
print "Function hasnt converged yet"
"""
Optimized CF shows no convergence and maximum iterations exceeding
So use minimize instead of fmin
"""
def OptRegTheta(ORTy, ORTX, ORTtheta, ORTlambda):
	result2 = optimize.minimize(CostFunction, x0=ORTtheta, args=(ORTy, ORTX, ORTlambda), method='BFGS', options={"maxiter":400, "disp":False})
	return np.array([result2.x]), result2.fun

theta, optimizedCF = OptRegTheta(y, mappedX, init_theta,100.0)
print theta
print optimizedCF

"""
If i set
lambda = 0.0 optimizedCF is 0.263 ==========Overfit
lambda = 1.0 optimizedCF is 0.535 ==========Good fit
lambda = 10.0 optimizedCF is 0.6511 ========Under fit
lambda = 100.0 optimizedCF is 0.68652 ======Under fit
"""