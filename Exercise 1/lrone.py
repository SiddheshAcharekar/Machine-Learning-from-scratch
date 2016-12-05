import numpy as np
import matplotlib.pyplot as plt

datafile= 'ex1data1.txt'
cols= np.loadtxt(datafile, delimiter=',', usecols=(0,1), unpack=True)  #read all data seperated by commas and arranged in columns
#Form X and y matrix vectors

X= np.transpose(np.array(cols[:-1]))
y= np.transpose(np.array(cols[-1:]))
m= y.size

X= np.insert(X,0,1,axis=1)

#Plotting the data
plt.plot(X[:,1], y[:,0], 'rx', markersize=10)
plt.grid(True)
plt.xlabel('Population of city in 10,000s')
plt.ylabel('Profit in $10,000')
plt.show()
#================================================================================================================#
#Gradient descent
iterations=1500
alpha=0.01

def h(X, theta):
    return np.dot(X,theta)


def computecost(X, theta, y):
	m=y.size
	return (1.0/(2*m))*(np.sum(np.dot((h(X,theta)-y).T,(h(X,theta)-y))))

#Test initial cost with theta (0,0)
init_theta=np.zeros((2,1))
print computecost(X,init_theta,y)


#===============================================================================================================#
#Now gradient descent
def graddesc(X, y, alpha, iters, theta):
	m=y.size
	for val in xrange(iters):
		theta = theta- (alpha/m)*(np.dot((X.T),(h(X, theta)-y)))

	return theta

theta= graddesc(X, y, alpha, iterations, init_theta)

#=================================================================================================================#
#Predict profits
predict1=np.array((1, 3.5))  #Population=35000
predict2=np.array((1, 7.0))  #Population=70000

print "Profit in $ for population 35000: ", h(predict1, theta)[0]*10000
print "Profit in $ for population 70000: ", h(predict2, theta)[0]*10000
