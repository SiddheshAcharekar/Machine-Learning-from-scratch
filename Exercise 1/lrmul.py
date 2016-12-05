import numpy as np 
import matplotlib.pyplot as plt 

#Load data #=====================================================================================================#
data='ex1data2.txt'
#Read file contents
cols=np.loadtxt(data, delimiter=',', usecols=(0,1,2), unpack=True)
X= np.transpose(np.array(cols[:-1]))
y= np.transpose(np.array(cols[-1:]))
m= y.size

X= np.insert(X,0,1, axis=1)

#Plot and visualize the data
plt.grid(True)
plt.xlim([-100,5000])
dummy= plt.hist(X[:,0], label='col1')
dummy= plt.hist(X[:,1], label='col2')
dummy= plt.hist(X[:,2], label='col3')
dummy= plt.legend()
plt.title('This shows the need for feature normalization')
plt.xlabel('Values of features')
plt.ylabel('Count')
plt.show()

#Feature Normalization==============================================================================================#
#Create a copy of X
Xc=X.copy()
feature_std=[]
feature_m=[]
for icol in xrange(1, X.shape[1]):
	feature_std.append(np.std(Xc[:,icol]))
	feature_m.append(np.mean(Xc[:,icol]))
	Xc[:,icol]= (Xc[:,icol]-feature_m[-1])/feature_std[-1]

#Now plot copy of X to see normalised features
plt.grid(True)
plt.xlim([-5,5])
dummy= plt.hist(Xc[:,0], label='col1')
dummy= plt.hist(Xc[:,1], label='col2')
dummy= plt.hist(Xc[:,2], label='col3')
plt.title('Normalised features')
plt.xlabel('Value of features')
plt.ylabel('Count')
dummy=plt.legend()
plt.show()

#Gradient Descent======================================================================================================#

def h(X, theta):
	return np.dot(X, theta)

def costfunc(X, y, theta):
	m=y.size
	return (1./(2*m))*(np.sum(np.dot((h(X,theta) -y).T, h(X, theta)- y)))

print "Initial hypothesis: ", h(Xc, np.zeros((3,1)))
print "Initial Cost function: ", costfunc(Xc, y, np.zeros((3,1)))

def graddesc(X, y, theta, alpha, iterations):
	m= y.size
	for val in xrange(iterations):
		theta = theta- (alpha/m)*(np.dot((X.T), (h(X, theta) - y)))
	return theta

#Define parameters for gradient descent
init_theta=np.zeros((3,1))
alpha=0.1
iters=1500
theta = graddesc(Xc, y, init_theta, alpha, iters)

#To find house prices, normalise values of test also=================================================================#

predict1=[1, 1650, 3]
for pos in range(1, len(predict1)):
	predict1[pos]= (predict1[pos]-feature_m[pos-1])/(feature_std[pos-1])

print predict1
print "Cost of a house with 1650sq.ft and 3 rooms is: ", h(predict1, theta)