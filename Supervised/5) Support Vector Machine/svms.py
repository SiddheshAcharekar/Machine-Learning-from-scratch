
import numpy as np
import matplotlib.pyplot as plt
import scipy.io 
import scipy.optimize 
from sklearn import svm 


##### Support Vector Machines #########################################################

##### Visualize the dataset ################################################

datafile = 'ex6data1.mat'
data_mat = scipy.io.loadmat( datafile )
#Training set
X, y = data_mat['X'], data_mat['y']


X_pos = np.array([X[i] for i in xrange(X.shape[0]) if y[i] == 1])
X_neg = np.array([X[i] for i in xrange(X.shape[0]) if y[i] == 0])


def plotData():
    plt.figure(figsize=(10,6))
    plt.plot(X_pos[:,0],X_pos[:,1],'k+',label='Positive')
    plt.plot(X_neg[:,0],X_neg[:,1],'yo',label='Negative')
    plt.xlabel('1st Variable')
    plt.ylabel('2nd Variable')
    plt.legend()
    
plotData()

#Function to draw the SVM boundary
def plotBoundary(my_svm, xmin, xmax, ymin, ymax):
   
    xvals = np.linspace(xmin,xmax,100)
    yvals = np.linspace(ymin,ymax,100)
    zvals = np.zeros((len(xvals),len(yvals)))
    for i in xrange(len(xvals)):
        for j in xrange(len(yvals)):
            zvals[i][j] = float(my_svm.predict(np.array([xvals[i],yvals[j]])))
    zvals = zvals.transpose()

    u, v = np.meshgrid( xvals, yvals )
    mycontour = plt.contour( xvals, yvals, zvals, [0])
    plt.title("Decision Boundary")


#'linear' kernel
linear_svm = svm.SVC(C=1, kernel='linear')
linear_svm.fit( X, y.flatten() )
plotData()
plotBoundary(linear_svm,0,4.5,1.5,5)


# C = 100
linear_svm = svm.SVC(C=100, kernel='linear')
linear_svm.fit( X, y.flatten() )
plotData()
plotBoundary(linear_svm,0,4.5,1.5,5)

# SVM with a custom kernel:
def gauss_Kernel(x1, x2, sigma):
    sigmasquared = np.power(sigma,2)
    return np.exp(-(x1-x2).T.dot(x1-x2)/(2*sigmasquared))


print gauss_Kernel(np.array([1, 2, 1]),np.array([0, 4, -1]), 2.)


datafile = 'ex6data2.mat'
data_mat = scipy.io.loadmat( datafile )
#Training set
X, y = data_mat['X'], data_mat['y']

#Divide the sample into two: ones with positive classification, one with null classification
X_pos = np.array([X[i] for i in xrange(X.shape[0]) if y[i] == 1])
X_neg = np.array([X[i] for i in xrange(X.shape[0]) if y[i] == 0])


plotData()


sigma = 0.1
gamma = np.power(sigma,-2.)
gaus_svm = svm.SVC(C=1, kernel='rbf', gamma=gamma)
gaus_svm.fit( X, y.flatten() )
plotData()
plotBoundary(gaus_svm,0,1,.4,1.0)

datafile = 'ex6data3.mat'
data_mat = scipy.io.loadmat( datafile )
#Training set
X, y = mat['X'], mat['y']
X_val, y_val = mat['Xval'], mat['yval']

#Divide the sample into two: ones with positive classification, one with null classification
X_pos = np.array([X[i] for i in xrange(X.shape[0]) if y[i] == 1])
X_neg = np.array([X[i] for i in xrange(X.shape[0]) if y[i] == 0])


plotData()


# Cvalue checker from Kaleko
Cvalues = (0.01, 0.03, 0.1, 0.3, 1., 3., 10., 30.)
sigmavalues = Cvalues
best_pair, best_score = (0, 0), 0

for Cvalue in Cvalues:
    for sigmavalue in sigmavalues:
        gamma = np.power(sigmavalue,-2.)
        gaus_svm = svm.SVC(C=Cvalue, kernel='rbf', gamma=gamma)
        gaus_svm.fit( X, y.flatten() )
        score = gaus_svm.score(X_val,y_val)
        #print this_score
        if score > best_score:
            best_score = score
            best_pair = (Cvalue, sigmavalue)
            
print "Best C, sigma pair is (%f, %f) with a score of %f."%(best_pair[0],best_pair[1],best_score)

gaus_svm = svm.SVC(C=best_pair[0], kernel='rbf', gamma = np.power(best_pair[1],-2.))
gaus_svm.fit( X, y.flatten() )
plotData()
plotBoundary(gaus_svm,-.5,.3,-.8,.6)
