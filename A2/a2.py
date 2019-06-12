import numpy as np
from numpy.linalg import inv
from scipy.stats import multivariate_normal
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from numpy.linalg import multi_dot
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
class GM:
    """Gaussian Mixture"""
    def __init__(self ):
        self.log = "Init."
        np.random.seed(0)
    
    def train(self,X,y):
        self.pi_0 = y[y==0].shape[0]/y.shape[0]
        self.n0 =  y[y==0].shape[0]
        self.n1 =  y[y==1].shape[0]
        self.pi_1 = 1.0 - self.pi_0

        self.mean_0 =np.average( X[y==0], axis=0)
        
        self.mean_1 =np.average( X[y==1], axis=0)
        self.s_0 =np.cov( X[y==0], rowvar=False)
        self.s_1 =np.cov( X[y==1], rowvar=False)
        self.sigma = self.pi_0*self.s_0 + self.pi_1*self.s_1
        self.Px_0 = multivariate_normal(self.mean_0,self.sigma)
        self.Px_1 = multivariate_normal(self.mean_1,self.sigma)
        #for idx, img in enumerate(X):
        #    print(idx)
        #    print(self.Px_0.pdf(img))
        #    print(self.Px_1.pdf(img))
    
    def predict(self,X):
        preds = []
        for idx, img in enumerate(X):
            if self.Px_0.pdf(img)>self.Px_1.pdf(img):
                preds.append(0)
            else:
                preds.append(1)
        
        return np.array(preds)

class LogisRegression:
    """Logistic Regression"""
    def __init__(self ):
        self.log = "Init."
        np.random.seed(0)
        

    def train(self,X,y,lb = 1, iter = 100):
        np.random.seed(0)
        self.lb = lb
        self.w = np.random.uniform(size=X.shape[1]+1)
        ones = np.ones((X.shape[0],1))
        augX = np.append(ones,X,axis=1)

        for i in range(0,iter):
            hess = self.Hess(augX,y)
            grad = self.Grad(augX,y)
            self.w = self.w -0.01*np.dot(inv(hess),grad)
            #print('At ' +str(i) + ' loss:'+str(self.nll(augX,y)))



    def Grad(self,X,y):
        score = np.dot(X,self.w)
        p = self.sigma(score)
        delta = p-y
        G = np.dot(X.T,delta) + 0.5*self.lb*self.w
        return G

    def Hess(self,X,y):
        score = np.dot(X,self.w)
        p = self.sigma(score)
        diags = np.multiply(p,1-p)
        R = np.diag(diags)
        H = multi_dot([X.T, R, X]) + 0.5*self.lb*np.identity(self.w.shape[0])
        return H



    
    def nll(self,X,y):
        score = np.dot(X,self.w)
        p = self.sigma(score)
        lw = y*np.log(p+ 1e-6) + (1-y)*np.log(1- p + 1e-6)
        return - np.sum(lw) + 0.5*self.lb*np.dot(self.w.T,self.w)

    def predict(self,X):
        ones = np.ones((X.shape[0],1))
        augX = np.append(ones,X,axis=1)

        score = np.dot(augX,self.w)
        p = self.sigma(score)
        p[p>=0.5] =1
        p[p<0.5] =0
        return p

    def separate(self, X):
        ones = np.ones((X.shape[0],1))
        augX = np.append(ones,X,axis=1)
        score = np.dot(augX,self.w)

        score[score>0] = 1
        score[score<0] = 0
        return score





    def sigma(self,x):
        return 1 / (1 + np.exp(-x))


def merging(Xcv,ycv):
    Xmerged = {}
    ymerged = {}
    for i in range(0,len(Xcv)):
        if i == 0:
            Xmerged = Xcv[0]
            ymerged = ycv[0]
        Xmerged = np.append(Xmerged,Xcv[i],axis=0)
        ymerged = np.append(ymerged,ycv[i],axis=0)

    return Xmerged, ymerged

def holdout(Xcv,ycv,k):
    Xmerged = {}
    ymerged = {}
    Xhold = {}
    yhold = {}
    count = 0
    for i in range(0,len(Xcv)):
        if i == k:
            Xhold  = Xcv[k]
            yhold =  ycv[k]
            continue
        if count==0:
            Xmerged = Xcv[i]
            ymerged = ycv[i]
            count = count +1
        else:
            
            Xmerged = np.append(Xmerged,Xcv[i],axis=0)
            ymerged = np.append(ymerged,ycv[i],axis=0)
            count = count +1

    return Xmerged, ymerged, Xhold, yhold
    

if __name__ == '__main__':
    Xcv = []
    ycv = []
    for i in range(1,11):
        Xtmp= np.loadtxt('trainData{}.csv'.format(i),delimiter=',')
        ytmp = np.loadtxt('trainLabels{}.csv'.format(i),delimiter=',')
        ytmp [ytmp==5] = 0 
        ytmp [ytmp==6] = 1
        Xcv.append(Xtmp)
        ycv.append(ytmp)



    Xtest= np.loadtxt('testData.csv',delimiter=',')
    ytest = np.loadtxt('testLabels.csv',delimiter=',')
    ytest [ytest==5] = 0 
    ytest [ytest==6] = 1
    #Xmerged = Xcv[0]
    #ymerged = ycv[0]
    #for i in range(1,10):
    #    Xmerged = np.append(Xmerged,Xcv[i],axis=0)
    #    ymerged = np.append(ymerged,ycv[i],axis=0)

    Xmerged, ymerged = merging(Xcv,ycv)

    X = Xcv[0]
    y = ycv[0]
    gm = GM()
    gm.train(Xmerged,ymerged )
    print('Pi_1 = {}'.format(gm.pi_0))
    print('Mean_1 = {}'.format(gm.mean_0))
    print('Mean_2 = {}'.format(gm.mean_1))
    print('Diagonals of Sigma = {}'.format(np.diag(gm.sigma)))
    print('Accuracy on test data of GM {}'.format(accuracy_score(ytest,gm.predict(Xtest))))

    #w = np.random.uniform(size=5)
    #someX = np.ones((100,5))

    #print(np.dot(someX, w))
    #ones = np.ones((X.shape[0],1))
    #augX = np.append(ones,X,axis=1)
    #print(augX)
    lbs = [0.1,0.5,1,3,5]

    for i in range(0,len(Xcv)):
        mm_scaler = preprocessing.MinMaxScaler()
        Xcv[i] = mm_scaler.fit_transform(Xcv[i])
    
    cv_scores = []
    for lb in lbs:
        scores = []
        for ind  in range(0,10):
            Xtrain, ytrain, Xhold, yhold = holdout(Xcv,ycv,ind)
            logG = LogisRegression()
            logG.train(Xtrain,ytrain,lb)
            scores.append(accuracy_score(yhold,logG.predict(Xhold)))
        
        cv_scores.append(np.average(scores))

    max_score = np.max(cv_scores)
    result = np.where(cv_scores == max_score)
    print(result)

    logGFinal = LogisRegression()
    logGFinal.train(Xmerged,ymerged,cv_scores[result[0][0]],500)
    print('Weigts: {}'.format(logGFinal.w))
    print('Accuracy on test data of LG {}'.format(accuracy_score(ytest,logGFinal.predict(Xtest))))
    
    plt.figure(1)
    plt.plot(lbs, cv_scores, 'b*-', label='Major')
    plt.xlabel("Lambda")
    plt.ylabel("Cross-validation score")
    plt.show()

    scores = logGFinal.separate(Xmerged)

    print(confusion_matrix(ymerged, scores))






    


    






   







