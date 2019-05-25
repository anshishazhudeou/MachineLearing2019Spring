# Generated with SMOP  0.41
from libsmop import *
# knn.m

    
@function
def knn(X=None,y=None,testX=None,testy=None,*args,**kwargs):
    varargin = knn.varargin
    nargin = knn.nargin

    n,d=size(X,nargout=2)
# knn.m:3
    m,d=size(testX,nargout=2)
# knn.m:4
    mistakes=concat([0,0,0,0,0])
# knn.m:6
    for z in arange(1,5).reshape(-1):
        k=dot((z - 1),2) + 1
# knn.m:8
        mistakes[z]=0
# knn.m:9
        for i in arange(1,10).reshape(-1):
            validationTestX=X(arange(dot(3000,(i - 1)) + 1,dot(3000,i)),arange())
# knn.m:11
            validationTestY=y(arange(dot(3000,(i - 1)) + 1,dot(3000,i)))
# knn.m:12
            validationTrainX=concat([[X(arange(1,dot(3000,(i - 1))),arange())],[X(arange(dot(3000,i),30000),arange())]])
# knn.m:14
            validationTrainY=concat([[y(arange(1,dot(3000,(i - 1))))],[y(arange(dot(3000,i),30000))]])
# knn.m:15
            # D(i,t) = distance between point i and t
        # Credit to : https://www.mathworks.com/matlabcentral/fileexchange/71-distance-m
        # for fastet matrix calculation
            a=validationTrainX.T
# knn.m:21
            b=validationTestX.T
# knn.m:22
            aa=sum(multiply(a,a),1)
# knn.m:23
            bb=sum(multiply(b,b),1)
# knn.m:23
            ab=dot(a.T,b)
# knn.m:23
            D=sqrt(abs(repmat(aa.T,concat([1,size(bb,2)])) + repmat(bb,concat([size(aa,2),1])) - dot(2,ab)))
# knn.m:24
            sorted,A=sort(D,nargout=2)
# knn.m:25
            q,w=size(validationTestX,nargout=2)
# knn.m:26
            topMatrix=A(arange(1,k),arange())
# knn.m:27
            for i in arange(1,k).reshape(-1):
                for r in arange(1,q).reshape(-1):
                    majority[i,r]=validationTrainY(A(i,r))
# knn.m:30
            result=mode(majority,1)
# knn.m:33
            for i in arange(1,q).reshape(-1):
                if (result(i) != validationTestY(i)):
                    mistakes[z]=mistakes(z) + 1
# knn.m:36
        mistakes(z)
        mistakes[z]=mistakes(z) / (dot(10,3000))
# knn.m:41
    
    figure
    num=concat([1,3,5,7,9])
# knn.m:45
    plot(num,mistakes,'-o')
    title('Knn effectiveness')
    xlabel('K')
    ylabel('Error percent')
    result=0
# knn.m:51
    # Choose K as the smallest error for validation set
    
    K=1
# knn.m:55
    # Use the L1 norm to calculate
    
    # D(i,t) = distance between point i and t
    
    # Credit to : https://www.mathworks.com/matlabcentral/fileexchange/71-distance-m
    
    # for fastet matrix calculation
    
    a=X.T
# knn.m:67
    b=testX.T
# knn.m:69
    aa=sum(multiply(a,a),1)
# knn.m:71
    bb=sum(multiply(b,b),1)
# knn.m:71
    ab=dot(a.T,b)
# knn.m:71
    D=sqrt(abs(repmat(aa.T,concat([1,size(bb,2)])) + repmat(bb,concat([size(aa,2),1])) - dot(2,ab)))
# knn.m:73
    # My version
    
    #for t=1:1
    
    #       D(:,t) =  sqrt(sum(bsxfun(@minus, X, testX(t,:)).^2, 2));
    
    #end
    
    sorted,A=sort(D,nargout=2)
# knn.m:85
    topMatrix=A(arange(1,K),arange())
# knn.m:87
    for i in arange(1,K).reshape(-1):
        for r in arange(1,m).reshape(-1):
            majority[i,r]=y(A(i,r))
# knn.m:93
    
    result=mode(majority,1)
# knn.m:99
    error=0
# knn.m:101
    for i in arange(1,m).reshape(-1):
        if (result(i) != testy(i)):
            error=error + 1
# knn.m:107
    
    
    error / m
    return result
    
if __name__ == '__main__':
    pass
    