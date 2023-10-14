#Setup
import numpy as np
import matplotlib.pyplot as plt 
import scipy
import sympy as sym
from pylanczos import PyLanczos

import qiskit
from qiskit.quantum_info import Operator,Pauli,SparsePauliOp, Statevector

##GRAPH THEORETIC FUNCTIONS

#Get all the edges from an adjaceny matrix as a list (of tuples)
def getEdges(A):
    D=[]
    for i in range(len(A)):
        for j in range(i+1,len(A)):
            if(A[i][j]==1):
                D.append((i,j))
    return D

#Get the adjaceny matrix from a list of edges (symmetric |V| x |V| matrix)
def getMatrix(D,n):
    A=np.zeros((n,n),dtype='int')
    for i in D:
        A[i[0]][i[1]]=1
        A[i[1]][i[0]]=1
    return A 

#Induced Subgraph from vertex set l
def InducedSubgraph(A,l):
    B = A.copy()
    for i in range(len(B)):
        if i not in l:
            for k in range(len(B)):
                B[i][k], B[k][i] = 0,0
    return B

#Cost Function (is submodular for g>=0 as desired)
def CostFunction(A,g):
    def Cost(l):
        return -(np.sum(InducedSubgraph(A,l))//2)+g*len(l)
    return Cost

### Implents the Greedy - Wolfe Algorithm to minimize w^Tx for x in B_f
def Greedy(f, w):
    z = list(zip(w,range(len(w))))
    z.sort()
    v = [i[1] for i in z]
    return VertexSet(f,v)

###Obtains a vertex of B_f to start the Fujishige-Wolfe Algorithm (from the identity permutation)
def Vertex(f,N):
    return VertexSet(f,list(range(N)))

###Get a vertex from a permutation of the numbers
def VertexSet(f,l):
    x=[0]*len(l)
    for j in range(len(l)):
        x[l[j]]=(f(l[:j+1])-f(l[:j]))
    return x


###Affine Minimizer
#We handle the case for degenerate systems seperately (we generally don't see degeneracy though)
def AffineMinimizer(S):
    if(len(np.array(sym.Matrix(np.array(S).T).nullspace())) != 0):
        B=np.array(S).T
        v=np.array(sym.Matrix(B).nullspace(),dtype='float')[0]
        v=v/np.sum(v)
        return np.zeros(len(B)), np.reshape(v,len(v))
    else:
        B = np.array(S).T
        M = np.linalg.inv(B.T.dot(B))
        one = np.ones((len(S),1))
        alpha = M.dot(one)/(one.T.dot(M).dot(one))
        return np.reshape(B.dot(alpha),len(B.dot(alpha))),np.reshape(alpha,len(alpha))
    
    
#Fujishige-Wolfe Algorithm 
def SubmodularOpt(f,N,tol):
    q=np.array(Vertex(f,N))  ##Initialization
    x=q
    S=[x]
    l = [1]
    while(True):   #MAJOR cycle
        q  = np.array(Greedy(f,x))
        if(x.dot(x) <= (x.dot(q))+tol):
            break
        if(q.tolist() not in np.array(S).tolist()):
            S.append(q)
            l.append(0)
        while(True): #MINOR cycle
            y,alpha = AffineMinimizer(S)
            if np.all(alpha>=0):
                x=y
                l=list(alpha)
                break
            else:
                k = [i for i in range(len(alpha)) if alpha[i]<0]
                j = (min(k, key = lambda i: l[i]/(l[i]-alpha[i]))) 
                theta = l[j]/(l[j]-alpha[j])
                x=theta * y + (1-theta) * x
                for i in range(len(l)):
                    l[i]=theta*alpha[i]+(1-theta)*l[i]
                S = [v for i,v in enumerate(S) if l[i]>0]
                l = [v for v in l if v>0]
    return x

##Use a LP problem to check for the desnset subgraph
def twoSegmentCheck(G):
    Edge_List = getEdges(G)
    c = [1]*len(Edge_List) + [0]*len(G)

    M = np.zeros((2*len(Edge_List)+1,len(c)))
    for k,e in enumerate(Edge_List):
        i=e[0]
        j=e[1]
        M[2*k][k],M[2*k][len(Edge_List)+i]=1,-1
        M[2*k+1][k],M[2*k+1][len(Edge_List)+j]=1,-1
    M[len(M)-1]=[0]*len(Edge_List)+[1]*len(G)

    res=scipy.optimize.linprog(c=-np.array(c), A_ub=M, b_ub=[0]*(len(M)-1)+[1],A_eq=None, b_eq=None, bounds = [(0,1)]*len(c))

    X = np.zeros((len(Edge_List)))
    Y = res.x[len(Edge_List):]
    for i,e in enumerate(Edge_List):
        X[i]=min(Y[e[0]],Y[e[1]])

    S_opt = []
    d_opt = 0
    for r in set(Y):
        S = [i for i in range(len(Y)) if Y[i]>=r]
        E = [i for i in range(len(X)) if X[i]>=r]
        if(d_opt < len(E)/len(S)):
            S_opt=S
            d_opt = len(E)/len(S)
    return d_opt==len(Edge_List)/len(G)


### This section implements the specified Hamiltonian 

#Obtain operator O_n
def indexedOperator(O,n,numQubits):
    return SparsePauliOp('I'*(numQubits - n - 1) + O + 'I'*n)

#Get the Hamiltonian for Graph A and transvesre field g
def getIsingHamiltonian(A,g):
    numQubits = len(A)
    O=0*SparsePauliOp('I'*numQubits)
    for i in range(numQubits):
        O+= -g*indexedOperator('X',i,numQubits)
        for j in [q for q in range(len(A)) if A[i][q]==1]:
            O+= - 0.5 * indexedOperator('Z',i,numQubits)@indexedOperator('Z',j,numQubits)
    return O.simplify()


###Generate the data for a graph and a set of g - values
def generateData(G,g_values):
    
    two_segmented = twoSegmentCheck(G)
    if(not two_segmented):
        graph_data = []
        values = []
        for g in g_values:
            cost = CostFunction(G,g)
            x = SubmodularOpt(cost,len(G),1e-7)
            l=[]
            for i,v in enumerate(x):
                if(v<=0):
                    l.append(i)
            values.append(cost(l)-g*len(G))
            graph_data.append(InducedSubgraph(G,l))
    else:
        values = []
        graph_data=[]
        for g in (g_values):
            if(g< len(getEdges(G))/len(G)):
                values.append(-len(getEdges(G)))
                graph_data.append(G)
            else:
                values.append(-g * len(G))
                graph_data.append(np.zeros((len(G),len(G))))
    
    exact_expectations = []
    for i in range(len(g_values)):
        H=getIsingHamiltonian(G,g_values[i])
        H=(-1*H).to_matrix(sparse=True)
        engine = PyLanczos(H, True, 1)  # Find 2 maximum eigenpairs
        eigenvalues, eigenvectors = engine.run()
        exact_expectations.append(-eigenvalues[0].real)

    return values,exact_expectations, graph_data
