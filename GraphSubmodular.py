#Setup
import numpy as np
import matplotlib.pyplot as plt 
import scipy
import sympy as sym
from pylanczos import PyLanczos

import qiskit
from qiskit.quantum_info import Operator,Pauli,SparsePauliOp, Statevector

##GRAPH THEORETIC FUNCTIONS

def getEdges(A):
    '''
    Get all the edges from an adjaceny matrix as a list (of tuples)
    (np.array) A: Adjacency matrix of graph
    
    Returns:
    (List[(Int,Int)]): List of all edges as tuples
    '''
    D=[]
    for i in range(len(A)):
        for j in range(i+1,len(A)):
            if(A[i][j]==1):
                D.append((i,j))
    return D

def getMatrix(D,n):
    '''
    Get the adjaceny matrix from a list of edges (symmetric n x n matrix)
    (List[(Int,Int)]) D: List of all edges as tuples
    (Int) n: Number of verticies
    
    Returns:
    (np.array): Adjacency matrix of graph
    '''
    A=np.zeros((n,n),dtype='int')
    for i in D:
        A[i[0]][i[1]]=1
        A[i[1]][i[0]]=1
    return A 


def InducedSubgraph(A,l):
    '''
    Induced Subgraph from vertex set l
    (np.array) A: Adjacency matrix of initial graph
    (List[(Int,Int)]) l: List of all edges as tuples
    
    Returns:
    (np.array): Adjacency matrix of induced graph
    '''
    B = A.copy()
    for i in range(len(B)):
        if i not in l:
            for k in range(len(B)):
                B[i][k], B[k][i] = 0,0
    return B

def CostFunction(A,g):
    '''
    Cost Function (is submodular for g>=0 as desired)
    (np.array) A: Adjacency matrix of graph
    (Float) g: Magnetic parameter
    
    Returns:
    (Function): Cost function
    '''
    def Cost(l):
        return -(np.sum(InducedSubgraph(A,l))//2)+g*len(l)
    return Cost

 
def Greedy(f, w):
    '''
    Implents the Greedy - Wolfe Algorithm to minimize w^Tx for x in B_f
    (List[Float]) w: weight vector
    (Function) f: Submodular function
    
    Returns:
    (List[Float]): Minimal point
    '''
    z = list(zip(w,range(len(w))))
    z.sort()
    v = [i[1] for i in z]
    return VertexSet(f,v)


def Vertex(f,N):
    '''
    Obtains a vertex of B_f to start the Fujishige-Wolfe Algorithm (from the identity permutation)
    (Function) f: Submodular function
    (Int) N: Number of qubits
    
    Returns:
    (List[Float]): B_f vertex 
    '''
    return VertexSet(f,list(range(N)))

def VertexSet(f,l):
    '''
    Get a vertex from a permutation of the numbers
    (Function) f: Submodular function
    Float[Int]: Re-arrangement of vertex indicies
    
    Returns:
    (List[Float]): B_f vertex 
    '''
    x=[0]*len(l)
    for j in range(len(l)):
        x[l[j]]=(f(l[:j+1])-f(l[:j]))
    return x


#We handle the case for degenerate systems seperately (we generally don't see degeneracy though)
def AffineMinimizer(S):
    '''
    Affine Minimizer Function
    (List[List(Float)]): List of verticies to run the affine minimizer over
    
    Returns:
    (List[Float],List[Float]): Point in the affine hull of S and associated coefficients
    '''
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
    
     
def SubmodularOpt(f,N,tol):
    '''
    Fujishige-Wolfe Algorithm
    (Function) f: Submodular function
    (Int) N: number of verticies
    (Float) tol: Tolerance
    
    Returns:
    (List[Float]): Minimum norm point in B_f
    '''
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


def twoSegmentCheck(G):
    '''
    Use a LP to check for the desnset subgraph
    (np.array) G: Adjacency Matrix
    
    Returns:
    (Bool): Whether or not G is the densest subgraph
    '''
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

def indexedOperator(O,n,numQubits):
    '''
    Obtain operator O_n
    
    (String) O: Label for operator as a string
    (Int) n: Operator index
    (Int) numQubits: number of qubits
    
    Returns:
    (SparsePauliOp): Indexed operator O_n
    '''
    return SparsePauliOp('I'*(numQubits - n - 1) + O + 'I'*n)


def getIsingHamiltonian(A,g):
    '''
    Get the Hamiltonian for Graph A and transvesre field g
    
    (np.array) A: Adjacency matrix
    (Float) g: Magnetic parameter
    
    Returns:
    (SparsePauliOp): Specified Hamiltonian
    '''
    numQubits = len(A)
    O=0*SparsePauliOp('I'*numQubits)
    for i in range(numQubits):
        O+= -g*indexedOperator('X',i,numQubits)
        for j in [q for q in range(len(A)) if A[i][q]==1]:
            O+= - 0.5 * indexedOperator('Z',i,numQubits)@indexedOperator('Z',j,numQubits)
    return O.simplify()



def generateData(G,g_values):
    '''
    Generate the data for a graph and a set of g - values
    (np.array) G: Adjacency list
    (Iterable[Float]) g_values: Collection of g values
    
    Returns:
    (List[Float],List[Float],List[np.array]): List of computed approximations, List of exact eigenvalues, List of corresponding subgraphs (as adjacency matricies)
    '''
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
