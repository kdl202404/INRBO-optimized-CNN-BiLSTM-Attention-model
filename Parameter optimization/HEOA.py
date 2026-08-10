import numpy as np
import random
import math
import copy

''' 种群初始化函数 '''


def initial(pop, dim, ub, lb):
    X = np.zeros([pop, dim])
    for i in range(pop):
        for j in range(dim):
            X[i, j] = random.random() * (ub[j] - lb[j]) + lb[j]

    return X, lb, ub


'''边界检查函数'''


def BorderCheck(X, ub, lb, pop, dim):
    for i in range(pop):
        for j in range(dim):
            if X[i, j] > ub[j]:
                X[i, j] = ub[j]
            elif X[i, j] < lb[j]:
                X[i, j] = lb[j]
    return X


'''计算适应度函数'''


def CaculateFitness(X, fun):
    pop = X.shape[0]
    fitness = np.zeros([pop, 1])
    for i in range(pop):
        fitness[i] = fun(X[i, :])
    return fitness


'''适应度排序'''


def SortFitness(Fit):
    fitness = np.sort(Fit, axis=0)
    index = np.argsort(Fit, axis=0)
    return fitness, index


'''根据适应度对位置进行排序'''


def SortPosition(X, index):
    Xnew = np.zeros(X.shape)
    for i in range(X.shape[0]):
        Xnew[i, :] = X[index[i], :]
    return Xnew

''' Levy飞行'''


def Levy(d):
    beta = 3/2
    sigma = (math.gamma(1 + beta)*np.sin(math.pi*beta/2)) / \
        (math.gamma((1 + beta)/2)*beta*2**((beta-1)/2))**(1/beta)
    u = np.random.randn(1, d)*sigma
    v = np.random.randn(1, d)
    step = u/np.abs(v)**(1/beta)
    L = 0.05*step
    return L


'''人类进化优化算法2023'''


def HEOA(pop, dim, lb, ub, MaxIter, fun):

    X, lb, ub = initial(pop, dim, ub, lb)  # 初始化种群
    fitness = CaculateFitness(X, fun)  # 计算适应度值
    indexBest = np.argmin(fitness)
    GbestScore = copy.copy(fitness[indexBest])
    GbestPositon = np.zeros([1, dim])
    GbestPositon[0, :] = copy.copy(X[indexBest, :])
    Curve = np.zeros([MaxIter, 1])
    Xnew = copy.deepcopy(X)
    fitNew = copy.deepcopy(fitness)

    jump_factor = np.abs(lb[0] - ub[0]) / 1000
    A = 0.6  # Warning value
    LN = 0.4 # Percentage of leaders
    EN = 0.4 # Percentage of explorers
    FN = 0.1 # Percentage of followers
    LNNumber = np.int32(np.round(pop * LN)) # Number of leaders
    ENNumber = np.int32(np.round(pop * EN)) # Number of explorers
    FNNumber = np.int32(np.round(pop * FN)) # Number of followers
    pi = math.pi
    for t in range(MaxIter):
        print("第"+str(t)+"次迭代")
        R = np.random.random()
        if t<=MaxIter/4:
             for i in range(pop):
                 Xnew[i,:]=GbestPositon[0,:]*(1-t/MaxIter)+(np.mean(X[i,:])-GbestPositon[0,:])*np.floor(np.random.random()/jump_factor)*jump_factor + 0.2*(1-t/MaxIter)*(X[i,:]-GbestPositon[0,:])*Levy(dim)
        else:
            for i in range(LNNumber): #Leaders
                if R<A:
                    Xnew[i,:] = 0.2*np.cos(pi/2*(1-t/MaxIter))*X[i,:]*np.exp((-t*np.random.randn())/(np.random.random()*MaxIter))
                else:
                    Xnew[i,:] = 0.2*np.cos(pi/2*(1-t/MaxIter))*X[i,:] + np.random.randn()
                for j in range(dim):
                    if Xnew[i,j]>ub[j]:
                            Xnew[i,j]=ub[j]
                    if Xnew[i,j]<lb[j]:
                        Xnew[i,j]=lb[j]
            fitNew[i]=fun(Xnew[i,:])
            if fitNew[i]<fitness[i]:
                X[i,:]=copy.copy(Xnew[i,:])
                fitness[i]=copy.copy(fitNew[i])

            for i in range(LNNumber,LNNumber + ENNumber): # Explorers
                Xnew[i,:] = np.random.randn()*np.exp((X[-1,:]-X[i,:])/i**2)
                for j in range(dim):
                    if Xnew[i,j]>ub[j]:
                            Xnew[i,j]=ub[j]
                    if Xnew[i,j]<lb[j]:
                        Xnew[i,j]=lb[j]
                fitNew[i]=fun(Xnew[i,:])
                if fitNew[i]<fitness[i]:
                    X[i,:]=copy.copy(Xnew[i,:])
                    fitness[i]=copy.copy(fitNew[i])

            for i in range(LNNumber + ENNumber,LNNumber + ENNumber+FNNumber): #Followers
                Xnew[i,:] = X[i,:]+0.2*np.cos(pi/2*(1-(t/MaxIter)))*np.random.random([1,dim])*(X[0,:]-X[i,:])
                for j in range(dim):
                    if Xnew[i,j]>ub[j]:
                            Xnew[i,j]=ub[j]
                    if Xnew[i,j]<lb[j]:
                        Xnew[i,j]=lb[j]
                fitNew[i]=fun(Xnew[i,:])
                if fitNew[i]<fitness[i]:
                    X[i,:]=copy.copy(Xnew[i,:])
                    fitness[i]=copy.copy(fitNew[i])
            
            for i in range(LNNumber + ENNumber+FNNumber,pop): # Losers
                Xnew[i,:] = GbestPositon[0,:]+(GbestPositon[0,:]-X[i,:])*np.random.randn()
                for j in range(dim):
                    if Xnew[i,j]>ub[j]:
                            Xnew[i,j]=ub[j]
                    if Xnew[i,j]<lb[j]:
                        Xnew[i,j]=lb[j]
                fitNew[i]=fun(Xnew[i,:])
                if fitNew[i]<fitness[i]:
                    X[i,:]=copy.copy(Xnew[i,:])
                    fitness[i]=copy.copy(fitNew[i])

        indexBest = np.argmin(fitness)
        if fitness[indexBest] <= GbestScore:  # 更新全局最优
            GbestScore = copy.copy(fitness[indexBest])
            GbestPositon[0, :] = copy.copy(X[indexBest, :])
        fitness, sortIndex = SortFitness(fitness)  # 对适应度值排序
        X = SortPosition(X, sortIndex)  # 种群排序
        Curve[t] = GbestScore

    return GbestScore, GbestPositon, Curve
