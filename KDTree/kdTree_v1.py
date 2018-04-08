# --*-- coding:utf-8 --*--  
import numpy as np
import queue as Q

class Node:  
    def __init__(self, data, lchild = None, rchild = None):  
        self.data = data  
        self.lchild = lchild  
        self.rchild = rchild  
  
class KdTree:  
    def __init__(self):  
        self.kdTree = None  
  
    def create(self, dataSet, depth):   #create kd tree, return root node 
        if (len(dataSet) > 0):  
            m, n = np.shape(dataSet)    # get size of dataset  
            midIndex = int(m / 2) # get mid point  
            axis = depth % n    #judge which axis to seg plane;  
            # sortedDataSet = self.BubleSort(dataSet, axis) # Buble sort point along axis
            sortDataSet = dataSet[:]
            sortedDataSet = sorted(sortDataSet, key = lambda x: x[axis])
            print("the sort dataSet is" + str(sortedDataSet))
            node = Node(sortedDataSet[midIndex]) # create the node of mid point  
            # print sortedDataSet[midIndex]  
            leftDataSet = sortedDataSet[: midIndex]  
            rightDataSet = sortedDataSet[midIndex+1 :]  
            print("the left dataSet is" + str(leftDataSet))  
            print("the right dataSet is" + str(rightDataSet))  
            node.lchild = self.create(leftDataSet, depth+1)   
            node.rchild = self.create(rightDataSet, depth+1)  
            return node  
        else:  
            return None  
  
    def BubleSort(self, dataSet, axis):  # buble sort  
        sortDataSet = dataSet[:]    # copy a repica 
        m, n = np.shape(sortDataSet)  
        for i in range(m):  
            for j in range(0, m - i - 1):  
                if (sortDataSet[j][axis] > sortDataSet[j+1][axis]):  
                    temp = sortDataSet[j]  
                    sortDataSet[j] = sortDataSet[j+1]  
                    sortDataSet[j+1] = temp
        print("the sort dataSet is" + str(sortDataSet))  
        return sortDataSet

  
    def preOrder(self, node):  
        if node != None:  
            print("tttt->%s" % node.data)  
            self.preOrder(node.lchild)  
            self.preOrder(node.rchild)  

    def search(self, tree, x):  
        self.nearestPoint = None    # initial condition
        self.nearestValue = 0     
        def travel(node, depth = 0):    #recurse search  
            if node != None:    #base case  
                n = len(x)    
                axis = depth % n   
                if x[axis] < node.data[axis]:     
                    travel(node.lchild, depth+1)  
                else:  
                    travel(node.rchild, depth+1)  
  
     
                distNodeAndX = self.dist(x, node.data)   
                if (self.nearestPoint == None):  
                    self.nearestPoint = node.data  
                    self.nearestValue = distNodeAndX  
                elif (self.nearestValue > distNodeAndX):  
                    self.nearestPoint = node.data  
                    self.nearestValue = distNodeAndX  
  
                print(node.data, depth, self.nearestValue, node.data[axis], x[axis])  
                if (abs(x[axis] - node.data[axis]) <= self.nearestValue):  #find whether there is closer point by using radius   
                    if x[axis] < node.data[axis]:  
                        travel(node.rchild, depth+1)  
                    else:  
                        travel(node.lchild, depth + 1)  
        travel(tree)  
        return self.nearestPoint

    def Ksearch(self, tree, x, k):  # k nearst neighbors search
        que = Q.PriorityQueue()
        res = []

        def travel(node, depth = 0):    #recurse search  
            if node != None:    #base case
                distNodeAndX = self.dist(x, node.data)

                if (que.qsize() < k):
                    que.put((-distNodeAndX,node.data))
                else:
                    temp = que.get()
                    if (temp[0] < -distNodeAndX):
                        que.put((-distNodeAndX,node.data))
                    else:
                        que.put(temp)

                n = len(x)    
                axis = depth % n   
                if x[axis] < node.data[axis]:     
                    travel(node.lchild, depth+1)  
                else:  
                    travel(node.rchild, depth+1)  

                temp = que.get()
                que.put(temp)  
                if abs(x[axis] - node.data[axis]) <= abs(temp[0]) or que.qsize() < k:  #find whether there is closer point in another subbranch   
                    if x[axis] < node.data[axis]:  
                        travel(node.rchild, depth+1)  
                    else:  
                        travel(node.lchild, depth + 1)  
        if (k > 0):
            travel(tree)
        while not que.empty():
            res.append(que.get()[1])
        return res

  
    def dist(self, x1, x2): #calculate Euclidean distance  
        return ((np.array(x1) - np.array(x2)) ** 2).sum() ** 0.5  
    



# dataSet = [[2, 3, 7], [5, 4, 9], [9, 6, 5], [4, 7, 1], [8, 1, 2], [7, 2, 5]]  
# x = [5, 3, 2]
dataSet = [[2, 3], [5, 4], [9, 6], [4, 7], [8, 1], [7, 2]]  
x = [5, 3]    
kdtree = KdTree()  
tree = kdtree.create(dataSet, 0)  
kdtree.preOrder(tree)
print("The NN of " + str(x) + " is " + str(kdtree.search(tree, x)))
print("The 3NN of " + str(x) + " is " + str(kdtree.Ksearch(tree, x, 3)))
