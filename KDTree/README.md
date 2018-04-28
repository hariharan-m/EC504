Animation Demo: http://christopherstoll.org/2011/09/k-d-tree-nearest-neighbor-search.html

##  KD Tree

### Dependencies

All of the package dependencies are available via pip. Python 3 is preffered.

Required python libraries.
* numpy

### Usage 
`kdtree = KdTree()`
`tree = kdtree.create(dataSet, depth)`
dataSet = list of list of points in the space

`kdtree.preOrder(tree)`
Prints the tree in preorder

`kdtree.search(tree, x)`
Searches the tree for x or NN to x

`kdtree.Ksearch(tree, x, n)`
Searches the tree for n NNs of x 

### Running the Code

python kdTree_v1.py  

Changes to name of dataset file and values can be passed in the main() function.

