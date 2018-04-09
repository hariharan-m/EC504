##  Locality Sensitive Hashing Structure

### Dependencies

All of the package dependecies are available via pip. This module can be used in both Python2 and Python3.

Required python libraries:
* BitVector
* numpy
* random
* re
* string
* sys,os,signal



### Usage 
The overall functionality is exposed via:
~~~~
lsh = LSH( datafile= "data_for_lsh.csv", dim = 10, r = 50, b = 100)
lsh.get_data_from_csv()
lsh.initialize_hash_store()
lsh.hash_all_data()
similarity_neighborhoods = lsh.lsh_basic_for_nearest_neighbors()
~~~~


### Example 

~~~~
1. Running the test code in the file
$ python LSH.py
2. Running the Example.py
$ python Example.py

Example input Sample:
Enter the symbolic name for a data sample (must match names used in your datafile): sample0_1
~~~~


### Help Module Contents

~~~~
NAME
    LSH
Functions in Class LSH

    __init__(self, *args, **kwargs )
        Get the keywords and initialize the variables and dictionaries.
        
        Keyword Arguments:
            datafile: data_for_lsh.csv
            r: Number of rows in each band (each row is for one hash func)
            b: Number of hash bands.
            dim: dimension of the samples
    
    get_data_from_csv()
         Extracts the numerical data from your CSV file
  
    initialize_hash_store()
         Before hash_all_data(), we need initialization to generate the desired number of hyperplane orientations randomly.    
    
    hash_all_data()
          Hash all of data in the csv file with r*b number of hash functions.
    
    lsh_basic_for_nearest_neighbors()
          Implementation of the hyperplane based LSH algorithm to find nearest neighbors for data

~~~~

