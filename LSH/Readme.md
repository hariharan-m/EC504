##  Locality Sensitive Hashing Structure

### [Find Code here](https://github.com/hariharan-m/EC504/tree/master/LSH/final)

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
coalesced_similarity_groups = lsh.merge_similarity_groups_with_coalescence( similarity_groups )
similarity_group_mean_values = lsh.get_similarity_group_mean_values( coalesced_similarity_groups )
similar_set = lsh.search_for_similar_set(similarity_group_mean_values, new_data)
~~~~


### Example 

~~~~
1. Running the test code in the file
$ python LSH.py
  This is the test code which includes writeJson part and search part.
  The total running time includes store hash table and search in the hash table
  
2. Running the writeJson.py and search.py
$ python writeJson.py
  Store the clusters in a json file for further search
$ python search.py
  Search a new data vector's nearest nerghborhood
  (New data must have same dimenstion as the samples in the dataset

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
          
    lsh_basic_for_neighborhood_clusters():
          Merges the keys with the values to create neighborhood clusters in the hash table.
          These clusters are returned as a list of similarity groups, with each group being a set.
          
    merge_similarity_groups_with_coalescence(self, similarity_groups)
          Coalesce the clusters based on the basis of shared data samples
          
    get_similarity_group_mean_values(self, similarity_groups)
          Get mean values of each clusters
          
    search_for_similar_set(self, similarity_group_mean_values, new_data)
          Find the minimun distance between new_data vector and mean vector in each clusters


~~~~

