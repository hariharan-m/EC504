import numpy
import random
import re
import string
import sys,os,signal
from BitVector import *
import timeit
import csv
import json


# EC504 LSH for similar image search
# LSH structure
# Author: Xintong Hao, Yanling Li
# Reference: https://engineering.purdue.edu/kak/distLSH/LocalitySensitiveHashing-1.0.1.html

#------------------------------LSH FUNCTION---------------------------------------
# Returns the index of the hashband
def band_hash_group_index(block_name):
    firstitem = block_name.split()[0]
    m = re.search(r'(\d+)$', firstitem)
    return int(m.group(1))

# Convert int to float to match with sample features
def convert(value):
    try:
        answer = float(value)
        return answer
    except:
        return value
# Compute the L2-norm of two lists 
def l2norm(list1, list2):
    return numpy.linalg.norm(numpy.array(list1) - numpy.array(list2))
# Kill process
def Ctrl_c_handler( signum, frame ): os.kill(os.getpid(),signal.SIGKILL)
signal.signal(signal.SIGINT, Ctrl_c_handler)

#----------------------------------- LSH Class Definition ------------------------------------

class LSH(object):
    def __init__(self, *args, **kwargs ):
        if kwargs and args:
            raise Exception('Input can only be the keys: "datafile", "r", "b", "dim", "num_clusters".')
        keys = 'datafile','dim','r','b','num_clusters'
        input_keys = kwargs.keys()
        for key in input_keys:
            if key not in keys:
                raise SyntaxError(key + ': Wrong key used. Input can only be the keys: "datafile", "r", "b", "dim", "num_clusters".')
        datafile = dim = r = b = num_clusters = None
        if kwargs and not args:
            if 'datafile' in kwargs : datafile = kwargs.pop('datafile')
            if 'dim' in kwargs : dim = kwargs.pop('dim')
            if 'r' in kwargs : r = kwargs.pop('r')
            if 'b' in kwargs : b = kwargs.pop('b')
            if 'num_clusters' in kwargs : num_clusters = kwargs.pop('num_clusters')

        if datafile:
            self.datafile = datafile
        else:
            raise Exception('Please supply a datafile.')
        if dim:
            self.dim = dim
        else:
            raise Exception('Please supply a value of dimention')
        # Number of rows in each band (each row is for one hash func)
        self.r = r
        # Number of hash bands.
        self.b = b
        self.how_many_hashes = r * b 
        self.num_clusters = num_clusters
        # sample_name =>  vector_of_floats extracted from CSV stored here
        self._data_dict = {}
        self.how_many_data_samples = 0
        # hyperplane =>  {'plus' => set(), 'minus'=> set()}
        self.hash_store = {}
        self.htable_rows  = {}
        self.index_to_hplane_mapping = {}
        # BitVector column =>  bucket for samples
        self.band_hash = {}
        # Coalesce those keys of self.band_hash that have data samples in common
        self.coalesced_band_hash = {} 
        self.similarity_groups = []
        # Is a list of sets
        self.coalescence_merged_similarity_groups = [] 
        self.l2norm_merged_similarity_groups = [] 
        self.merged_similarity_groups = None

    # Get the data from csv file and store in the _data_dict dictionary
    def get_data_from_csv(self):
        if not self.datafile.endswith('.csv'):
            Exception('Datafile has to be csv file')
        data_dict = {}
        with open(self.datafile) as f:
            for i,line in enumerate(f):
                if line.startswith('#'): continue
                record = line
                parts = record.rstrip().split(r',')
                data_dict[parts[0].strip('"')] = list(map(lambda x: convert(x), parts[1:]))
                if i%10000 == 0:
                    print('.'),
                    sys.stdout.flush()
                sys.stdout = sys.__stdout__
            f.close()
        self.how_many_data_samples = i+1
        self._data_dict = data_dict

    # Generate b*r hash functions and generate a dictionary to store hash values of our samples
    def initialize_hash_store(self):
        for x in range(self.how_many_hashes):
            hplane = numpy.random.uniform(low=-1.0, high=1.0, size=self.dim)
            hplane = hplane / numpy.linalg.norm(hplane)
            self.hash_store[str(hplane)] = {'plus' : set(), 'minus' : set()}

    # Compute all the data in the dataset with r*b hash functions and streo in the hash_store with plus and minus bins
    def hash_all_data(self):
        for hplane in self.hash_store:
            for sample in self._data_dict:
                # if python.version is 2, it needs translate into byte
                hplane_vals = hplane.translate(bytes.maketrans(b"][", b"  ")) \
                       if sys.version_info[0] == 3 else hplane.translate(string.maketrans("][","  "))
                bin_val = numpy.dot(list(map(convert, hplane_vals.split())), self._data_dict[sample])
                bin_val = 1 if bin_val>= 0 else -1      
                if bin_val>= 0:
                    self.hash_store[hplane]['plus'].add(sample)
                else:
                    self.hash_store[hplane]['minus'].add(sample)

    # Generate a cluster for each sample and store all similarity samples' name in each cluster
    def lsh_basic_for_nearest_neighbors(self,in_sample_name):
        # in_sample_name is the input sample name
        for (i,_) in enumerate(sorted(self.hash_store)):
            self.htable_rows[i] = BitVector(size = len(self._data_dict))
        for (i,hplane) in enumerate(sorted(self.hash_store)):
            self.index_to_hplane_mapping[i] = hplane
            for (j,sample) in enumerate(self._data_dict):
            # for (j,sample) in enumerate(sorted(self._data_dict, key=lambda x: sample_index(x))):        
                if sample in self.hash_store[hplane]['plus']:
                    self.htable_rows[i][j] =  1
                elif sample in self.hash_store[hplane]['minus']:
                    self.htable_rows[i][j] =  0
                else:
                    raise Exception("An untenable condition encountered")

        for (k,sample) in enumerate(self._data_dict):
            for band_index in range(self.b):
                bits_in_column_k = BitVector(bitlist = [self.htable_rows[i][k] for i in 
                                                     range(band_index*self.r, (band_index+1)*self.r)])
                key_index = "band" + str(band_index) + " " + str(bits_in_column_k)
                if key_index not in self.band_hash:
                    self.band_hash[key_index] = set()
                    self.band_hash[key_index].add(sample)
                else:
                    self.band_hash[key_index].add(sample)
        
            similarity_neighborhoods = {sample_name : set() for sample_name in 
                                         self._data_dict.keys()}
        for key in sorted(self.band_hash, key=lambda x: band_hash_group_index(x)):        
            for sample_name in self.band_hash[key]:
                similarity_neighborhoods[sample_name].update( set(self.band_hash[key]) - set([sample_name]) )

        if in_sample_name in similarity_neighborhoods:
            return str(similarity_neighborhoods[in_sample_name]) 
        else:
            return None

    # Merges the keys with the values to create neighborhood clusters in the hash table.
    # These clusters are returned as a list of similarity groups, with each group being a set.
    def lsh_basic_for_neighborhood_clusters(self):
        for (i,_) in enumerate(sorted(self.hash_store)):
            self.htable_rows[i] = BitVector(size = len(self._data_dict))
        for (i,hplane) in enumerate(sorted(self.hash_store)):
            self.index_to_hplane_mapping[i] = hplane
            for (j,sample) in enumerate(self._data_dict):        
                if sample in self.hash_store[hplane]['plus']:
                    self.htable_rows[i][j] =  1
                elif sample in self.hash_store[hplane]['minus']:
                    self.htable_rows[i][j] =  0
                else:
                    raise Exception("An untenable condition encountered")
        for (k,sample) in enumerate(self._data_dict):
            for band_index in range(self.b):
                bits_in_column_k = BitVector(bitlist = [self.htable_rows[i][k] for i in 
                                                     range(band_index*self.r, (band_index+1)*self.r)])
                key_index = "band" + str(band_index) + " " + str(bits_in_column_k)
                if key_index not in self.band_hash:
                    self.band_hash[key_index] = set()
                    self.band_hash[key_index].add(sample)
                else:
                    self.band_hash[key_index].add(sample)

        similarity_neighborhoods = {sample_name : set() for sample_name in 
                                         self._data_dict.keys()}
        for key in sorted(self.band_hash, key=lambda x: band_hash_group_index(x)):        
            for sample_name in self.band_hash[key]:
                similarity_neighborhoods[sample_name].update( set(self.band_hash[key]) - set([sample_name]) )
        print("\n\nSimilarity neighborhoods calculated by the basic LSH algo:")
        for key in similarity_neighborhoods:
            simgroup = set(similarity_neighborhoods[key])
            simgroup.add(key)
            self.similarity_groups.append(simgroup)
        print( "\n\nSimilarity groups calculated by the basic LSH algo:\n" )
        for group in self.similarity_groups:
            print(str(group))
            print()
        print( "\nTotal number of similarity groups found by the basic LSH algo: %d" % len(self.similarity_groups) )
        return self.similarity_groups

    # Coalesce the clusters based on the basis of shared data samples
    def merge_similarity_groups_with_coalescence(self, similarity_groups):
        merged_similarity_groups = []
        for group in similarity_groups:
            if len(merged_similarity_groups) == 0:
                merged_similarity_groups.append(group)
            else:
                new_merged_similarity_groups = []
                merge_flag = 0
                for mgroup in merged_similarity_groups:
                    if len(set.intersection(group, mgroup)) > 0:
                        new_merged_similarity_groups.append(mgroup.union(group))
                        merge_flag = 1
                    else:
                       new_merged_similarity_groups.append(mgroup)
                if merge_flag == 0:
                    new_merged_similarity_groups.append(group)     
                merged_similarity_groups = list(map(set, new_merged_similarity_groups))
        for group in merged_similarity_groups:
            print( str(group) )
            print()
        print( "\n\nTotal number of MERGED similarity groups using coalescence: %d" % len(merged_similarity_groups) )
        self.coalescence_merged_similarity_groups = merged_similarity_groups
        return merged_similarity_groups

    # Get mean values of each clusters
    def get_similarity_group_mean_values(self, similarity_groups):
        similarity_group_mean_values = {}
        for group in similarity_groups:            #  A group is a set of sample names
            vector_list = [self._data_dict[sample_name] for sample_name in group]
            group_mean = [float(sum(col))/len(col) for col in zip(*vector_list)]
            similarity_group_mean_values[str(group)] = group_mean
        print('Write to json...')
        with open('output.json','w') as file:
            json.dump(similarity_group_mean_values, file)              
        return similarity_group_mean_values

    # Find the minimun distance between new_data vector and mean vector in each clusters
    def search_for_similar_set(self, similarity_group_mean_values, new_data):
        min_dist = 99999
        for mean_val in similarity_group_mean_values:
            dist = l2norm(similarity_group_mean_values[mean_val],new_data)

            if min_dist>dist:
                min_dist =  dist
                min_set = mean_val  

        return min_set

#------------------------------------  Test Code Follows  -------------------------------------

if __name__ == '__main__':
    lsh = LSH( datafile = "data_for_lsh.csv",  
                                    dim = 10,
                                    r = 50,                             
                                    b = 100,

                                  )
    lsh.get_data_from_csv()
    lsh.initialize_hash_store()
    lsh.hash_all_data()
    lsh.display_contents_of_all_hash_bins_pre_lsh()
    # similarity_neighborhoods = lsh.lsh_basic_for_nearest_neighbors('sample0_1')
    # print(similarity_neighborhoods)
    similarity_groups = lsh.lsh_basic_for_neighborhood_clusters()

    coalesced_similarity_groups = lsh.merge_similarity_groups_with_coalescence( similarity_groups )
    similarity_group_mean_values = lsh.get_similarity_group_mean_values( coalesced_similarity_groups )

    # with open('output.json','w') as file:
    #     json.dump(similarity_group_mean_values, file)

    '''
    1. Import new data and search for its nearest neighbours
        new_data : a list which has the same dimention
        similar_set : set of nearest neighbours 
    2. Compute the running time of search process
    '''
    start = timeit.default_timer()

    new_data = [-0.067, -0.015, 0.907, 0.034, -0.05, -0.017, -0.144, -0.204, -0.042, 0.013]
    similar_set = lsh.search_for_similar_set(similarity_group_mean_values, new_data)
    end = timeit.default_timer()

    print('==============')
    print(similar_set)
    print('Running time is '+ str(end-start) + 's')

    


