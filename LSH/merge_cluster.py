import numpy
import random
import re
import string
import sys,os,signal
from BitVector import *

# EC504 LSH for similar image search
# LSH structure
# Author: Xintong Hao, Yanling Li
# Reference: https://engineering.purdue.edu/kak/distLSH/LocalitySensitiveHashing-1.0.1.html

#------------------------------LSH FUNCTION---------------------------------------
# We assume the raw data stored with name sample0_0 or sample1_8
# We store data with this name just for testing if the program works well
# This function returns the second integer of our sample name
debug=0
def sample_index(sample_name):
    m = re.search(r'_(.+)$', sample_name)
    return int(m.group(1))

# This function returns the first integer of our sample name
def sample_group_index(sample_group_name):
    m = re.search(r'^.*(\d+)', sample_group_name)
    return int(m.group(1))

#  This function returns the index of the hashband
def band_hash_group_index(block_name):
    firstitem = block_name.split()[0]
    m = re.search(r'(\d+)$', firstitem)
    return int(m.group(1))

#This function covert int to float to match with sample features
def convert(value):
    try:
        answer = float(value)
        return answer
    except:
        return value
# calculate the L2-norm of two lists, use it to reduce the number of clusters
# not use in mid-report yet
def l2norm(list1, list2):
    return numpy.linalg.norm(numpy.array(list1) - numpy.array(list2))

def Ctrl_c_handler( signum, frame ): os.kill(os.getpid(),signal.SIGKILL)
signal.signal(signal.SIGINT, Ctrl_c_handler)

#----------------------------------- LSH Class Definition ------------------------------------

class LSH(object):
    def __init__(self, *args, **kwargs ):
        if kwargs and args:
            raise Exception('Input can only be the keys: "datafile", "r", "b", "dim", "num_clusters","expected_num_of_clusters".')
        keys = 'datafile','dim','r','b','expected_num_of_clusters'
        input_keys = kwargs.keys()
        for key in input_keys:
            if key not in keys:
                raise SyntaxError(key + ': Wrong key used. Input can only be the keys: "datafile", "r", "b", "dim", "num_clusters","expected_num_of_clusters".')
        datafile = dim = r = b = num_clusters =expected_num_of_clusters= None
        if kwargs and not args:
            if 'datafile' in kwargs : datafile = kwargs.pop('datafile')
            if 'dim' in kwargs : dim = kwargs.pop('dim')
            if 'r' in kwargs : r = kwargs.pop('r')
            if 'b' in kwargs : b = kwargs.pop('b')
            if 'num_clusters' in kwargs : num_clusters = kwargs.pop('num_clusters')
            if 'expected_num_of_clusters' in kwargs  :  expected_num_of_clusters = kwargs.pop('expected_num_of_clusters')
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
        self.expected_num_of_clusters = expected_num_of_clusters
        self.num_clusters = num_clusters
        # sample_name =>  vector_of_floats extracted from CSV stored here
        self._data_dict = {}
        self.how_many_data_samples = 0
        # hyperplane =>  {'plus' => set(), 'minus'=> set()}
        self.hash_store = {}
        self.htable_rows  = {}
        self.similarity_groups = []
        self.index_to_hplane_mapping = {}
        # BitVector column =>  bucket for samples
        self.band_hash = {}
        # Coalesce those keys of self.band_hash that have data samples in common
        self.coalesced_band_hash = {} 

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

    def initialize_hash_store(self):
    # Generate b*r hash functions and generate a dictionary to store hash values of our samples
        for x in range(self.how_many_hashes):
            hplane = numpy.random.uniform(low=-1.0, high=1.0, size=self.dim)
            hplane = hplane / numpy.linalg.norm(hplane)
            self.hash_store[str(hplane)] = {'plus' : set(), 'minus' : set()}

    def hash_all_data(self):
        for hplane in self.hash_store:
            for sample in self._data_dict:
                # if python.version is 2, it needs translate into byte
                hplane_vals = hplane.translate(bytes.maketrans(b"][", b"  "))                        if sys.version_info[0] == 3 else hplane.translate(string.maketrans("][","  "))
                bin_val = numpy.dot(list(map(convert, hplane_vals.split())), self._data_dict[sample])
                bin_val = 1 if bin_val>= 0 else -1      
                if bin_val>= 0:
                    self.hash_store[hplane]['plus'].add(sample)
                else:
                    self.hash_store[hplane]['minus'].add(sample)

    def display_contents_of_all_hash_bins_pre_lsh(self):
        for hplane in self.hash_store:
            print( "\n\n hyperplane: %s" % str(hplane) )
            print( "\n samples in plus bin: %s" % str(self.hash_store[hplane]['plus']) )
            print( "\n samples in minus bin: %s" % str(self.hash_store[hplane]['minus']) )



    def lsh_basic_for_neighborhood_clusters(self):
        '''
        This method is a variation on the method lsh_basic_for_nearest_neighbors() in the following
        sense: Whereas the previous method outputs a hash table whose keys are the data sample names
        and whose values are the immediate neighbors of the key sample names, this method merges
        the keys with the values to create neighborhood clusters.  These clusters are returned as 
        a list of similarity groups, with each group being a set.
        '''
        for (i,_) in enumerate(sorted(self.hash_store)):
            self.htable_rows[i] = BitVector(size = len(self._data_dict))
        for (i,hplane) in enumerate(sorted(self.hash_store)):
            self.index_to_hplane_mapping[i] = hplane
            for (j,sample) in enumerate(sorted(self._data_dict, key=lambda x: sample_index(x))):        
                if sample in self.hash_store[hplane]['plus']:
                    self.htable_rows[i][j] =  1
                elif sample in self.hash_store[hplane]['minus']:
                    self.htable_rows[i][j] =  0
                else:
                    raise Exception("An untenable condition encountered")
        for (i,_) in enumerate(sorted(self.hash_store)):
            if i % self.r == 0: print
            print( str(self.htable_rows[i]) ) 
        for (k,sample) in enumerate(sorted(self._data_dict, key=lambda x: sample_index(x))):                
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
                                         sorted(self._data_dict.keys(), key=lambda x: sample_index(x))}
        for key in sorted(self.band_hash, key=lambda x: band_hash_group_index(x)):        
            for sample_name in self.band_hash[key]:
                similarity_neighborhoods[sample_name].update( set(self.band_hash[key]) - set([sample_name]) )
        print("\n\nSimilarity neighborhoods calculated by the basic LSH algo:")
        for key in sorted(similarity_neighborhoods, key=lambda x: sample_index(x)):
            print( "\n  %s   =>  %s" % (key, str(sorted(similarity_neighborhoods[key], key=lambda x: sample_index(x)))) )
            simgroup = set(similarity_neighborhoods[key])
            simgroup.add(key)
            self.similarity_groups.append(simgroup)
        print( "\n\nSimilarity groups calculated by the basic LSH algo:\n" )
        for group in self.similarity_groups:
            print(str(group))
            print()
        print( "\nTotal number of similarity groups found by the basic LSH algo: %d" % len(self.similarity_groups) )
        return self.similarity_groups

    def merge_similarity_groups_with_coalescence(self, similarity_groups):
        merged_similarity_groups = []
        similarity_groups=self.similarity_groups
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


    def merge_similarity_groups_with_l2norm_set_based(self, similarity_groups):
        similarity_groups=self.similarity_groups
        similarity_group_mean_values = {}
        for group in similarity_groups:                # A group is a set of sample names
            vector_list = [self._data_dict[sample_name] for sample_name in group]
            group_mean = [float(sum(col))/len(col) for col in zip(*vector_list)]
            similarity_group_mean_values[str(group)] = group_mean
        if len(similarity_groups) > self.expected_num_of_clusters:
            new_similarity_groups = []
            key_to_small_group_mapping = {}
            key_to_large_group_mapping = {}
            ordered_sim_groups_by_size = sorted(similarity_groups, key=lambda x: len(x), reverse=True)
            retained_similarity_groups = ordered_sim_groups_by_size[:self.expected_num_of_clusters]
            straggler_groups = ordered_sim_groups_by_size[self.expected_num_of_clusters :]
            print( "\n\nStraggler groups: %s" % str(straggler_groups) )
            print( "\n\nNumber of samples in retained groups: %d" % len(list(set.union(*retained_similarity_groups))) )
            print( "\n\nNumber of samples in straggler groups: %d" % len(list(set.union(*straggler_groups))) )
            retained_stringified_list = [str(item) for item in retained_similarity_groups]
            small_group_pool_for_a_given_large_group = {x : [] for x in retained_stringified_list}
            for group1 in straggler_groups:
                key_to_small_group_mapping[str(group1)] = group1
                dist_to_closest_large_group_mean, closest_large_group = None, None
                for group2 in retained_similarity_groups:
                    key_to_large_group_mapping[str(group2)] = group2
                    dist = l2norm(similarity_group_mean_values[str(group2)], similarity_group_mean_values[str(group1)])
                    if dist_to_closest_large_group_mean is None:
                        dist_to_closest_large_group_mean = dist
                        closest_large_group = group2
                    elif dist < dist_to_closest_large_group_mean:
                        dist_to_closest_large_group_mean = dist
                        closest_large_group = group2
                    else:
                        pass
                small_group_pool_for_a_given_large_group[str(closest_large_group)].append(group1)
            if any(len(small_group_pool_for_a_given_large_group[x]) > 0 for x in small_group_pool_for_a_given_large_group):
                print( "\n\nTHERE IS NON-ZERO POOL FOR MERGING FOR AT LEAST ONE LARGER SIMILARITY GROUPS" )
                print( str(small_group_pool_for_a_given_large_group.values()) )
            for key in small_group_pool_for_a_given_large_group:
                lgroup = key_to_large_group_mapping[key]
                list_fo_small_groups = small_group_pool_for_a_given_large_group[key]
                print( "\n\nFor group %s, the pool of small groups for merging =====>  %s" % 
                                                                          (str(lgroup), str(list_fo_small_groups)) )
            for group in sorted(retained_similarity_groups, key=lambda x: len(x), reverse=True):
                group_copy = set(group)     # shallow copy
                if len(small_group_pool_for_a_given_large_group[str(group)]) > 0:
                    for setitem in small_group_pool_for_a_given_large_group[str(group)]:
                        group_copy.update(setitem)  
                    new_similarity_groups.append(group_copy)
                else:
                    new_similarity_groups.append(group_copy)
            self.merged_similarity_groups_with_l2norm = new_similarity_groups
            print( "\n\nDisplaying set based l2 norm merged similarity groups:")
            for group in new_similarity_groups:
                print( str(group) )
            return new_similarity_groups
        else:
            print('''\n\nNo set based merging carried out since the number of clusters yielded by coalescence '''
                  '''is fewer than the expected number of clusters.''')
            return similarity_groups


#------------------------------------  Test Code Follows  -------------------------------------

if __name__ == '__main__':
    lsh = LSH(
                                    datafile = "data_for_lsh.csv",  
                                    dim = 100,
                                    r = 50,                             
                                    b = 100, 
                                    expected_num_of_clusters = 20,
                                  )
    lsh.get_data_from_csv()
    lsh.initialize_hash_store()
    lsh.hash_all_data()
    lsh.display_contents_of_all_hash_bins_pre_lsh()

    similarity_groups = lsh.lsh_basic_for_neighborhood_clusters()
    coalesced_similarity_groups = lsh.merge_similarity_groups_with_coalescence( similarity_groups )
    merged_similarity_groups = lsh.merge_similarity_groups_with_l2norm_set_based( coalesced_similarity_groups )
    while True:
        sample_name = None
        # For python3
        if sys.version_info[0] == 3:
            sample_name =  input('''\nEnter the symbolic name for a data sample '''
                                     '''(Enter quit to Exit): ''')
        # For python2
        else:
            sample_name = raw_input('''\nEnter the symbolic name for a data sample '''
                                        '''(Enter quit to Exit): ''')
        if sample_name=="quit":
            break
        else:
            count=1
            for each in merged_similarity_groups:
                if sample_name in each:
                    print( "\nthe sample is in cluster",count,":\n",each)
                else:
                    print( "\nThe name you entered doesn't exist in cluster",count)
                count+=1
