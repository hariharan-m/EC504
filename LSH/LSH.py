
import numpy
import random
import re
import string
import sys,os,signal
from BitVector import *

#-----------------------------------  Utility Functions  ------------------------------------

def sample_index(sample_name):
    '''
    We assume that the raw data is stored in the following form:

       sample0_0,0.951,-0.134,-0.102,0.079,0.12,0.123,-0.03,-0.078,0.036,0.138
       sample0_1,1.041,0.057,0.095,0.026,-0.154,0.231,-0.074,0.005,0.055,0.14
       ...
       ...
       sample1_8,-0.153,1.083,0.041,0.086,-0.059,0.042,-0.172,0.014,-0.153,0.091
       sample1_9,0.051,1.122,-0.014,-0.117,0.015,-0.044,0.011,0.008,-0.121,-0.017
       ...
       ...

    This function returns the second integer in the name of each data record.
    It is useful for sorting the samples and for visualizing whether or not
    the final clustering step is working correctly.
    '''
    m = re.search(r'_(.+)$', sample_name)
    return int(m.group(1))

def sample_group_index(sample_group_name):
    '''
    As the comment block for the previous function explains, the data sample
    for LSH are supposed to have a symbolic name at the beginning of the 
    comma separated string.  These symbolic names look like 'sample0_0', 
    'sample3_4', etc., where the first element of the name, such as 'sample0',
    indicates the group affiliation of the sample.  The purpose of this
    function is to return just the integer part of the group name.
    '''
    m = re.search(r'^.*(\d+)', sample_group_name)
    return int(m.group(1))

def band_hash_group_index(block_name):
    '''
    The keys of the final output that is stored in the hash self.coalesced_band_hash
    are strings that look like:

         "block3 10110"

    This function returns the block index, which is the integer that follows the 
    word "block" in the first substring in the string that you see above.
    '''
    firstitem = block_name.split()[0]
    m = re.search(r'(\d+)$', firstitem)
    return int(m.group(1))


def convert(value):
    try:
        answer = float(value)
        return answer
    except:
        return value

def l2norm(list1, list2):
    return numpy.linalg.norm(numpy.array(list1) - numpy.array(list2))

def Ctrl_c_handler( signum, frame ): os.kill(os.getpid(),signal.SIGKILL)
signal.signal(signal.SIGINT, Ctrl_c_handler)
#----------------------------------- LSH Class Definition ------------------------------------


class LocalitySensitiveHashing(object):
    def __init__(self, *args, **kwargs ):
        if kwargs and args:
            raise Exception(  
                   '''LocalitySensitiveHashing constructor can only be called with keyword arguments for the 
                      following keywords: datafile,r,b,expected_num_of_clusters''') 
        allowed_keys = 'datafile','dim','r','b','expected_num_of_clusters'
        keywords_used = kwargs.keys()
        for keyword in keywords_used:
            if keyword not in allowed_keys:
                raise SyntaxError(keyword + ":  Wrong keyword used --- check spelling") 
        datafile=dim=r=b=expected_num_of_clusters=None
        if kwargs and not args:
            if 'datafile' in kwargs : datafile = kwargs.pop('datafile')
            if 'dim' in kwargs :  dim = kwargs.pop('dim')
            if 'r' in kwargs  :  r = kwargs.pop('r')
            if 'b' in kwargs  :  b = kwargs.pop('b')
            if 'expected_num_of_clusters' in kwargs  :  
                expected_num_of_clusters = kwargs.pop('expected_num_of_clusters')
        if datafile:
            self.datafile = datafile
        else:
            raise Exception("You must supply a datafile")
        self.expected_num_of_clusters = expected_num_of_clusters
        if dim:
            self.dim = dim
        else:
            raise Exception("You must supply a value for 'dim' which stand for data dimensionality")
        self.r = r                               # Number of rows in each band (each row is for one hash func)
        self.b = b                               # Number of bands.
        self.how_many_hashes =  r * b
        self._data_dict = {}                     # sample_name =>  vector_of_floats extracted from CSV stored here
        self.how_many_data_samples = 0
        self.hash_store = {}                     # hyperplane =>  {'plus' => set(), 'minus'=> set()}
        self.htable_rows  = {}
        self.index_to_hplane_mapping = {}
        self.band_hash = {}                      # BitVector column =>  bucket for samples  (for the AND action)
        self.band_hash_mean_values = {}          # Store the mean of the bucket contents in band_hash dictionary
        self.similarity_group_mean_values = {}
        self.coalesced_band_hash = {}            # Coalesce those keys of self.band_hash that have data samples in common
        self.similarity_groups = []
        self.coalescence_merged_similarity_groups = []  # Is a list of sets
        self.l2norm_merged_similarity_groups = []  # Is a list of sets
        self.merged_similarity_groups = None
        self.pruned_similarity_groups = []
        self.evaluation_classes = {}             # Used for evaluation of clustering quality if data in particular format


    def get_data_from_csv(self):
        if not self.datafile.endswith('.csv'): 
            Exception("Aborted. get_training_data_from_csv() is only for CSV files")
        data_dict = {}
        with open(self.datafile) as f:
            for i,line in enumerate(f):
                if line.startswith("#"): continue      
                record = line
                parts = record.rstrip().split(r',')
                data_dict[parts[0].strip('"')] = list(map(lambda x: convert(x), parts[1:]))
                if i%10000 == 0:
                    print('.'),
                    sys.stdout.flush()
                sys.stdout = sys.__stdout__
            f.close() 
        self.how_many_data_samples = i + 1
        self._data_dict = data_dict

    def show_data_for_lsh(self):
        print("\n\nData Samples:\n\n")
        for item in sorted(self._data_dict.items(), key = lambda x: sample_index(x[0]) ):
            print(item)

    def initialize_hash_store(self):
        for x in range(self.how_many_hashes):
            hplane = numpy.random.uniform(low=-1.0, high=1.0, size=self.dim)
            hplane = hplane / numpy.linalg.norm(hplane)
            self.hash_store[str(hplane)] = {'plus' : set(), 'minus' : set()}

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


    def lsh_basic_for_nearest_neighbors(self):
        '''
        Regarding this implementation of LSH, note that each row of self.htable_rows corresponds to 
        one hash function.  So if you have 3000 hash functions for 3000 different randomly chosen 
        orientations of a hyperplane passing through the origin of the vector space in which the
        numerical data is defined, this table has 3000 rows.  Each column of self.htable_rows is for
        one data sample in the vector space.  So if you have 80 samples, then the table has 80 columns.
        The output of this method consists of an interactive session in which the user is asked to
        enter the symbolic name of a data record in the dataset processed by the LSH algorithm. The
        method then returns the names (some if not all) of the nearest neighbors of that data point.
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
            if i % self.r == 0: print()
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
        while True:
            sample_name = None
            if sys.version_info[0] == 3:
                sample_name =  input('''\nEnter the symbolic name for a data sample '''
                                     '''(must match names used in your datafile): ''')
            else:
                sample_name = raw_input('''\nEnter the symbolic name for a data sample '''
                                        '''(must match names used in your datafile): ''')
            if sample_name in similarity_neighborhoods:
                print( "\nThe nearest neighbors of the sample: %s" % str(similarity_neighborhoods[sample_name]) )
            else:
                print( "\nThe name you entered does not match any names in the database.  Try again." )
        return similarity_neighborhoods

    def write_clusters_to_file(self, clusters, filename):
        FILEOUT = open(filename, 'w')
        for cluster in clusters:
            FILEOUT.write( str(cluster) + "\n\n" )
        FILEOUT.close()


    def display_contents_of_all_hash_bins_pre_lsh(self):
        for hplane in self.hash_store:
            print( "\n\n hyperplane: %s" % str(hplane) )
            print( "\n samples in plus bin: %s" % str(self.hash_store[hplane]['plus']) )
            print( "\n samples in minus bin: %s" % str(self.hash_store[hplane]['minus']) )


    def show_sample_to_initial_similarity_group_mapping(self):
        self.sample_to_similarity_group_mapping = {sample : [] for sample in self._data_dict}
        for sample in sorted(self._data_dict.keys(), key=lambda x: sample_index(x)):        
            for key in sorted(self.coalesced_band_hash, key=lambda x: band_hash_group_index(x)):            
                if (self.coalesced_band_hash[key] is not None) and (sample in self.coalesced_band_hash[key]):
                    self.sample_to_similarity_group_mapping[sample].append(key)
        print( "\n\nShowing sample to initial similarity group mappings:" )
        for sample in sorted(self.sample_to_similarity_group_mapping.keys(), key=lambda x: sample_index(x)):
            print( "\n %s     =>    %s" % (sample, str(self.sample_to_similarity_group_mapping[sample])) )


#------------------------------------  Test Code Follows  -------------------------------------

if __name__ == '__main__':
    lsh = LocalitySensitiveHashing( datafile = "data_for_lsh.csv",  
                                    dim = 10,
                                    r = 5,                              # number of rows in each band
                                    b = 20,                 # number of bands.   IMPORTANT: Total number of hash fns:  r * b
                                  )
    lsh.get_data_from_csv()
    lsh.show_data_for_lsh()
    lsh.initialize_hash_store()
    lsh.hash_all_data()
    lsh.display_contents_of_all_hash_bins_pre_lsh()
    similarity_neighborhoods = lsh.lsh_basic_for_nearest_neighbors()

    

