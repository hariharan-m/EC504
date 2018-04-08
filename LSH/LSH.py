import numpy
import random
import re
import string
import sys,os,signal
from BitVector import *
#------------------------------LSH FUNCTION---------------------------------------
#We assume the raw data stored with name sample0_0 or sample1_8
#We store data with this name just for testing if the program works well
#This function returns the second integer of our sample name
def sample_index(sample_name):
    m = re.search(r'_(.+)$', sample_name)
    return int(m.group(1))

#This function returns the first integer of our sample name
def sample_group_index(sample_group_name):
    m = re.search(r'^.*(\d+)', sample_group_name)
    return int(m.group(1))

#This function returns the index of the hashband
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
#calculate the L2-norm of two lists, use it to reduce the number of clusters
#not use in mid-report yet
def l2norm(list1, list2):
    return numpy.linalg.norm(numpy.array(list1) - numpy.array(list2))

def Ctrl_c_handler( signum, frame ): os.kill(os.getpid(),signal.SIGKILL)
signal.signal(signal.SIGINT, Ctrl_c_handler)
#----------------------------------- LSH Class Definition ------------------------------------


class LocalitySensitiveHashing(object):
    def __init__(self, *args, **kwargs ):
        if kwargs and args:
            raise Exception('''please input the allowed_keys = 'datafile','dim','r','b','expected_num_of_clusters'''')
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
        #self.band_hash_mean_values = {}          # Store the mean of the bucket contents in band_hash dictionary
        #self.similarity_group_mean_values = {}
        self.coalesced_band_hash = {}            # Coalesce those keys of self.band_hash that have data samples in common
        #self.similarity_groups = []
        #self.coalescence_merged_similarity_groups = []  # Is a list of sets
        #self.l2norm_merged_similarity_groups = []  # Is a list of sets
        #self.merged_similarity_groups = None
        #self.pruned_similarity_groups = []
        #self.evaluation_classes = {}             # Used for evaluation of clustering quality if data in particular format


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
    #generate b*r hash functions and generate a dictionary to store hash values of our samples
        for x in range(self.how_many_hashes):
            hplane = numpy.random.uniform(low=-1.0, high=1.0, size=self.dim)
            hplane = hplane / numpy.linalg.norm(hplane)#standardlization hplane matrix
            self.hash_store[str(hplane)] = {'plus' : set(), 'minus' : set()}

    def hash_all_data(self):
        for hplane in self.hash_store:
            for sample in self._data_dict:
                # if python.version is 2, it needs translate into byte
                hplane_vals = hplane.translate(bytes.maketrans(b"][", b"  ")) \
                       if sys.version_info[0] == 3 else hplane.translate(string.maketrans("][","  "))
                bin_val = numpy.dot(list(map(convert, hplane_vals.split())), self._data_dict[sample])
                bin_val = 1 if bin_val>= 0 else -1      
                #if the dot vector is positive, move to plus list and vice versa
                if bin_val>= 0:
                    self.hash_store[hplane]['plus'].add(sample)
                else:
                    self.hash_store[hplane]['minus'].add(sample)


    def lsh_basic_for_nearest_neighbors(self):
    #build a cluster for each sample and store all similarity samples' name in each cluster
    #numbers of clusters are same as number of samples we have
        for (i,_) in enumerate(sorted(self.hash_store)):
            self.htable_rows[i] = BitVector(size = len(self._data_dict))
        for (i,hplane) in enumerate(sorted(self.hash_store)):#transfer the above list of dictionaries to a 2-d matrix
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
        for (k,sample) in enumerate(sorted(self._data_dict, key=lambda x: sample_index(x))):#go through every sample             
            for band_index in range(self.b):#go through every hash band
                bits_in_column_k = BitVector(bitlist = [self.htable_rows[i][k] for i in 
                                                     range(band_index*self.r, (band_index+1)*self.r)])
                key_index = "band" + str(band_index) + " " + str(bits_in_column_k)
                if key_index not in self.band_hash:
                    self.band_hash[key_index] = set()
                    self.band_hash[key_index].add(sample)#put all similar samples' name in current cluster
                else:
                    self.band_hash[key_index].add(sample)
        
        similarity_neighborhoods = {sample_name : set() for sample_name in 
                                         sorted(self._data_dict.keys(), key=lambda x: sample_index(x))}
        for key in sorted(self.band_hash, key=lambda x: band_hash_group_index(x)):        
            for sample_name in self.band_hash[key]:
                similarity_neighborhoods[sample_name].update( set(self.band_hash[key]) - set([sample_name]) )
        while True:
            sample_name = None
            if sys.version_info[0] == 3:#consider the difference between python3 and python2
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

    

