from LSH import *

lsh = LSH( 
           datafile = "xyz_1000_new.csv",
           dim = 50,
           r = 20,                
           b = 10,               
      )
lsh.get_data_from_csv()
lsh.initialize_hash_store()
lsh.hash_all_data()
lsh.display_contents_of_all_hash_bins_pre_lsh()
similarity_neighborhoods = lsh.lsh_basic_for_nearest_neighbors()


