#!/usr/bin/env python

##  LSH_basic_for_demonstraing_nearest_neighbors.py

##  This script demonstrates the basic functioning of the LSH algorithm for finding
##  the nearest neighbors of given data elements.

##  Call syntax:
##
##         LSH_basic_for_demonstrating_nearest_neighbors.py
##
##  The last method called in this script, lsh_basic_for_nearest_neighbors(), places
##  the user in an interactive mode.  The script asks you to enter the name of a data
##  record from from the file that was processed by the LSH algorithm.  It
##  subsequently returns the nearest neighbors of that data record.


from LocalitySensitiveHashing import *

lsh = LocalitySensitiveHashing( 
           datafile = "data_for_lsh.csv",
           dim = 10,
           r = 50,                # number of rows in each band for r-wise AND in each band
           b = 100,               # number of bands for b-wise OR over all b bands
      )
lsh.get_data_from_csv()
lsh.show_data_for_lsh()
lsh.initialize_hash_store()
lsh.hash_all_data()
lsh.display_contents_of_all_hash_bins_pre_lsh()
similarity_neighborhoods = lsh.lsh_basic_for_nearest_neighbors()


