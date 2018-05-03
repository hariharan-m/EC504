from LSH import *
lsh = LSH( datafile = "data_for_lsh.csv",  
                                dim = 10,
                                r = 50,                             
                                b = 100,

                              )
lsh.get_data_from_csv()
lsh.initialize_hash_store()
lsh.hash_all_data()
similarity_groups = lsh.lsh_basic_for_neighborhood_clusters()

coalesced_similarity_groups = lsh.merge_similarity_groups_with_coalescence( similarity_groups )
similarity_group_mean_values = lsh.get_similarity_group_mean_values( coalesced_similarity_groups )

print(similarity_group_mean_values)

