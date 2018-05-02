from LSH import *
lsh = LSH( datafile = "data_for_lsh.csv",  
                                dim = 10,
                                r = 50,                             
                                b = 100,

                              )
lsh.get_data_from_csv()
lsh.initialize_hash_store()
lsh.hash_all_data()
# lsh.display_contents_of_all_hash_bins_pre_lsh()
# similarity_neighborhoods = lsh.lsh_basic_for_nearest_neighbors('sample0_1')
# print(similarity_neighborhoods)
similarity_groups = lsh.lsh_basic_for_neighborhood_clusters()

coalesced_similarity_groups = lsh.merge_similarity_groups_with_coalescence( similarity_groups )
similarity_group_mean_values = lsh.merge_similarity_groups_with_l2norm_sample_based( coalesced_similarity_groups )

with open('output.json','w') as file:
    json.dump(similarity_group_mean_values, file)
