from LSH import *
lsh = LSH( datafile = "data_for_lsh.csv",  
                                dim = 10,
                                r = 50,                             
                                b = 100,

                              )

start = timeit.default_timer()

new_data = [-0.067, -0.015, 0.907, 0.034, -0.05, -0.017, -0.144, -0.204, -0.042, 0.013]

with open('output.json','r') as file:
	similarity_group_mean_values = json.load(file)

similar_set = lsh.search_for_similar_set(similarity_group_mean_values, new_data)
end = timeit.default_timer()

print('==============')
print(similar_set)
print('Running time is '+ str(end-start) + 's')

