import numpy
import random
import re
import string
import sys,os,signal
from BitVector import *
import timeit
import csv
import json

def l2norm(list1, list2):
    return numpy.linalg.norm(numpy.array(list1) - numpy.array(list2))

def search_for_similar_set(similarity_group_mean_values, new_data):
    min_dist = 99999
    for mean_val in similarity_group_mean_values:
        dist = l2norm(similarity_group_mean_values[mean_val],new_data)

        if min_dist>dist:
            min_dist =  dist
            min_set = mean_val  

    return min_set

start = timeit.default_timer()

new_data = [-0.067, -0.015, 0.907, 0.034, -0.05, -0.017, -0.144, -0.204, -0.042, 0.013]

with open('output.json','r') as file:
	similarity_group_mean_values = json.load(file)

similar_set = search_for_similar_set(similarity_group_mean_values, new_data)
end = timeit.default_timer()

print('==============')
print(similar_set)
print('Running time is '+ str(end-start) + 's')