# -*- coding: utf-8 -*-
"""
Created on Sun May 30 20:36:26 2021

@author: BOUCHIHA
"""

import csv

# Corpus_size is the number of examples in the dataset and NBR is the number of example in a Class
Corpus_size = 2000
NBR = int(Corpus_size / 5)

NewF = "arabic_dataset_classifiction_" +str(Corpus_size)+".csv" 

c0 = 0
c1 = 0
c2 = 0
c3 = 0
c4 = 0

#from pprint import pprint

#import time import time

#start = time.time()

#texts = list()
#Y = list()

row_list = [['text', 'targe']]

with open('arabic_dataset_classifiction.csv', newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        if ((int(row['targe']) == 0) and (c0 < NBR)):
            row_list.append([row['text'], row['targe']])
            c0 = c0 + 1

        if ((int(row['targe']) == 1) and (c1 < NBR)):
            row_list.append([row['text'], row['targe']])
            c1 = c1 + 1

        if ((int(row['targe']) == 2) and (c2 < NBR)):
            row_list.append([row['text'], row['targe']])
            c2 = c2 + 1

        if ((int(row['targe']) == 3) and (c3 < NBR)):
            row_list.append([row['text'], row['targe']])
            c3 = c3 + 1

        if ((int(row['targe']) == 4) and (c4 < NBR)):
            row_list.append([row['text'], row['targe']])
            c4 = c4 + 1

        #Y.append(int(row['targe']))
        #line = row['text'].split()
        #texts.append(line)




with open(NewF, 'w', newline='') as file:
    writer = csv.writer(file, delimiter=',')
    writer.writerows(row_list)



#pprint(texts)
#pprint(Y)
#end = time.time()

#print("Read csv without chunks: ",(end-start),"sec")