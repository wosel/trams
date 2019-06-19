#!/usr/bin/env python
# coding: utf-8

# In[1]:


import argparse
import pickle

from utils import prep_test_file, postprocess_prediction, print_to_csv


parser = argparse.ArgumentParser()
parser.add_argument('filepath')
parser.add_argument('-m', '--model', default='best_model.pkl')
parser.add_argument('-o', '--out_fpath', default='out.csv')

args = parser.parse_args()

model = pickle.load(open(args.model, 'rb'))
prepped_file = prep_test_file(args.filepath, step=22050, length=66150)
predictions = model.predict(prepped_file['data'])
occurence_list = postprocess_prediction(predictions, prepped_file['begin_times'])
print_to_csv(occurence_list, args.out_fpath)




# In[ ]:




