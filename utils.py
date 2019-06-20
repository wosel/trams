#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from scipy import signal
from scipy.io import wavfile
import numpy as np
import os, sys, time
from glob import iglob
import pickle

NPERSEG = 1000

def get_spectrogram_sum(fpath):
    sample_rate, samples = wavfile.read(fpath)

    frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate, nperseg=NPERSEG)
    return prep_spectrogram(spectrogram)

def convert_dir_to_numpy(dir_id, dir_path):
    spec_sum_list = []
    lab_list = []
    
    for fpath in iglob(dir_path):
        spec_sum_list.append(get_spectrogram_sum(fpath))
    lab_list = [dir_id] * len(spec_sum_list)
    #print(dir_id)
    return np.array(spec_sum_list), np.array(lab_list)

def prep_spectrogram(spectrogram):
    return np.mean(spectrogram, axis=1)


# In[2]:


def prep_test_file(fpath, step, length):
    ret = {
        'data': [],
        'begin_times': []
    }
    
    sample_rate, samples = wavfile.read(fpath)
    pos = 0
    
    while pos < len(samples):
        cutout = samples[pos:pos+length]
        _, _, spectrogram = signal.spectrogram(cutout, sample_rate, nperseg=NPERSEG)
        
        prepped = prep_spectrogram(spectrogram)
        
        
        ret['data'].append(prepped)
        ret['begin_times'].append(pos / sample_rate)
        pos +=  step
    return ret

      


def postprocess_prediction(predictions, begin_times, id2label=None ,negative_class=8, join_thresh=5.1, min_occurences=3):
    
    if id2label is None:
        id2label = pickle.load(open('id2label.pkl', 'rb'))
        
    tram_cache = {}
    for key in id2label.keys():
        if key != negative_class:
            tram_cache[key] = {
                                'last_seen': -1. - join_thresh,
                                'occurence_counter': 0,
                                'current_begin_time': -1.
                              }
    occurence_list = []
    
    for output, begin_time in zip(predictions, begin_times):

        #reset all tram sightings if the trams haven't been picked up for too long
        #for those that were long enough to be counted, output them
        for cls in tram_cache.keys():
            tc = tram_cache[cls]
            
            #reset if gap betweeen detections too long
            if begin_time - tc['last_seen'] > join_thresh:
                
                # if good enough, output first
                if tc['occurence_counter'] >= min_occurences:
                    
                    occurence_list.append((cls, tc['current_begin_time']))
                
                #now reset
                tram_cache[cls]['occurence_counter'] = 0
        
        
        #continue existing streaks, add new ones
        
        #no tram = no streak
        if output == negative_class:
            continue
        
        #continued streak
        if begin_time - tram_cache[output]['last_seen'] <= join_thresh:
            tram_cache[output]['last_seen'] = begin_time
            tram_cache[output]['occurence_counter'] += 1
        
        #new streak
        else:
            #sanity check
            if tram_cache[output]['occurence_counter'] != 0:
                raise('Weird counting error - tram is indicating streak but time is past join threshold')
                
            tram_cache[output]['last_seen'] = begin_time
            tram_cache[output]['current_begin_time'] = begin_time
            tram_cache[output]['occurence_counter'] = 1

    #output all good enough streaks that haven't been outputed yet (too close to EOF)
    for cls in tram_cache.keys():
        tc = tram_cache[cls]
        if tc['occurence_counter'] >= min_occurences:

            occurence_list.append((cls, tc['current_begin_time']))
                
    return occurence_list




      
def print_to_csv(occurence_list, out_filename, id2label=None):
    header = 'accelerating_1_New,accelerating_2_CKD_Long,accelerating_3_CKD_Short,accelerating_4_Old,braking_1_New,braking_2_CKD_Long,braking_3_CKD_Short,braking_4_Old'
    label2pos_list = list(enumerate(header.split(',')))
    label2pos = {}
    for pos, label in label2pos_list:
        label2pos[label] = pos

    out_file = open(out_filename, 'w')
    out_file.write('seconds_offset,')
    out_file.write(header)
    out_file.write('\n')
    
    if id2label is None:
        id2label = pickle.load(open('id2label.pkl', 'rb'))
    
    
    for cls, time in sorted(occurence_list, key = lambda x: x[1]):
        out_arr = ['0'] * len(label2pos)
        out_arr[label2pos[id2label[cls]]] = '1'
        out_arr = [str(time)] + out_arr

        out_file.write(','.join(out_arr))
        out_file.write('\n')
        
def predict(prepped_file, model, out_filename, id2label=None ,negative_class=8, join_thresh=5.1, min_occurences=3):
    
    
    if id2label is None:
        id2label = pickle.load(open('id2label.pkl', 'rb'))
    
    header = 'accelerating_1_New,accelerating_2_CKD_Long,accelerating_3_CKD_Short,accelerating_4_Old,braking_1_New,braking_2_CKD_Long,braking_3_CKD_Short,braking_4_Old'
    label2pos_list = list(enumerate(header.split(',')))
    label2pos = {}
    for pos, label in label2pos_list:
        label2pos[label] = pos
        
    out_file = open(out_filename, 'w')
    out_file.write('seconds_offset,')
    out_file.write(header)
    out_file.write('\n')
    outputs = model.predict(prepped_file['data'])
    
    tram_cache = {}
    for key in id2label.keys():
        if key != negative_class:
            tram_cache[key] = {
                                'last_seen': -1. - join_thresh,
                                'occurence_counter': 0,
                                'current_begin_time': -1.
                              }
            
    # this loop combines multiple tram sightings in a row  
    occurence_list = []
    
    for output, begin_time in zip(outputs, prepped_file['begin_times']):
        
        
        #reset all tram sightings if the trams haven't been picked up for too long
        #for those that were long enough to be counted, output them
        for cls in tram_cache.keys():
            tc = tram_cache[cls]
            
            #reset if gap betweeen detections too long
            if begin_time - tc['last_seen'] > join_thresh:
                
                # if good enough, output first
                if tc['occurence_counter'] >= min_occurences:
                    
                    out_arr = ['0'] * len(label2pos)
                    out_arr[label2pos[id2label[cls]]] = '1'
                    out_arr = [str(tc['current_begin_time'])] + out_arr
                
                    out_file.write(','.join(out_arr))
                    out_file.write('\n')
                    occurence_list.append(cls, tc['current_begin_time'])
                
                #now reset
                tram_cache[cls]['occurence_counter'] = 0
        
        
        #continue existing streaks, add new ones
        
        #no tram = no streak
        if output == negative_class:
            continue
        
        #continued streak
        if begin_time - tram_cache[output]['last_seen'] <= join_thresh:
            tram_cache[output]['last_seen'] = begin_time
            tram_cache[output]['occurence_counter'] += 1
        
        #new streak
        else:
            #sanity check
            if tram_cache[output]['occurence_counter'] != 0:
                raise('Weird counting error - tram is indicating streak but time is past join threshold')
                
            tram_cache[output]['last_seen'] = begin_time
            tram_cache[output]['current_begin_time'] = begin_time
            tram_cache[output]['occurence_counter'] = 1

    #output all good enough streaks that haven't been outputed yet (too close to EOF)
    for cls in tram_cache.keys():
        tc = tram_cache[cls]
        if tc['occurence_counter'] >= min_occurences:

            out_arr = ['0'] * len(label2pos)
            out_arr[label2pos[id2label[cls]]] = '1'
            out_arr = [str(tc['current_begin_time'])] + out_arr

            out_file.write(','.join(out_arr))
            out_file.write('\n')
            occurence_list.append(cls, tc['current_begin_time'])
                
                
    return occurence_list
    out_file.close()

    


# In[ ]:




