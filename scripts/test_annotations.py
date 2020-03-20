#!/usr/bin/env python3

import sys
import os
import pickle

    
def load_annotations(filename):
    with open(filename, 'rb') as fid:
        items, labels = pickle.load(fid)
    return (items,labels)
    
test_file = "test_annot_keypoint.pkl"
(titems, tlabels) = load_annotations(test_file)
print(tlabels)