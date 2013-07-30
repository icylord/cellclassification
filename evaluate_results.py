#!/usr/bin/env python
"""This script evaluates the performance of a classification
   algorithm by comparing its output with the ground truth.
   See http://nerone.diiie.unisa.it/hep2contest for details.
"""

import sys
import os

CLASSES=['homogeneous', 'coarse_speckled', 'fine_speckled',
         'nucleolar', 'centromere', 'cytoplasmatic']

def read_csv(fname):
    f=open(fname)
    first=f.readline().strip().lower()
    flds=[x.strip() for x in first.split(';')]
    csv=[]
    for line in f:
       vals=[x.strip() for x in line.strip().lower().split(';')]
       if len(vals)==0:
           continue
       entry={ }
       for k, v in zip(flds,vals):
           entry[k]=v
       csv.append(entry)
    f.close()
    return csv 


def read_cls_output(fname):
    f=open(fname)
    result=[]
    keys=set()
    for line in f:
       vals=[x.strip() for x in line.strip().lower().split(';')]
       if len(vals)==0:
           continue
       img, cls = vals
       if not cls in CLASSES:
           print >> sys.stderr, 'Unvalid class:', cls
           continue
       k=int(os.path.splitext(os.path.split(img)[1])[0])
       if k in keys:
           print >> sys.stderr, 'Duplicate result for image:', k
           continue
       keys.add(k)
       result.append((k, cls))
    f.close()
    return result

def read_ground_truth(fname):
    csv=read_csv(fname)
    for x in csv:
        if not x['pattern'] in CLASSES:
            print >> sys.stderr, "Unvalid class in ground truth:", x['pattern']
    return [(int(x['id']), x['pattern']) for x in csv]

def evaluate_results(cls_output, ground_truth):
    gt={}
    for k, v in ground_truth:
        gt[k]=v
    found=set()
    correct=0
    for k, v in cls_output:
        if not k in gt:
            print >> sys.stderr, "Unknown image:", k
            continue
        found.add(k)
        if v==gt[k]:
            correct=correct+1
    missing=0
    for k in gt:
        if not k in found:
            missing=missing+1
    total=len(ground_truth)
    return 100.0*correct/total, 100.0*missing/total


def main():
    if len(sys.argv)!=3:
       print 'Usage: %s classifier_output ground_truth'
       sys.exit(1)
    cls_fname=sys.argv[1]
    base=os.path.split(cls_fname)[1]
    base=os.path.splitext(base)[0]
    gt_fname=sys.argv[2]

    cls_output=read_cls_output(cls_fname)
    ground_truth=read_ground_truth(gt_fname)
    correct, missing = evaluate_results(cls_output, ground_truth)
    print '%s: correct=%6.2f%%    missing=%6.2f%%' % (base, correct, missing)

if __name__=='__main__':
    main()
