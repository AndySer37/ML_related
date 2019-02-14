#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import argparse
from os import walk
from os.path import join
import shutil 
import csv

parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Training With Pytorch')
train_set = parser.add_mutually_exclusive_group()
parser.add_argument('--dataset', default='[0]', type=str, help='datalist')

args = parser.parse_args()

c = 0.
_sum = 0.
count = 0
pre = count

for i in range(len(args.dataset)):
	if args.dataset[i] != ' ':
		count += 1
	else:
		if pre == count:
			pre = count = count+1
		else:
			_sum += float(args.dataset[pre:count])
			c += 1
			count += 1
			pre = count
if args.dataset[-1] != ' ':
	_sum += float(args.dataset[pre:count])
	c += 1

print "\n\nAvg in class num {} = {} ".format(c, _sum/c)