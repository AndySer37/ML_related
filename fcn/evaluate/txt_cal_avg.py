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
parser.add_argument('--file', default='[0]', type=str, help='datalist')
args = parser.parse_args()
_count = 0
_max_ep = 9999
_max_l = []
_max = 0
_ep_l = []
with open(args.file, 'r') as f:
	for line in f:
		count = 0
		pre = 0
		if 'epoch' in line:
			if sum(_ep_l) > _max :
				_max_ep = _count
				_max_l = _ep_l
				_max = sum(_max_l)
			_ep_l = []
			_count +=1
			count += 1
			_bool = False
		for i in range(len(line)):
			if _bool:
				if line[i] != ' ':
					count += 1
				else:
					if pre == count:
						pre = count = count+1
					else:
						
						#print(line[pre:count],float(line[pre:count]))
						_ep_l.append(float(line[pre:count]))
						count += 1
						pre = count	
			else:
				count += 1
				if line[i] == '[':
					_bool = True
					count -= 1
					pre = count
		if line[pre] != ']':
			#print(line[pre:-2],float(line[pre:-2]))
			#print(line[-2])
			if line[-2] == ']':
				_ep_l.append(float(line[pre:-2]))
			else:
				_ep_l.append(float(line[pre:-1]))
if sum(_ep_l) > _max :
	_max_ep = _count
	_max_l = _ep_l
	_max = sum(_max_l)


print "Max epoch : ", _max_ep
print "Max avg   : ", sum(_max_l) / len(_ep_l)
print "List      : \n\n", _max_l