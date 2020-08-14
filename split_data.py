# -*- coding: UTF-8 -*-
import random
import os

def split_data(file_path, is_training=True):
	if is_training == False:
		return

	list_all = []
	map_all = {}
	file = open(os.path.join(file_path, "alldata.txt"),"r")
	bg = 0
	count = 0
	for line in file.readlines():
		count = count + 1
		print(count)
		if bg == 0:
			bg = bg + 1
			continue
		line = line.strip('\n')
		line_str = line.split('\t')
		if len(line_str) != 2:
			print("data error...continue")
			continue
		map_all[line_str[0]] = line_str[1]
		list_all.append(line)
	
	file.close()
	
	label_list = []
	map_all_data = {}
	key_l = list(map_all.keys())
	for i in range(len(key_l)):
		label_list.append(key_l[i])

	
	#popStr = list_all.pop(0)
	#print("pop  " + popStr)
	count = len(list_all)
	label_count = len(label_list)
	print("all data count ",count)
	print("all label count ",label_count)
	
	label_cur = 0
	for j in range(label_count):
		print(" j is %d , label_count is %d..."%(j,label_count))

		labelstr = label_list[j]
		list_d = []
		for i in range(count):
			str = list_all[i]
			line_str = str.split("\t")
			if labelstr == line_str[0]:
				list_d.append(line_str[1])
		
		map_all_data[labelstr] = list_d
	
	file_train = open(os.path.join(file_path, "train.tsv"),"w")
	file_dev = open(os.path.join(file_path, "dev.tsv"),"w")
	
	file_train.write("label\ttxt\n")
	file_dev.write("label\ttxt\n")
	
	train_list = []
	dev_list = []
	
	for label in map_all_data:
		label_data = map_all_data[label]
		data_count = len(label_data)
		#print("processing label %s, count %d\n"%(label,data_count))
		li = list(range(data_count))
		random.shuffle(li)
		count_cur = 0.0
		for i in li:
			#if i == 0:
				#train_list.append(label+"\t"+label+"\n")
			str = label_data[i]
			cur = float(count_cur / data_count)
			if data_count < 10:
				train_list.append(label+"\t"+str+"\n")
				dev_list.append(label+"\t"+str+"\n")
				continue
			
			if cur < 0.9 :
				train_list.append(label+"\t"+str+"\n")
			else:
				dev_list.append(label+"\t"+str+"\n")
			count_cur = count_cur + 1

	shuffle_data(train_list,file_train)
	shuffle_data(dev_list,file_dev)

	file_train.close()
	file_dev.close()

def shuffle_data (list_in,file_in):
	#shuffle
	list_len = len(list_in)
	list_li = list(range(list_len))
	random.shuffle(list_li)
	for i in list_li:
		file_in.write(list_in[i])

