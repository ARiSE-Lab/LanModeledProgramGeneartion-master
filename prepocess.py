
def genearte_unchanged_change():
	f= open("files.txt", "r")
	lines = f.readlines()
	f.close()
	file_counts = {}
	for line in lines:
		if line != "\r\n":
			line = line.split('/')[-1].rstrip("\n")
			if line in file_counts:
				file_counts[line] +=1
			else:
				file_counts[line] = 1
	with open('unchanged_files.txt', 'w') as f, open('changed_files.txt', 'w') as c_f:
		c = 0
		for line in lines:
			if line != "\r\n":
				line2 = line.split('/')[-1].rstrip("\n")
				if file_counts[line2] == 1:
					f.write(line)
					c+=1
				else:
					c_f.write(line)
		print ' unchanged: '+ str(c)+ ' of: ', len(file_counts)

def genearte_variable_version_eliminatating_method_tokens(file, output_file):
	f = open(file, "r")
	lines = f.readlines()
	f.close()

	new_lines = list()
	for line in lines:
		if line != "\r\n":
			line = line.split(' ', 1)[1].rsplit(" ", 1)[0] #trim <method start> and <method end>
			new_lines.append(line[:-1])

	n_total = len(new_lines)

	train = open(output_file, "wb")
	for line in new_lines[:int(0.8*n_total)]:
		train.write(line+"\n")
	train.close()

# genearte_variable_version_eliminatating_method_tokens('unchanged_train.txt', 'unchanged_train.data')

'''for now we only consider the diff of last two snapshots, 
the changed lines of the last snapshots are considered to be test sentences. All other version of a method 
except the last snapshot version are considered as training'''

def change_unchanged_v2():
	snapshots_history = {}
	f= open("files.txt", "r")
	lines = f.readlines()
	f.close()
	file_counts = {}
	for line_full in lines:
		if line_full != "\r\n":
			line = line_full.split('/')[-1].rstrip("\n")
			if line in file_counts:
				file_counts[line] +=1
				snapshots_history[line].append(line_full.rstrip('\n'))
			else:
				file_counts[line] = 1
				snapshots_history[line] = []
				snapshots_history[line].append(line_full.rstrip('\n'))
	with open('unchanged_files.txt', 'w') as f, open('changed_files.txt', 'w') as c_f, open('all_train_file_list.txt', 'w') as t_f:
		c = 0
		for line in lines:
			if line != "\r\n":
				line2 = line.split('/')[-1].rstrip("\n")
				if file_counts[line2] == 1:
					f.write(line)
					t_f.write(line)
					c+=1
				else:
					c_f.write(line)
		print ' unchanged: '+ str(c)+ ' of: ', len(file_counts)
		
		old_chnaged_files = {}
		with open('old_changed_files.txt', 'w') as ocf:
			for ff, pp in  snapshots_history.items():
				if(len(pp)>1): 
					# print ff, ' paths: ', pp, "\n\n" # changed files have frequency > 1, taken care already
					old_chnaged_files[ff] = pp[:-1]
					# print len(old_chnaged_files[ff]), len(pp), " freq: ", file_counts[ff], "\n"
					for p in pp[:-1]:
						ocf.write(p+'\n')
						t_f.write(p+'\n')

		

			
def generate_train_data_from_all_but_last_snapshot():
	change_unchanged_v2()

generate_train_data_from_all_but_last_snapshot()
















'''


# f = open("all_method_body_new.txt", "r")
# lines = f.readlines()
# f.close()

# new_lines = list()
# for line in lines:
# 	if line != "\r\n":
# 		line = line.split(' ', 1)[1].rsplit(" ", 1)[0] #trim <method start> and <method end>
# 		new_lines.append(line[:-1])

# n_total = len(new_lines)

# train = open("train_ori.data", "wb")
# for line in new_lines[:int(0.8*n_total)]:
# 	train.write(line+"\n")
# train.close()

# val = open("val_ori.data", "wb")
# for line in new_lines[int(0.8*n_total):int(0.9*n_total)]:
# 	val.write(line+"\n")
# val.close()

# test = open("test_ori.data", 'wb')
# for line in new_lines[int(0.9*n_total):]:
# 	test.write(line+"\n")
# test.close()
'''