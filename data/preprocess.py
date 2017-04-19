f = open("parsed_methods.txt", "r")
lines = f.readlines()
f.close()

new_lines = list()
for line in lines:
	if line != "\r\n":
		new_lines.append(line[:-1])

n_total = len(new_lines)

train = open("train.data", "wb")
for line in new_lines[:int(0.8*n_total)]:
	train.write(line+"\n")
train.close()

val = open("val.data", "wb")
for line in new_lines[int(0.8*n_total):int(0.9*n_total)]:
	val.write(line+"\n")
val.close()

test = open("test.data", 'wb')
for line in new_lines[int(0.9*n_total):]:
	test.write(line+"\n")
test.close()