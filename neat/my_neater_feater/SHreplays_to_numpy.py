
import os
import pickle

pth = r"C:\Users\Matej\PycharmProjects\neat\convd_no_remake\convd\\"
files = os.listdir(pth)
count = 1000
ls = []
for f in files:
	if count == 0:
		print(".")
		count = 1000
	else:
		count -= 1
	txt = open(pth + f).read()
	lines = txt.split("\n")
	lineslist = []
	lastline = False
	for line in lines:
		if line == "":
			lastline = True
			continue
		if not lastline:
			lineslist.append(list(map(int, line.split(" "))))
		else:
			lastline = list(map(int, line.split(" ")))
			break

	ls.append([lineslist, lastline])
for i in ls:
	if i[0] == []:
		ls.remove(i)

local_dir = os.path.dirname(__file__)
lnoremake_path = os.path.join(local_dir, 'no_remake.pck')
print(ls[0])
with open(lnoremake_path, "wb") as f:
	pickle.dump(ls, f)
