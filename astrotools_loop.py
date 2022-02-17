import subprocess
import shutil, os
import sys

i = 0
# centroid location rounded x,y
x = int(sys.argv[3])
y = int(sys.argv[4])
for i in range(3):
  i1=int(i + x - 1)
  file = str(i1)
  for j in range(3):
    j1=int(j + y - 1)
    file1 = str(j1)
    symfile = ("python astrotoolsa.py 6 " + sys.argv[1] + " " + sys.argv[2] + " " + file + " " + file1)
#    symfile = ("python astrotoolsa.py " + file)
    print(symfile)
    os.system(symfile)