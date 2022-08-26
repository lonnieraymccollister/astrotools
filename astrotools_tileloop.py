import subprocess
import shutil, os
import sys

i = 0
xa = 0
for k in range(int(int((int(sys.argv[3])/100)*2)-1)):
  ya = 0
  xa = xa + 50
  for k in range(int(int((int(sys.argv[4])/100)*2)-1)):
    ya = ya + 50
# centroid location rounded x,y
    x = int(xa)
    y = int(ya)
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