import sys
sys.path.insert(0, '/mnt/home/project/csatsangi.chetan/projects/sahyadri-sandbox/scripts/post-process')
from readers import HaloReader

hr = HaloReader(sim_stem = 'sahyadri/default2048', real=1, snap=100)
halos = hr.read_this()
print("NUmber of halos:", halos.size)
print("Columns:", halos.dtype.names)
print("First mvir values:", halos['mvir'][:5])
