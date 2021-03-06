# py2: 3m30sec
# orig: 5m45sec
import os
import array as ar
import numpy as np
from scipy.misc import imread
from scipy.linalg import svd

# specifiy the CelebA folder
dirname = "/nvme/bm/img/"

files = [filename for (dirpath, dirnames, filenames) in os.walk(dirname)
                  for filename in filenames if filename[-4:] == ".jpg"]
files=sorted(files)
if len(files) == 0:
    print ("ERROR: goto folder im_align_celeba and inspect the README file")
    import sys
    sys.exit(1)

# if you want to subsample in index space
# files = files[::10]

# downsample the resolution by 4
subx, suby = 4, 4
dimx, dimy = (218+subx-1)//subx, (178+suby-1)//suby

data = np.zeros((len(files), dimx*dimy), dtype=np.float32)
print (dimx*dimy)

for index, filename in enumerate(files):
    if index % 1000 == 0:
        print (index)
    data[index] = np.mean(imread(dirname+filename)[::4,::4] , axis=2).ravel()

with open("first_step.bin", "wb") as f:
    np.asarray(data, dtype=np.float32).tofile(f)
    #f.write(ar.array("f", data.flatten()))
