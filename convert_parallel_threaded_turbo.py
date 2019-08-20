# py3 :23sec py2: 17
#from __future__ import print_function
import time
import threading
import os
import array as ar
import numpy as np
import sys
#from imageio import imread
#from scipy.misc import imread
#from scipy.linalg import svd
import cv2
from turbojpeg import TurboJPEG, TJPF_GRAY, TJSAMP_GRAY

t1=time.time()

#dirname = "testimg/"
dirname = "/nvme/bm/img/"

files = [filename for (dirpath, dirnames, filenames) in os.walk(dirname) for filename in filenames if filename[-4:] == ".jpg"]
files=sorted(files)

if len(files) == 0:
    print('ERROR goto folder im_align_celeba and inspect the README file\n')
    sys.exit(1)
subx, suby = 4, 4
dimx, dimy = (218+subx-1)//subx, (178+suby-1)//suby
print(dimx)
print(dimy)
data = np.zeros((len(files), dimx*dimy), dtype=np.float32)

def worker(num,num2):
    print( 'spawning: '+str(num)+' '+str(num2)+'\n')
    for i in range(num,num2+1):
      in_file=open(dirname+files[i], 'rb')
      gray=jpeg.decode(in_file.read(), pixel_format=TJPF_GRAY, scaling_factor=(1,4))
      data[i]=gray.ravel()
      in_file.close()
      #data[i]=np.mean(imread(dirname+files[i])[::4,::4,:],axis=2).ravel()
    return

jpeg = TurboJPEG('/usr/lib/x86_64-linux-gnu/libturbojpeg.so')

nr_threads=8
imgcount=len(files)
batches=imgcount//nr_threads
threads = []
idxstart=0
idxstop=batches-1
print( 'images: ' + str(imgcount) + ' in batches of size: ' + str(batches) + ' with ' + str(nr_threads) + ' threads')

for i in range(nr_threads):
    t = threading.Thread(target=worker, args=[idxstart,idxstop])
    threads.append(t)
    t.start()
    idxstart +=batches
    idxstop  +=batches

#for i in threads: # join now or after next (remainder)batch
#    i.join()

if(batches*nr_threads < imgcount):
     worker(batches*nr_threads,imgcount-1)


for i in threads:
    i.join()
  
with open("/nvme/bm/convert4.bin", "wb") as f:
	np.asarray(data, dtype=np.float32).tofile(f)
	#f.write(ar.array("f", data.ravel()))   

t2=time.time()-t1
print(str(t2) + ' seconds')

    
