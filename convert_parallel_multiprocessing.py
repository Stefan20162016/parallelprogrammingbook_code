#!/usr/bin/python
# py2: 46sec py3: 50sec

#from __future__ import print_function
import multiprocessing
import os
import ctypes
import array as ar
import numpy as np
import sys
from scipy.misc import imread
from scipy.linalg import svd
import time

t1=time.time()
dirname = "/nvme/bm/img/"
files = [filename for (dirpath, dirnames, filenames) in os.walk(dirname)
                  for filename in filenames if filename[-4:] == ".jpg"]
files=sorted(files)

if len(files) == 0:
    print('ERROR goto folder im_align_celeba and inspect the README file\n')
    sys.exit(1)


def worker(num,num2,workerData):
    print( 'spawning: '+str(num)+' '+str(num2)+'\n')
    for i in range(num,num2+1):
     workerData[i]=np.mean(imread(dirname+files[i])[::4,::4,:],axis=2).reshape(-1)
    return

if __name__ == '__main__':
    shared_array_base = multiprocessing.Array(ctypes.c_float, 202599*55*45, lock=False)
    data = np.frombuffer(shared_array_base, dtype=ctypes.c_float)
    data = data.reshape(202599,55*45)
    print('data.shape: ' +str(data.shape) + '\n')

    nr_threads=8
    imgcount=len(files)
    batches=imgcount//nr_threads
    processes = []
    idxstart=0
    idxstop=batches-1
    print( 'images: ' + str(imgcount) + ' in batches of size: ' + str(batches) + ' with ' + str(nr_threads) + ' threads')

    for i in range(nr_threads):
        p = multiprocessing.Process(target=worker, args=[idxstart,idxstop,data])
        processes.append(p)
        p.start()
        idxstart+=batches
        idxstop+=batches

    for i in processes:
        i.join()

    if(batches*nr_threads < imgcount):
         worker(batches*nr_threads,imgcount-1,data)
    
    with open("/nvme/bm/convert_multiprocessing", "wb") as f:   
       np.asarray(data, dtype=np.float32).tofile(f)

    t2=time.time()-t1
    print(str(t2) + ' seconds')


    