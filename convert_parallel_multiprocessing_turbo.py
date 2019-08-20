#!/usr/bin/python
#using PyTurboJPEG
# py2: 8sec py3: 9

#from __future__ import print_function
import multiprocessing
import os
import ctypes
import array as ar
import numpy as np
import sys
from turbojpeg import TurboJPEG, TJPF_GRAY, TJSAMP_GRAY
import time

jpeg = TurboJPEG('/usr/lib/x86_64-linux-gnu/libturbojpeg.so')
t1=time.time()

dirname = "/nvme/bm/img/"
files = [filename for (dirpath, dirnames, filenames) in os.walk(dirname)
                  for filename in filenames if filename[-4:] == ".jpg"]
files=sorted(files)

if len(files) == 0:
    print('ERROR goto folder im_align_celeba and inspect the README file\n')
    sys.exit(1)

#@profile
def worker(num,num2,workerData):
    print( 'spawning: '+str(num)+' '+str(num2))
    for i in range(num,num2+1):
        in_file=open(dirname+files[i], 'rb')
        gray=jpeg.decode(in_file.read(), pixel_format=TJPF_GRAY, scaling_factor=(1,4))
        workerData[i]=gray.ravel()
        in_file.close()
    return

if __name__ == '__main__':
    shared_array_base = multiprocessing.Array(ctypes.c_float, 202599*55*45, lock=False) #ctypes.c_float 
    data = np.frombuffer(shared_array_base, dtype=np.float32)
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
    
    with open("/nvme/bm/convert_multiprocessing_turbo", "wb") as f:
       np.asarray(data, dtype=np.float32).tofile(f)

    t2=time.time()-t1
    print(str(t2) + ' seconds')

    