# convert_images

fastest python script is multiprocessing and turbojpg 8 sec vs 3:30min for singleprocess-original

fastest C++ program is using OpenMP and Turbojpeg; NVJPEG not that fast, probably because of the small image sizes and some CPU work and/or huffman decoding
still could be usefull if you use the decoded images afterwards on the GPU (see nvidia's dali, and also with lot's of image modification option, NVJPEG just decodes)
