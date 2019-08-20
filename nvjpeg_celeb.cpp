/*
g++ -O3 nvjpeg_celeb.cpp -fopenmp -lturbojpeg -lcudart -lnvjpeg -o nvjpeg_celeb -I/usr/local/cuda/include -Wl,-rpath=/usr/local/cuda/lib64 -L/usr/local/cuda/lib64 && time ./nvjpeg_celeb

7.5 sec REMOVE OUTPUTFILE BEFORE RUNNING else just 9sec for whatever reason

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            nvjpeg decodes jpg files/data to unsigned char only. NO RESCALING, etc
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
*/


#include <omp.h>
#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <sys/time.h> // timings
#include <string.h> // strcmpi
#include <dirent.h>  // linux dir traverse
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <turbojpeg.h>
#include "include/binary_IO.hpp"
#include "include/bitmap_IO.hpp"
#include "cuda_runtime.h"
#include <nvjpeg.h>

#define NTHREADS 8
#define BATCHSIZE 100 // >= 100 is 1 sec faster

double global_timer[NTHREADS*8]; //padding

#define nvjpegCheckError(call)                                   \
{                                                            \
    nvjpegStatus_t e = (call);                               \
    if (e != NVJPEG_STATUS_SUCCESS) {                        \
        std::cout << "nvjpeg failure: error #" << e  \
        << " "<< __FILE__ << ", line " << __LINE__ << std::endl;   \
    }                                                        \
}

#define cudaCheckError()                                                  \
{                                                                         \
    cudaError_t e = cudaGetLastError();                                   \
    if (e != cudaSuccess) {                                               \
        std::cout << "Cuda failure: '" << cudaGetErrorString(e)           \
        << " "<< __FILE__ << ", line " << __LINE__ << std::endl;                \
    }                                                                     \
}

#define print(x){ \
 std::cout << (x) << std::endl; \
}

int dev_malloc(void **p, size_t s) {
    return (int)cudaMalloc(p, s);
}

int dev_free(void *p) {
    return (int)cudaFree(p);
}


typedef std::vector<std::vector<char>> RawData; // read file contents bytewise into a RawData vector

// readInput: read "inputPath string" to filelist vector<string>
int readInput(const std::string &sInputPath, std::vector<std::string> &filelist) {
    int error_code = 1;
    struct stat s;

    if (stat(sInputPath.c_str(), &s) == 0) {
        if (s.st_mode & S_IFREG) {
            filelist.push_back(sInputPath);
        } else if (s.st_mode & S_IFDIR) {
            // processing each file in directory
            DIR *dir_handle;
            struct dirent *dir;
            dir_handle = opendir(sInputPath.c_str());
            if (dir_handle) {
                error_code = 0;
                while ((dir = readdir(dir_handle)) != NULL) {
                    if (dir->d_type == DT_REG) {
                        std::string sFileName = sInputPath + dir->d_name;
                        filelist.push_back(sFileName);
                    } else if (dir->d_type == DT_DIR) {
                        std::string sname = dir->d_name;
                        if (sname != "." && sname != "..") {
                            readInput(sInputPath + sname + "/", filelist);
                }}}
                closedir(dir_handle);
            } else {
                std::cout << "Cannot open input directory: " << sInputPath << std::endl;
                return error_code;
            }
        } else {
            std::cout << "Cannot open input: " << sInputPath << std::endl;
            return error_code;
        }
    } else {
        std::cout << "Cannot find input path " << sInputPath << std::endl;
        return error_code;
    }
    return 0;
}


void readFilesNV(int myTID, std::vector<std::string>& filenames, size_t start, size_t end,
                std::vector<std::vector<char>>& rawData, std::vector<size_t>& file_length,
                float* floatBig, nvjpegHandle_t& nvhandle, nvjpegJpegState_t&  nvjpeg_state,
                std::vector<nvjpegImage_t>& nvbuffer, cudaStream_t& stream){

    // load files to rawData-vector
    for(size_t i=start; i<end; ++i){
        std::ifstream input(filenames[i].c_str(), std::ios::in|std::ios::binary|std::ios::ate);
        if (!(input.good())){
            std::cerr << "error ifstreaming file " << std::endl;
        }
        std::streamsize file_size = input.tellg();
        input.seekg(0, std::ios::beg); 
        rawData[i].resize(file_size);
        if (!input.read(rawData[i].data(), file_size)){
            std::cerr << "Cannot read from file: "  << std::endl;
        }
        file_length[i] = file_size;
    }

/*
Benchmark notes: batched-3-phased with batchsize 2-4 is fastest with ~5sec decoding time
                 batched-single-phased with batchsize 100-200   with ~4sec (could be because of GPU Huffman decoding)
                 non-batched not tested
*/
// BEGIN: multiphased batchprocessing: WARNING CAN'T deal with last batch smaller than BATCHSIZE see in main (myTID==0)
/*
    double local_start = omp_get_wtime();
    // decode rawData to chars and convert them to floats
    int counter=0;
    for(size_t i=start; i<end; ++i){
        const unsigned char* jpg1 = reinterpret_cast<const unsigned char*>(rawData[i].data());
        nvjpegCheckError( nvjpegDecodeBatchedPhaseOne(
            nvhandle, nvjpeg_state, jpg1, (size_t)file_length[i], counter++, 0, stream) );
    }
    
    nvjpegCheckError( nvjpegDecodeBatchedPhaseTwo(nvhandle, nvjpeg_state, stream) );
    nvjpegCheckError( nvjpegDecodeBatchedPhaseThree(nvhandle, nvjpeg_state, nvbuffer.data(), stream) );
    cudaStreamSynchronize(stream);
    cudaCheckError();
*/
// END: multiphased batchprocessing

// BEGIN: hack for single-batch/ non-phased
 
    std::vector<const unsigned char *> raw_batch(0);
    for (int i=start; i < end; i++) {
        raw_batch.push_back((const unsigned char * const)rawData[i].data());
    }
    std::vector<size_t> file_length_batch(0);
    for(int i=start; i< end; i++){
        size_t t=file_length[i]; 
        file_length_batch.push_back(t);
    }


// single batch
    double local_start = omp_get_wtime();
    nvjpegCheckError( nvjpegDecodeBatched(nvhandle, nvjpeg_state, raw_batch.data(), file_length_batch.data(), nvbuffer.data(), stream   ) );

    cudaStreamSynchronize(stream);
    cudaCheckError();
 
// END: hack for single-batch/ non-phased


 /* copy nvbuffer from GPU back to CPU memory... */
    
    std::vector<unsigned char> oneDecodedPic(218*178);
    unsigned char *oneDecodedPicPTR = oneDecodedPic.data();
    for(int i=start; i<end; i++){
        
        //printf("i: %d; j: %d; start: %d; end: %d\n", i, j, start, end);
        unsigned char * gpu_buffer_ptr = nvbuffer[i%BATCHSIZE].channel[0];

        cudaMemcpy(oneDecodedPicPTR, gpu_buffer_ptr, (size_t)218*178, cudaMemcpyDeviceToHost);
        cudaCheckError();
        //for(int j=0; j<55*45; j+=4){
        //    floatBig[i*55*45+j] = (float) oneDecodedPicPTR[j*4];   
        //}
        
        float tmp[55*45];
        //std::cout << "pre run" << std::endl;
        for(int ii=0; ii<218; ii+=4){
            for(int jj=0; jj<178; jj+=4){
                 tmp[(ii/4)*45 + jj/4] = (float) oneDecodedPicPTR[ii*178+ jj];
                 
            }
        }
        //dump_bitmap(tmp,55,45,"outsmall/outsmall_"+std::to_string(i)+".bmp",0);
        //std::cout << "mid run" << std::endl;
        for(int j=0; j<55*45; j++){
            floatBig[i*55*45+j] = tmp[j] ;   
        }
        
        //std::cout << "post run" << std::endl;
    }
    cudaCheckError();
    //std::cout << "batch from: " << start << " till " << end << std::endl;
    dump_bitmap(floatBig,55,45,"test_small.bmp",0);
    //std::cout << "postpost run" << std::endl;
    


// single-batch end

 /* 
    int counter=0;
    for(int i=start; i<end; i++){
        std::vector<unsigned char> onePic(218*178);
        unsigned char *onePicPTR = onePic.data();
        unsigned char *gpu2 = nvbuffer[counter++].channel[0];
        //cudaMemcpy(onePicPTR, gpu2, 218*178, cudaMemcpyDeviceToHost);
        cudaCheckError();
        dump_bitmap(onePicPTR,218,178,"outsmall/output_test_notphased"+std::to_string(i)+".bmp",0);
    }
*/
    
    double local_end = omp_get_wtime();
    double local_dif = local_end-local_start;
    global_timer[myTID*8] += local_dif;
}

int main(){
    //double t1=omp_get_wtime()*1e-9; 
    //std::cout << t1 << std::endl;
    //sleep(1);
    //double t2=omp_get_wtime()*1e-9;
    //std::cout << t2 << " " << t2-t1 <<  std::endl;
    //exit(1);
    int blocksize1= BATCHSIZE; // =="batchsize": that many files at once per thread nvjpeg likes 2-4 turbojpeg a bit more
    int nthreads=NTHREADS;
    std::vector<std::string> filenames;
    std::string imagepath="/nvme/bm/img/";
    int r=readInput(imagepath, filenames);
    std::sort(filenames.begin(), filenames.end());
    omp_set_num_threads(nthreads);
    int total_images=filenames.size();
    RawData rawData(total_images);                  // store all jpegs in char-vector-vector
    std::vector<size_t> file_length(total_images);  // file_length of jpgs in rawData of same index 
    int nblocks1=total_images/nthreads;             // 202599/8=25324
    int nblocks2=nblocks1/blocksize1;               // 25324/487==52=4*13

    std::cout <<"Starting " << nthreads << " threads w blocksize1:" << blocksize1 << " nblocks1(total_mages/nthreads):" << nblocks1 << " nblocks2 (nblocks1/blocksize1):" << nblocks2 << std::endl; 
    
    float* floatBig=reinterpret_cast<float*>(malloc(4*55*45*202599));   // yuge array to save decoded jpgs to, 202599 rows
    
    cudaSetDevice(0);
    cudaDeviceReset();
    cudaCheckError();
    cudaFree(0);
    cudaCheckError();
#pragma omp parallel
{
    //tjhandle handle = tjInitDecompress();
    cudaStream_t stream;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    cudaCheckError();
    nvjpegHandle_t nvHandle;
    nvjpegJpegState_t  nvjpeg_state;
    //nvjpegCheckError( nvjpegCreateSimple(&nvHandle) );
    nvjpegDevAllocator_t dev_allocator = {&dev_malloc, &dev_free};
    //nvjpegBackend_t backend=NVJPEG_BACKEND_HYBRID;
    nvjpegBackend_t backend=NVJPEG_BACKEND_GPU_HYBRID; //NVJPEG_BACKEND_HYBRID;         //NVJPEG_BACKEND_GPU_HYBRID; slow if batchsize>=100
    //nvjpegCheckError( nvjpegCreate(backend, &dev_allocator, &nvHandle) );
    nvjpegCheckError( nvjpegCreateEx(backend, NULL, NULL, 0, &nvHandle) );
    nvjpegCheckError( nvjpegJpegStateCreate(nvHandle, &nvjpeg_state) );
    nvjpegCheckError( nvjpegDecodeBatchedInitialize(nvHandle, nvjpeg_state, blocksize1, 1, NVJPEG_OUTPUT_Y)  );
    size_t s=0;
    nvjpegCheckError( nvjpegGetDeviceMemoryPadding(&s, nvHandle ) );
    //std::cout << "DeviceMemoryPadding: " << s << std::endl;
    //nvjpegCheckError( nvjpegSetPinnedMemoryPadding((size_t)64, nvHandle ));
    nvjpegCheckError( nvjpegGetPinnedMemoryPadding(&s, nvHandle  ));
    //std::cout << "PinnedMemoryPadding: " << s << std::endl;
    //unsigned char* buffer = (unsigned char*)malloc(55*45*1);
    int myTID=omp_get_thread_num();
    int per_thread_start=myTID*nblocks2*blocksize1; // tid*52*487= tid*25324
   
    std::vector<nvjpegImage_t> nvbuffer(blocksize1);
    for (int i = 0; i < nvbuffer.size(); i++) {
        for (int c = 0; c < NVJPEG_MAX_COMPONENT; c++) {
            nvbuffer[i].channel[c] = NULL;
            nvbuffer[i].pitch[c] = 0;
        }
    }
    for(int i=0; i<nvbuffer.size(); i++){
        nvbuffer[i].pitch[0]=178;
        cudaMalloc(&nvbuffer[i].channel[0], 218*178);
    }
    cudaCheckError();
    
// @ LOOP
    for(int loop1=0; loop1<nblocks2; loop1++){              // 0 ... 51
        size_t index=per_thread_start + loop1*blocksize1; // + 0,487,2*487,...,51*487=24837
        // @ readfiles and decompress files
        
        readFilesNV(myTID, filenames, index, index+blocksize1, rawData, file_length, floatBig, nvHandle, nvjpeg_state, nvbuffer, stream);
    }

    std::cout << "global_timer["<<myTID<<"]=" << global_timer[myTID*8]<< std::endl;

    //std:: cout << "pre remaining call " << myTID << std::endl;
/* process remaining with NVJPEG comment out for phased or make new call with smaller nvbuffer (I guess) will segfault otherwise */
    if(myTID == 0){
        int done=nblocks2*blocksize1*nthreads; // 202592
        int rest=total_images-done; // 202599-52*487*8=7
        readFilesNV(myTID, filenames, done-1, total_images , rawData, file_length, floatBig, nvHandle, nvjpeg_state, nvbuffer, stream);
    }
    //std:: cout << "post remaining call " << myTID << std::endl;

    for(int i=0; i<nvbuffer.size(); i++){
        nvbuffer[i].pitch[0]=178;
        cudaFree(nvbuffer[i].channel[0]);
    }
    cudaCheckError();
    
    nvjpegCheckError( nvjpegJpegStateDestroy(nvjpeg_state) );
    nvjpegCheckError( nvjpegDestroy(nvHandle) ); 
    cudaStreamDestroy(stream);
    cudaCheckError();
    
}

    std::ofstream ofile_float("/nvme/bm/output_floatNVJPEG.bin", std::ios::binary);
 
   
    ofile_float.write((char*)floatBig, sizeof(float)*55*45*202599);
    ofile_float.close();
  

//for(int i=0; i<20; i++){
 //   dump_bitmap(floatBig+i*55*45, 55, 45, "outp/output_test"+std::to_string(i)+".bmp");
//}

// print file stats
    int sum=0,c=0,c2=0;
    for(auto v: file_length){
        if(v==0){
            c++; c2++;
            //std::cout << c2 << std::endl;
        } else { 
            c2++;}
        sum+=v;
    }
    std::cout << c2 << " Dateien; Sum: " << sum << " average: "<< sum/c2 << " bytes zerolength#: " << c << std::endl;

    free(floatBig);
    cudaDeviceReset();
    cudaCheckError();
}





