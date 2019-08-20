/*

g++  test1.cpp -fopenmp -lturbojpeg -o test1_gcc && time ./test1_gcc
icc test1.cpp -g  -xHost -qopenmp -lturbojpeg -qopt-report=5 -qopt-report-phase=vec -o test1_icc && time ./test1_icc

both 4.7sec

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

#define print(x){ \
 std::cout << (x) << std::endl; \
}

typedef std::vector<std::vector<char>> RawData; // read file contents bytewise into RawData vector

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

// readFiles: read (end-start)-many files and convert jpeg to downsized grayscale image && convert pixels from char to float
// use vector of filename-strings to read files and save contents bytewise to "vector[numOfFiles] of vector<char>" and also save corresponding filelength
void readFiles(std::vector<std::string>& filenames, size_t start, size_t end, RawData& rawData, std::vector<size_t>& file_length, float* floatBig, tjhandle& handle, unsigned char* buffer){
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
    //tjhandle handle = tjInitDecompress();
    //unsigned char* buffer = (unsigned char*)malloc(55*45*1);
    //unsigned char* buffer;
    //std::vector<float> tmpFloat(55*45);
    for(size_t i=start; i<end; ++i){
        unsigned char* jpg1 = reinterpret_cast<unsigned char*>(rawData[i].data());
        //buffer = grayImages+i*55*45;
        if(tjDecompress2(handle, jpg1, (unsigned long)file_length[i], buffer, 45, 0, 55, TJPF_GRAY,0)){
            //TJFLAG_FASTDCT)){//|TJFLAG_FASTUPSAMPLE)){ TJFLAG_FASTUPSAMPLE|TJFLAG_FASTDCT
            std::cout << "error in tjDecomp2" << std::endl;
        }

 #pragma ivdep       
        for(int j=0; j<55*45; j++){
            floatBig[i*55*45+j]=(float)buffer[j];
        }
/*
	for(int jj=0; jj<55*45; jj+= 8){
#pragma ivdep 
		for(int j=jj; j<jj+8; j++){
            
            if(j<55*45){
		        floatBig[i*55*45+j] = (float)buffer[j];
            }
		}
        }
*/
        
    }

    //tjDestroy(handle);
}
int main(){
    //std::cout.setf(std::ios::unitbuf); 
    int blocksize1=487; //487 how many files at once per thread
    int nthreads=8;
    std::vector<std::string> filenames;
    std::string imagepath="/nvme/bm/img/";
    //std::string imagepath="/home/bm/Downloads/img_align_celeba/";
    int r=readInput(imagepath, filenames);
    std::sort(filenames.begin(), filenames.end());
    omp_set_num_threads(nthreads);
    int total_images=filenames.size();
    RawData rawData(total_images);
    std::vector<size_t> file_length(total_images);
    //print(total_images)    
    int nblocks1=total_images/nthreads;   // 202599/8=25324
    int nblocks2=nblocks1/blocksize1;         // 25324/487=52

    std::cout <<"Starting " << nthreads << " threads w blocksize1:" << blocksize1 << " nblocks1(total_mages/nthreads):" << nblocks1 << " nblocks2 (nblocks1/blocksize1):" << nblocks2 << std::endl; 
    
    float* floatBig=reinterpret_cast<float*>(malloc(4*55*45*202599));
    std::vector<int> x;
    
#pragma omp parallel
{
    tjhandle handle = tjInitDecompress();
    unsigned char* buffer = (unsigned char*)malloc(55*45*1);
    //unsigned char* buffer = (unsigned char*)_mm_malloc(55*45*1,32);
    int myTID=omp_get_thread_num();
    int per_thread_start=myTID*nblocks2*blocksize1; // tid*52*487= tid*25324
    for(int loop1=0; loop1<nblocks2; loop1++){              // 0 ... 51
        size_t index=per_thread_start + loop1*blocksize1; // + 0,487,2*487,...,51*487=24837
        // @ readfiles and decompress files
        readFiles(filenames, index, index+blocksize1, rawData, file_length, floatBig, handle, buffer); // blocksize1 chunks: [0..25323]
    }
    tjDestroy(handle);
    free(buffer);
}
// process remaining files:
    int done=nblocks2*blocksize1*nthreads; // 202592
    int rest=total_images-done; // 202599-52*487*8=7
    tjhandle handle = tjInitDecompress();
    unsigned char* buffer = (unsigned char*)malloc(55*45*1);
    readFiles(filenames, done-1, total_images, rawData, file_length, floatBig, handle,buffer);
    tjDestroy(handle);
    free(buffer);

    std::ofstream ofile_float("/nvme/bm/output_float.bin", std::ios::binary);
 

/// DO NOT WRITE FOR BENCHMARK !!    
    ofile_float.write((char*)floatBig, sizeof(float)*55*45*202599);
    ofile_float.close();
    
//for(int i=0; i<20; i++){
 //   dump_bitmap(floatBig+i*55*45, 55, 45, "outp/output_test"+std::to_string(i)+".bmp");
//}


    int sum=0,c=0,c2=0;
    for(auto v: file_length){
        if(v==0){
            c++; c2++;
            std::cout << c2 << std::endl;
        } else { c2++;}

        sum+=v;
        }

    std::cout << "Summe Dateilaengen: " << sum << " average: "<< sum/file_length.size() << "bytes zerolength#: " << c << std::endl;

    

/*
    tjhandle handle = tjInitDecompress();
    int width=0, height=0, jpgSub=0, jpgCol=0;
    unsigned char* buffer = (unsigned char*)malloc(218*178*1);
    for(int i=0; i<202592; i++){
        unsigned char* jpg1 = reinterpret_cast<unsigned char*>(rawData[i].data());
        //int ret=tjDecompressHeader3(handle, jpg1, (unsigned long)file_length[i], &width, &height, &jpgSub, &jpgCol);
        //std::cout << "return: " << ret << " "<<width<<height<<jpgSub<<jpgCol<< std::endl;
        width=45; height=55;
        if(tjDecompress2(handle, jpg1, (unsigned long)file_length[i], buffer, width, 0, height, TJPF_GRAY, TJFLAG_FASTDCT|TJFLAG_FASTUPSAMPLE))
            std::cout << "error in tjDecomp2" << std::endl;
        //tjDecompress2(handle, jpg1, (unsigned long)file_length[i], buffer, width, 0, height, TJPF_GRAY, TJFLAG_FASTDCT);
        //dump_bitmap(buffer, height, width, "outp/output_test"+std::to_string(i)+".bmp");
    }
    tjDestroy(handle);
    //dump_bitmap(buffer, height, width, "output_test");

    for(int i=0; i<1; i++){
     for(int j=0; j<178; j++){
        std::cout << std::to_string(buffer[i*178+j]) <<" ";
     }
    std::cout<<""<<std::endl;
    }
    */

   //free(grayImages);
   free(floatBig);
}





