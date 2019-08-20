#include "../include/binary_IO.hpp"
#include "../include/bitmap_IO.hpp"
#include "../include/hpc_helpers.hpp"

#define BLOCK_SIZE 32

typedef struct {
  int width, height, stride;
  float *elements;
} Matrix;
__device__ float getElement(const Matrix A, int row, int col) {
  return A.elements[row * A.stride + col];
}
__device__ void setElement(Matrix A, int row, int col, float val) {
  A.elements[row * A.stride + col] = val;
}
__device__ Matrix getSubMatrix(Matrix A, int row, int col) {
  Matrix Asub;
  Asub.width  = BLOCK_SIZE;
  Asub.height = BLOCK_SIZE;
  Asub.stride = A.stride;
  Asub.elements = &A.elements[A.stride * BLOCK_SIZE * row + BLOCK_SIZE * col];
  return Asub;
}

__global__ void MatMulKernel_XtX(Matrix X, float *C) {
  int blockRow = blockIdx.y;
  int blockCol = blockIdx.x;
  // Matrix Csub=getSubMatrix(C,blockRow,blockCol);
  // Matrix Csubt=getSubMatrix(C,blockCol,blockRow);
  float Cvalue = 0.0;
  int row = threadIdx.y;
  int col = threadIdx.x;
  int x = blockCol * BLOCK_SIZE + col;
  int y = blockRow * BLOCK_SIZE + row;
  if (blockCol * BLOCK_SIZE > blockRow * BLOCK_SIZE) return;
  for (int m = 0; m < ((X.height + BLOCK_SIZE - 1) / BLOCK_SIZE); ++m) {
    Matrix Asub = getSubMatrix(
        X, m, blockRow);  // swapped m and blockRow to access transpose(X)
    Matrix Bsub = getSubMatrix(X, m, blockCol);
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];
    As[row][col] = getElement(Asub, col, row);  // swapped col and row
    Bs[row][col] = getElement(Bsub, row, col);
    __syncthreads();
    if (x <= y) {
      for (int i = 0; i < BLOCK_SIZE; ++i) {
        Cvalue += As[row][i] * Bs[i][col];
      }
    }
    __syncthreads();
  }

  if (y < X.width && x <= y) {
    C[x * X.width + y] = Cvalue / X.height;
    C[y * X.width + x] = Cvalue / X.height;
  }
}

template <typename index_t, typename value_t>
__global__ void compute_mean_kernel(value_t *Data, value_t *Mean,
                                    index_t num_entries, index_t num_features) {
  auto thid = blockDim.x * blockIdx.x + threadIdx.x;

  if (thid < num_features) {
    value_t accum = 0;

#pragma unroll 32
    for (index_t entry = 0; entry < num_entries; entry++)
      accum += Data[entry * num_features + thid];

    Mean[thid] = accum / num_entries;
  }
}

template <typename index_t, typename value_t>
__global__ void correction_kernel(value_t *Data, value_t *Mean,
                                  index_t num_entries, index_t num_features) {
  auto thid = blockDim.x * blockIdx.x + threadIdx.x;
  if (thid < num_features) {
    value_t value = Mean[thid];
    for (index_t entry = 0; entry < num_entries; entry++)
      Data[entry * num_features + thid] -= value;
  }
}

template <typename index_t, typename value_t>
__global__ void covariance_kernel(value_t *Data, value_t *Cov,
                                  index_t num_entries, index_t num_features) {
  auto j = blockDim.x * blockIdx.x + threadIdx.x;
  auto i = blockDim.y * blockIdx.y + threadIdx.y;

  if (i < num_features && j < num_features) {
    value_t accum = 0;

    for (index_t entry = 0; entry < num_entries; entry++)
      accum += Data[entry * num_features + i] * Data[entry * num_features + j];

    Cov[i * num_features + j] = accum / num_entries;
  }
}

template <typename index_t, typename value_t>
__global__ void symmetric_covariance_kernel(value_t *Data, value_t *Cov,
                                            index_t num_entries,
                                            index_t num_features) {
  auto j = blockDim.x * blockIdx.x + threadIdx.x;
  auto i = blockDim.y * blockIdx.y + threadIdx.y;

  if (i < num_features && j <= i) {
    value_t accum = 0;

    for (index_t entry = 0; entry < num_entries; entry++)
      accum += Data[entry * num_features + i] * Data[entry * num_features + j];

    Cov[i * num_features + j] = Cov[j * num_features + i] = accum / num_entries;
  }
}

#define BLOCKSIZE 32
template <typename index_t, typename value_t>
__global__ void shared_covariance_kernel(value_t *Data, value_t *Cov,
                                         index_t num_entries,
                                         index_t num_features) {
  // convenience variables
  const index_t base_x = blockIdx.x * BLOCKSIZE;
  const index_t base_y = blockIdx.y * BLOCKSIZE;
  const index_t thid_y = threadIdx.y;
  const index_t thid_x = threadIdx.x;
  const index_t x = base_x + thid_x;
  const index_t y = base_y + thid_y;
  if (base_x > base_y) return;
  __shared__ value_t cache_x[32][32];
  __shared__ value_t cache_y[32][32];
  const index_t num_chunks = SDIV(num_entries, BLOCKSIZE);
  value_t accum = 0;
  // for each chunk
  for (index_t chunk = 0; chunk < num_chunks; chunk++) {  // 202599/32=6332
    // assign thread IDs to rows and columns
    const index_t row = thid_y + chunk * BLOCKSIZE;
    const index_t col_x = thid_x + base_x;
    const index_t col_y = thid_x + base_y;
    // check if valid row or column indices
    const bool valid_row = row < num_entries;
    const bool valid_col_x = col_x < num_features;
    const bool valid_col_y = col_y < num_features;
    // fill shared memory with tiles where thid_y enumerates
    // image identifiers (entries) and thid_x denotes feature
    // coordinates (pixels). cache_x corresponds to x and
    // cache_y to y where Cov[x,y] is the pairwise covariance
    cache_x[thid_y][thid_x] = Data[row * num_features + col_x];
    cache_y[thid_x][thid_y] = Data[row * num_features + col_y];

    __syncthreads();
    if (x <= y)
      for (index_t i = 0; i < BLOCKSIZE; i++)
        accum += cache_y[thid_y][i] * cache_x[i][thid_x];
    // this is needed to ensure that shared memory can be over-
    // written again in the next iteration
    __syncthreads();
  }
  // since Cov[x,y] = Cov[y,x] we only compute one entry
  if (y < num_features && x <= y)
    Cov[y * num_features + x] = Cov[x * num_features + y] = accum / num_entries;
}

int main(int argc, char *argv[]) {
  // set the identifier of the used CUDA device+

  cudaSetDevice(0);
  cudaDeviceReset();

  // 202599 grayscale images each of shape 55 x 45
  constexpr uint32_t imgs = 202599, rows = 55, cols = 45;

  // pointer for data matrix and mean vector
  float *data = nullptr, *cov = nullptr, *data_py3;
  cudaMallocHost(&data, sizeof(float) * imgs * rows * cols);
  CUERR
  cudaMallocHost(&data_py3, sizeof(float) * imgs * rows * cols);
  CUERR
  cudaMallocHost(&cov, sizeof(float) * rows * cols * rows * cols);
  CUERR

  Matrix d_A;
  d_A.width = d_A.stride = rows * cols;
  d_A.height = imgs;
  size_t size = sizeof(float) * imgs * rows * cols;
  cudaMalloc(&d_A.elements, size);

  // allocate storage on GPU
  float *Data = nullptr, *Mean = nullptr, *Cov = nullptr;
  // cudaMalloc(&Data, sizeof(float)*imgs*rows*cols);                      CUERR
  cudaMalloc(&Mean, sizeof(float) * rows * cols);
  CUERR
  cudaMalloc(&Cov, sizeof(float) * rows * cols * rows * cols);
  CUERR

  // load data matrix from disk
  TIMERSTART(read_data_from_disk)
  auto file_name = "./data/celebA.bin";
  // auto file_name_py3 = "/nvme/bm/celebA_rev3.bin_notebook_one";
  //auto file_name_py3 = "/nvme/bm/first_one";
  auto file_name_py3 = "./data/output_float2.bin";
  load_binary(data, imgs * rows * cols, file_name);
  load_binary(data_py3, imgs * rows * cols, file_name_py3);
  TIMERSTOP(read_data_from_disk)

  int same = 0, diff = 0;
  for (int i = 0; i < imgs * rows * cols; ++i) {
    if (fabs(data[i] - data_py3[i]) < 5.7f) {
      same++;
    } else {
      diff++;
    }
  }
  printf("compare data and data_pt3: same: %d diff: %d\n", same, diff);

  dump_bitmap(data_py3, rows, cols, "dump_first_data_py3.bmp");
  dump_bitmap(data, rows, cols, "dump_first_data.bmp");
  // exit(27);

  // copy data to device and reset Mean
  TIMERSTART(data_H2D)
  //    cudaMemcpy(Data, data, sizeof(float)*imgs*rows*cols,
  //              cudaMemcpyHostToDevice); CUERR

  cudaMemcpy(d_A.elements, data_py3, size, cudaMemcpyHostToDevice);
  CUERR

  cudaMemset(Mean, 0, sizeof(float) * rows * cols);
  CUERR
  TIMERSTOP(data_H2D)

  // compute mean
  TIMERSTART(compute_mean_kernel)
  compute_mean_kernel<<<SDIV(rows * cols, 1024), 1024>>>(d_A.elements, Mean,
                                                         imgs, rows * cols);
  CUERR
  TIMERSTOP(compute_mean_kernel)

  // correct mean
  TIMERSTART(correction_kernel)
  correction_kernel<<<SDIV(rows * cols, 1024), 1024>>>(d_A.elements, Mean, imgs,
                                                       rows * cols);
  CUERR
  TIMERSTOP(correction_kernel)

  // compute covariance matrix
  TIMERSTART(covariance_kernel)

  dim3 grid(SDIV(rows*cols,32), SDIV(rows*cols, 32));
  dim3 block(32, 32, 1);
  //    shared_covariance_kernel<<<grid, block>>>(d_A.elements, Cov, imgs,
  //    rows*cols); CUERR

  MatMulKernel_XtX<<<grid, block>>>(d_A, Cov);
  CUERR

  TIMERSTOP(covariance_kernel)

  // transfer covariance back to host
  TIMERSTART(cov_D2H)
  cudaMemcpy(cov, Cov, sizeof(float) * rows * cols * rows * cols,
             cudaMemcpyDeviceToHost);
  CUERR
  TIMERSTOP(cov_D2H)

  // write mean image to disk
  TIMERSTART(write_mean_image_to_disk)
  dump_bitmap(cov, rows * cols, rows * cols, "imgs/celebA_covariance.bmp");
  TIMERSTOP(write_mean_image_to_disk)

  // get rid of the memory
  cudaFreeHost(data);
  CUERR
  cudaFreeHost(cov);
  CUERR
  cudaFree(Data);
  CUERR
  cudaFree(Mean);
  CUERR
  cudaFree(Cov);
  CUERR
  cudaDeviceReset();

  // cudaMallocHost(&data, sizeof(float)*imgs*rows*cols);                  CUERR
  // float *d1= malloc(sizeof(float)*45*55);

  // load_binary(data, imgs*rows*cols, file_name);
}
