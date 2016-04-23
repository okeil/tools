// Homework 2
// Image Blurring
//****************************************************************************

#include "reference_calc.cpp"
#include "utils.h"

__global__
void gaussian_blur(const unsigned char* const inputChannel,
                   unsigned char* const outputChannel,
                   int numRows, int numCols,
                   const float* const filter, const int filterWidth)
{

    int y = (threadIdx.y + (blockDim.y * blockIdx.y));
    int x = (threadIdx.x + (blockDim.x * blockIdx.x));
        if ((y >= numRows) || (x >= numCols)) { return;}
        float result = 0.f;
      //For every value in the filter around the pixel (c, r)
      for (int filter_r = -filterWidth/2; filter_r <= filterWidth/2; ++filter_r) {
        for (int filter_c = -filterWidth/2; filter_c <= filterWidth/2; ++filter_c) {
          //Find the global image position for this filter position
          //clamp to boundary of the image
          int image_r = min(max(y + filter_r, 0), static_cast<int>(numRows - 1));
          int image_c = min(max(x + filter_c, 0), static_cast<int>(numCols - 1));

          float image_value = static_cast<float>(inputChannel[image_r * numCols + image_c]);
          float filter_value = filter[(filter_r + filterWidth/2) * filterWidth + filter_c + filterWidth/2];

          result += image_value * filter_value;
        }
      }

      outputChannel[y * numCols + x] = result;
}


//This kernel takes in an image represented as a uchar4 and splits
//it into three images consisting of only one color channel each
__global__
void separateChannels(const uchar4* const inputImageRGBA,
                      int numRows,
                      int numCols,
                      unsigned char* const redChannel,
                      unsigned char* const greenChannel,
                      unsigned char* const blueChannel)
{
    int y = (threadIdx.y + (blockDim.y * blockIdx.y));
    int x = (threadIdx.x + (blockDim.x * blockIdx.x));
    int i = (y * numCols) + x;
    if (i >= numCols * numRows){ return;} else {
        uchar4 rgba = inputImageRGBA[i];    
        redChannel[i]   = rgba.x;
        greenChannel[i] = rgba.y;
        blueChannel[i]  = rgba.z;
    }
}

//This kernel takes in three color channels and recombines them
//into one image.  The alpha channel is set to 255 to represent
//that this image has no transparency.
__global__
void recombineChannels(const unsigned char* const redChannel,
                       const unsigned char* const greenChannel,
                       const unsigned char* const blueChannel,
                       uchar4* const outputImageRGBA,
                       int numRows,
                       int numCols)
{
  const int2 thread_2D_pos = make_int2( blockIdx.x * blockDim.x + threadIdx.x,
                                        blockIdx.y * blockDim.y + threadIdx.y);

  const int thread_1D_pos = thread_2D_pos.y * numCols + thread_2D_pos.x;

  //make sure we don't try and access memory outside the image
  //by having any threads mapped there return early
  if (thread_2D_pos.x >= numCols || thread_2D_pos.y >= numRows)
    return;

  unsigned char red   = redChannel[thread_1D_pos];
  unsigned char green = greenChannel[thread_1D_pos];
  unsigned char blue  = blueChannel[thread_1D_pos];

  //Alpha should be 255 for no transparency
  uchar4 outputPixel = make_uchar4(red, green, blue, 255);

  outputImageRGBA[thread_1D_pos] = outputPixel;
}

//These are weird pointers maybe they need to be change to trailing __
unsigned char *d_red, *d_green, *d_blue;
float         *d_filter;

void allocateMemoryAndCopyToGPU(const size_t numRowsImage, const size_t numColsImage,
                                const float* const h_filter, const size_t filterWidth)
{

  //allocate memory for the three different channels
  //original 
  checkCudaErrors(cudaMalloc(&d_red,   sizeof(unsigned char) * numRowsImage * numColsImage));
  checkCudaErrors(cudaMalloc(&d_green, sizeof(unsigned char) * numRowsImage * numColsImage));
  checkCudaErrors(cudaMalloc(&d_blue,  sizeof(unsigned char) * numRowsImage * numColsImage));

 
  checkCudaErrors(cudaMalloc(&d_filter, sizeof(float) * filterWidth * filterWidth));
  checkCudaErrors(cudaMemset(d_filter, 0, sizeof(float) * filterWidth * filterWidth)); //make sure no memory is left laying around
  checkCudaErrors(cudaMemcpy(d_filter, h_filter, 
                                sizeof(float) * filterWidth * filterWidth, cudaMemcpyHostToDevice));



}

void your_gaussian_blur(const uchar4 * const h_inputImageRGBA, uchar4 * const d_inputImageRGBA,
                        uchar4* const d_outputImageRGBA, const size_t numRows, const size_t numCols,
                        unsigned char *d_redBlurred, 
                        unsigned char *d_greenBlurred, 
                        unsigned char *d_blueBlurred,
                        const int filterWidth)
{
  //TODO: Set reasonable block size (i.e., number of threads per block)
    const dim3 blockSize(numRows,1,1);
    const dim3 gridSize(1,1,1);

    const int blockWidth = 16;
    const dim3 blockSize2(blockWidth,blockWidth,1);
    const dim3 gridSize2(((numCols/blockWidth) + 1),((numRows/blockWidth) + 1),1);
    
  //TODO:
  //Compute correct grid size (i.e., number of blocks per kernel launch)
  //from the image size and and block size.
  
  //TODO: Launch a kernel for separating the RGBA image into different color channels
  separateChannels<<<gridSize2, blockSize2>>>(d_inputImageRGBA, numRows, numCols,
                                            d_red, d_green, d_blue);
  
  // Call cudaDeviceSynchronize(), then call checkCudaErrors() immediately after
  // launching your kernel to make sure that you didn't make any mistakes.
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  //TODO: Call your convolution kernel here 3 times, once for each color channel.
  gaussian_blur<<<gridSize2, blockSize2>>>(d_red, d_redBlurred, numRows, numCols, d_filter, filterWidth);
  gaussian_blur<<<gridSize2, blockSize2>>>(d_green, d_greenBlurred, numRows, numCols, d_filter, filterWidth);
  gaussian_blur<<<gridSize2, blockSize2>>>(d_blue, d_blueBlurred, numRows, numCols, d_filter, filterWidth);
  
  // Again, call cudaDeviceSynchronize(), then call checkCudaErrors() immediately after
  // launching your kernel to make sure that you didn't make any mistakes.
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  // Now we recombine your results. We take care of launching this kernel for you.
  //
  // NOTE: This kernel launch depends on the gridSize and blockSize variables,
  // which you must set yourself.
  recombineChannels<<<gridSize2, blockSize2>>>(d_redBlurred,
                                             d_greenBlurred,
                                             d_blueBlurred,
                                             d_outputImageRGBA,
                                             numRows,
                                             numCols);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
}


//Free all the memory that we allocated
//TODO: make sure you free any arrays that you allocated
void cleanup() {
  checkCudaErrors(cudaFree(d_red));
  checkCudaErrors(cudaFree(d_green));
  checkCudaErrors(cudaFree(d_blue));
  checkCudaErrors(cudaFree(d_filter));
}
