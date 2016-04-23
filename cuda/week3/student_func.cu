/* Udacity Homework 3
   HDR Tone-mapping

  Background HDR
  ==============

  A High Definition Range (HDR) image contains a wider variation of intensity
  and color than is allowed by the RGB format with 1 byte per channel that we
  have used in the previous assignment.  

  To store this extra information we use single precision floating point for
  each channel.  This allows for an extremely wide range of intensity values.

  In the image for this assignment, the inside of church with light coming in
  through stained glass windows, the raw input floating point values for the
  channels range from 0 to 275.  But the mean is .41 and 98% of the values are
  less than 3!  This means that certain areas (the windows) are extremely bright
  compared to everywhere else.  If we linearly map this [0-275] range into the
  [0-255] range that we have been using then most values will be mapped to zero!
  The only thing we will be able to see are the very brightest areas - the
  windows - everything else will appear pitch black.

  The problem is that although we have cameras capable of recording the wide
  range of intensity that exists in the real world our monitors are not capable
  of displaying them.  Our eyes are also quite capable of observing a much wider
  range of intensities than our image formats / monitors are capable of
  displaying.

  Tone-mapping is a process that transforms the intensities in the image so that
  the brightest values aren't nearly so far away from the mean.  That way when
  we transform the values into [0-255] we can actually see the entire image.
  There are many ways to perform this process and it is as much an art as a
  science - there is no single "right" answer.  In this homework we will
  implement one possible technique.

  Background Chrominance-Luminance
  ================================

  The RGB space that we have been using to represent images can be thought of as
  one possible set of axes spanning a three dimensional space of color.  We
  sometimes choose other axes to represent this space because they make certain
  operations more convenient.

  Another possible way of representing a color image is to separate the color
  information (chromaticity) from the brightness information.  There are
  multiple different methods for doing this - a common one during the analog
  television days was known as Chrominance-Luminance or YUV.

  We choose to represent the image in this way so that we can remap only the
  intensity channel and then recombine the new intensity values with the color
  information to form the final image.

  Old TV signals used to be transmitted in this way so that black & white
  televisions could display the luminance channel while color televisions would
  display all three of the channels.
  

  Tone-mapping
  ============

  In this assignment we are going to transform the luminance channel (actually
  the log of the luminance, but this is unimportant for the parts of the
  algorithm that you will be implementing) by compressing its range to [0, 1].
  To do this we need the cumulative distribution of the luminance values.

  Example
  -------

  input : [2 4 3 3 1 7 4 5 7 0 9 4 3 2]
  min / max / range: 0 / 9 / 9

  histo with 3 bins: [4 7 3]

  cdf : [4 11 14]


  Your task is to calculate this cumulative distribution by following these
  steps.

*/


#include "reference_calc.cpp"
#include "utils.h"

__global__ void my_first_shmem_minmax_reduce(const bool max, float * d_out, const float * d_in)
{
    // sdata is allocated in the kernel call: 3rd arg to <<<b, t, shmem>>>
    extern __shared__ float sdata[];
    int myId = threadIdx.x + blockDim.x * blockIdx.x;
    int tid  = threadIdx.x;

    // load shared mem from global mem
    sdata[tid] = d_in[myId];
    __syncthreads();            // make sure entire block is loaded!

    // do max reduction in shared mem
    if (max){
        for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
        {
                if (tid < s) {sdata[tid] = fmax(sdata[tid], sdata[tid + s]);}
                __syncthreads();        // make sure all adds at one stage are done!
        }
    } else {
        for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
        {
                if (tid < s) {sdata[tid] = fmin(sdata[tid], sdata[tid + s]);}
                __syncthreads();        // make sure all adds at one stage are done!
        }
    }
    // only thread 0 writes result for this block back to global mem
    if (tid == 0) {d_out[blockIdx.x] = sdata[0];}
}

__global__ void my_first_keyValGen(unsigned int * d_out, const float * d_in, 
                                                float d_min, float d_range,
                                                const size_t numBins)
{
    unsigned int myId = threadIdx.x + blockDim.x * blockIdx.x;
    //bin = (lum[i] - lumMin) / lumRange * numBin
    d_out[myId] = fmin((d_in[myId] - d_min) / d_range * numBins, (numBins - 1));
}

__global__ void my_first_bleh(unsigned int * d_out, 
                                const unsigned int * d_in, 
                                const int threadRange)
{
    //extern __shared__ unsigned int bsdata[];
    int startInd  = threadIdx.x * threadRange;
    for (int i = startInd; i < (startInd + threadRange); i++) {
        atomicAdd(&(d_out[d_in[i]]), 1);
    }
}

//ERROR EXISTS!!
__global__ void my_first_exclusive_scan(unsigned int * d_cdf_out, const unsigned int * d_in)
{
    extern __shared__ unsigned int s_out_data[];
    int myId = threadIdx.x + blockDim.x * blockIdx.x;
    int tid  = threadIdx.x;
    int n_tid = blockDim.x - 1 - tid;
    s_out_data[myId] = d_in[myId];
    s_out_data[myId + blockDim.x] = d_in[myId];
    __syncthreads();            // make sure entire block is loaded!
    //++++++++++REDUCE++++++++++++++
    bool writing_to_bottom = true;
    for (int i = 2; i <= blockDim.x; i <<= 1) {
        if ((n_tid % i) == 0) {
            //case of totaling value
            if (writing_to_bottom){s_out_data[tid + blockDim.x] = 
                    s_out_data[tid] + s_out_data[tid - (i/2)];} 
            else{s_out_data[tid] = s_out_data[tid + blockDim.x] + 
                    s_out_data[tid + blockDim.x - (i/2)];}
        } else if (tid % 2 == 1) {
            if (writing_to_bottom){s_out_data[tid + blockDim.x] = 
                    s_out_data[tid];} 
            else{s_out_data[tid] = s_out_data[tid + blockDim.x];}            
        }
        writing_to_bottom = !writing_to_bottom;
        __syncthreads();        // make sure all adds at one stage are done!
    }
    
    //++++++++++DOWNSWEEP++++++++++++++
    if (!writing_to_bottom){s_out_data[tid] = s_out_data[tid + blockDim.x];}
     //results should be in bottom row?
        if (tid == (blockDim.x - 1)) {s_out_data[tid] = 0;}
            __syncthreads();        // make sure all adds at one stage are done!
    writing_to_bottom = true;
    for (int i = blockDim.x; i > 1; i >>= 1) {
        if ((n_tid % i) == 0) {
            //case of totalng value
            if (writing_to_bottom){
                s_out_data[tid + blockDim.x] = s_out_data[tid] + s_out_data[tid - (i/2)];
                s_out_data[tid + blockDim.x - (i/2)] = s_out_data[tid];
            } 
            else{
                s_out_data[tid] = s_out_data[tid + blockDim.x] + s_out_data[tid + blockDim.x - (i/2)];}
                s_out_data[tid - (i/2)] = s_out_data[tid + blockDim.x];
        } 
        writing_to_bottom = !writing_to_bottom;
        __syncthreads();        // make sure all adds at one stage are done!
    }
    if (writing_to_bottom){d_cdf_out[tid] = s_out_data[tid];} // think result shud be in bottom row?
    else {d_cdf_out[tid] = s_out_data[tid + blockDim.x];}
}   

void your_histogram_and_prefixsum(const float* const d_logLuminance,
                                  unsigned int* const d_cdf,
                                  float &min_logLum,
                                  float &max_logLum,
                                  const size_t numRows,
                                  const size_t numCols,
                                  const size_t numBins)
{
  //TODO
  /*Here are the steps you need to implement
    3) generate a histogram of all the values in the logLuminance channel using
       the formula: bin = (lum[i] - lumMin) / lumRange * numBins
    4) Perform an exclusive scan (prefix sum) on the histogram to get
       the cumulative distribution of luminance values (this should go in the
       incoming d_cdf pointer which already has been allocated for you)      
*/
    // Kernel specs
    const int maxThreadsPerBlock = 1024;
    int threads = maxThreadsPerBlock;
    int blocks = (numRows * numCols) / maxThreadsPerBlock;

    
    //  1) find the minimum and maximum value in the input logLuminance channel
    //   store in min_logLum and max_logLum
    float * d_intermediate, * d_max_logLum, * d_min_logLum;
    checkCudaErrors(cudaMalloc((void **) &d_intermediate, (threads * sizeof(float)))); //overallocated
    checkCudaErrors(cudaMalloc((void **) &d_max_logLum, sizeof(float)));
    checkCudaErrors(cudaMalloc((void **) &d_min_logLum, sizeof(float)));
    
    for (int i = 1; i < 3; i++) {
        bool max_bool = (i==1) ? true : false;    
        my_first_shmem_minmax_reduce<<<blocks, threads, (threads * sizeof(float))>>>
            (max_bool, d_intermediate, d_logLuminance);
         // now we're down to one block left, so reduce it
        threads = blocks; // launch one thread for each block in prev step
        blocks = 1; 
        //pull down max/min result
        if (max_bool) {
                my_first_shmem_minmax_reduce<<<blocks, threads, (threads * sizeof(float))>>>
                (max_bool, d_max_logLum, d_intermediate);
                checkCudaErrors(cudaMemcpy(&max_logLum, d_max_logLum, (sizeof(float)), cudaMemcpyDeviceToHost));
        }
        else {
                my_first_shmem_minmax_reduce<<<blocks, threads, (threads * sizeof(float))>>>
                        (max_bool, d_min_logLum, d_intermediate);
                checkCudaErrors(cudaMemcpy(&min_logLum, d_min_logLum, (sizeof(float)), cudaMemcpyDeviceToHost));
        }
    }
    
    
    // 2) subtract them to find the range
    float h_range = max_logLum - min_logLum;
    /*
     * 3) generate a histogram of all the values in the logLuminance channel using
       the formula: bin = (lum[i] - lumMin) / lumRange * numBins
    */  
    // Perhaps keep min_logLum in global memory, is it accessible via other kernels from global memory?
    unsigned int * d_histo, * d_keyArray;
    
    //checkCudaErrors(cudaMalloc((void **) &d_histo, sizeof(unsigned int) * numBins));
    checkCudaErrors(cudaMalloc(&d_histo, sizeof(unsigned int) * numBins));
    checkCudaErrors(cudaMalloc((void **) &d_keyArray, (sizeof(unsigned int) * numCols * numRows)));
    checkCudaErrors(cudaMemset(d_histo, 0, sizeof(unsigned int) * numBins)); //make sure no memory is left laying around
    
    threads = maxThreadsPerBlock;
    blocks = (numRows * numCols) / maxThreadsPerBlock;

    my_first_keyValGen<<<blocks, threads>>>
           (d_keyArray, d_logLuminance, min_logLum, h_range, numBins);
        cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    
    my_first_bleh<<<1, threads, (numBins * sizeof(unsigned int))>>>
                        (d_histo, d_keyArray, blocks);
        cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    
    /* 4) Perform an exclusive scan (prefix sum) on the histogram to get
       the cumulative distribution of luminance values (this should go in the
       incoming d_cdf pointer which already has been allocated for you)       */
    threads = maxThreadsPerBlock;
    blocks = numBins / threads;

    my_first_exclusive_scan<<<blocks, threads, ((numBins * 2) * sizeof(unsigned int))>>>
                    (d_cdf, d_histo);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    }