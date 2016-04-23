/* Udacity HW5
   Histogramming for Speed

   The goal of this assignment is compute a histogram
   as fast as possible.  We have simplified the problem as much as
   possible to allow you to focus solely on the histogramming algorithm.

   The input values that you need to histogram are already the exact
   bins that need to be updated.  This is unlike in HW3 where you needed
   to compute the range of the data and then do:
   bin = (val - valMin) / valRange to determine the bin.

   Here the bin is just:
   bin = val

   so the serial histogram calculation looks like:
   for (i = 0; i < numElems; ++i)
     histo[val[i]]++;

   That's it!  Your job is to make it run as fast as possible!

   The values are normally distributed - you may take
   advantage of this fact in your implementation.

*/


#include "utils.h"
#include "reference.cpp"


__global__
void yourHisto(const unsigned int* const vals, //INPUT
               unsigned int* const histo,      //OUPUT
               const unsigned int threadRange,
              const unsigned int numBins)
{
	extern __shared__  unsigned int s_data[];
    unsigned int tid = threadIdx.x;
    unsigned int myId = tid + (blockIdx.x * blockDim.x);
    unsigned int startLoc = myId * threadRange;
    
    for (int  i = startLoc; i < (startLoc + threadRange); i++) {
        //atomicAdd(&s_data[(vals[i])], 1);
        atomicAdd(&(histo[vals[i]]), 1);
    }
    //__syncthreads();
		if (tid < numBins) atomicAdd(&(histo[tid]), 1);    
	//avoid assumption that there are more threads than there are bins?
    // cant get shared memory working :(
    /*
    if (tid == 0) {
        for (int  i = 0; i < (numBins); i++) {
            atomicAdd(&histo[i], s_data[i]);
        }
    }
*/
}

void computeHistogram(const unsigned int* const d_vals, //INPUT
                      unsigned int* const d_histo,      //OUTPUT
                      const unsigned int numBins,
                      const unsigned int numElems)
{
    //TODO Launch the yourHisto kernel
    unsigned int blockX = 1024;
    unsigned int blockY = 1;
    unsigned int numThreads = blockX * blockY;
    unsigned int threadRange = 16;
    dim3 blocks(numElems/(numThreads * threadRange)); // change this val to make blocks elem ranged
    dim3 threads(blockX, blockY);
    yourHisto<<<blocks,threads, (numBins * sizeof(unsigned int))>>>(d_vals, d_histo, threadRange, numBins);  

    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

}


__global__
void yourHisto(const unsigned int* const vals, //INPUT
               unsigned int* const histo,      //OUPUT
               const unsigned int numElems,
               const unsigned int threadRange,
              const unsigned int numBins)
{
	extern __shared__ unsigned int s_data[];
    unsigned int tid = threadIdx.x;
    unsigned int myId = tid + (blockIdx.x * blockDim.x);
    unsigned int startLoc = myId * threadRange;
    
    for (int  i = startLoc; i < (startLoc + threadRange); i++) {
        atomicAdd(&(s_data[vals[i]]), 1);
        //atomicAdd(&(histo[vals[i]]), 1);
    }
    __syncthreads();
    
    if (tid < numBins) atomicAdd(&(histo[tid]), 1);

}
