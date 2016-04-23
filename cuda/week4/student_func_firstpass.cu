 //Udacity HW 4
//Radix Sorting

#include "utils.h"
#include <thrust/host_vector.h>

/* Red Eye Removal
   ===============
   
   For this assignment we are implementing red eye removal.  This is
   accomplished by first creating a score for every pixel that tells us how
   likely it is to be a red eye pixel.  We have already done this for you - you
   are receiving the scores and need to sort them in ascending order so that we
   know which pixels to alter to remove the red eye.

   Note: ascending order == smallest to largest

   Each score is associated with a position, when you sort the scores, you must
   also move the positions accordingly.

   Implementing Parallel Radix Sort with CUDA
   ==========================================

   The basic idea is to construct a histogram on each pass of how many of each
   "digit" there are.   Then we scan this histogram so that we know where to put
   the output of each digit.  For example, the first 1 must come after all the
   0s so we have to know how many 0s there are to be able to start moving 1s
   into the correct position.


   2) Exclusive Prefix Sum of Histogram
   3) Determine relative offset of each digit
        For example [0 0 1 1 0 0 1]
                ->  [0 1 0 1 2 3 2]
   4) Combine the results of steps 2 & 3 to determine the final
      output location for each element and move it there

   LSB Radix sort is an out-of-place sort and you will need to ping-pong values
   between the input and output buffers we have provided.  Make sure the final
   sorted results end up in the output buffer!  Hint: You may need to do a copy
   at the end.

 */

//   1) Histogram of the number of occurrences of each digit
__global__ void my_first_histogram(unsigned int* const inputChannel,
																	 unsigned int* const outputChannel)
{
	extern __shared__ unsigned int s_data[];
	int myId = (threadIdx.x * threadIdx.y) + (blockDim.x * blockDim.y * blockIdx.x);
  int tid  = threadIdx.x * threadIdx.y;
	int arrayOffset = threadIdx.y * blockDim.x;
	s_data[tid] = inputChannel[myId];
  __syncthreads();

	unsigned int myCount = 0;
 	unsigned int mask = (1 << threadIdx.x);
	unsigned int i;
	// mask = (1 << n_bit)
	// assess: (element & mask) >> n_bit	
	for (i = arrayOffset; i < (arrayOffset + blockDim.x); i++) {
		myCount += (s_data[i] & (mask)) >> threadIdx.x; 
	}
  __syncthreads();  // All threads have their results ready to be put into shmem
	//write results to the correct location in shmem
	s_data[arrayOffset + threadIdx.x] = myCount;
  __syncthreads();  // results have all been written!

	// problem?
	//now reduce the 32 rows of 32 bit totals into 1 row of 32 bit totals using 32 threads
	if (myId < blockDim.x) {
		myCount = 0;
		for (i = 0; i < blockDim.x; i++) {
			myCount += s_data[(i * blockDim.x) + myId];
		}	
	  __syncthreads();  // The lucky 32 threads  have the reduce histo!
		// this step can by coalesced  later, so all LSBs are in first 32 spaces etc
		outputChannel[blockIdx.x * blockDim.x] = myCount;
	}
}

__global__ void simple_histogram_reduce(unsigned int* const inputChannel,
													  unsigned int* const outputChannel,
														unsigned int const numRows,
														unsigned int const numCols)
{
		extern __shared__ unsigned int r_data[];
		r_data[threadIdx.x] = inputChannel[threadIdx.x];
	  __syncthreads(); // load em upppp
		
		if (threadIdx.x < numCols) {
			unsigned int myCount = 0;
			for (unsigned int i = 0; i < numRows; i++) {
				myCount += r_data[threadIdx.x + (i * numCols)];
			}
			outputChannel[threadIdx.x] = myCount;
		}
}

__global__ void silly_scan(unsigned int * inputChannel, unsigned int * outputChannel)
{
    extern __shared__ unsigned int s_out_data[];
    int myId = threadIdx.x + blockDim.x * blockIdx.x;
    int tid  = threadIdx.x;
    int n_tid = blockDim.x - 1 - tid;
    s_out_data[myId] = inputChannel[myId];

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
    if (writing_to_bottom){outputChannel[tid] = s_out_data[tid];} // think result shud be in bottom row?
    else {outputChannel[tid] = s_out_data[tid + blockDim.x];}
}   

void your_sort(unsigned int* const d_inputVals,
               unsigned int* const d_inputPos,
               unsigned int* const d_outputVals,
               unsigned int* const d_outputPos,
               const size_t numElems)
{ 

//   1) Histogram of the number of occurrences of each digit
	unsigned int const valBitCount = 32;	
	unsigned int const maxThreads = 1024;
	unsigned int blockWidth = 32;	
	unsigned int numThreads = (blockWidth * blockWidth);		
	unsigned int const numBlocks = numElems / numThreads;
	const dim3 blockSize(blockWidth,blockWidth,1);
	const dim3 gridSize(numBlocks,1,1);

	unsigned int *d_intermediate;
	checkCudaErrors(cudaMalloc(&d_intermediate,
											sizeof(unsigned int) * numBlocks * valBitCount));

	//Iterate only once.. this should be more scalable..
		my_first_histogram<<<gridSize,blockSize,
													(numThreads * sizeof(unsigned int))>>>
		(d_inputVals, d_intermediate);
	//check for errors in kernels
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

	// Output of fist iteration	was numBlocks histograms... need to reduce them now
	//on host or on device?.. may as well do a single 1024 thread block
	// **** assuming numBlocks < maxThreads ***
	numThreads = numBlocks; 
	blockWidth = numBlocks / maxThreads + 1;
	simple_histogram_reduce<<<blockWidth,numThreads,
													(numBlocks * valBitCount * sizeof(unsigned int))>>>
		(d_intermediate, d_outputVals, numBlocks, valBitCount);	

	//check for errors in kernels
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

//   2) Exclusive Prefix Sum of Histogram
	silly_scan<<<gridSize, numThreads, ((numElems * 2) * sizeof(unsigned int))>>>
  	            (d_inputVals, d_outputVals);
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
//   3) Determine relative offset of each digit
//        For example [0 0 1 1 0 0 1]
//                ->  [0 1 0 1 2 3 2]
//   4) Combine the results of steps 2 & 3 to determine the final
//      output location for each element and move it there

// SERIAL HOST IMPLEMENTAITION
	unsigned int *h_inputVals = new unsigned int[numElems]; 
  unsigned int *h_inputPos = new unsigned int[numElems]; 
	unsigned int *h_outputVals = new unsigned int[numElems]; 
	unsigned int *h_outputPos = new unsigned int[numElems]; 
	checkCudaErrors(cudaMemcpy(h_inputVals, d_inputVals, (sizeof(unsigned int) * numElems), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(h_inputPos, d_inputPos, (sizeof(unsigned int) * numElems), cudaMemcpyDeviceToHost));


  
	const int numBits = 1;
  const int numBins = 1 << numBits;

  unsigned int *binHistogram = new unsigned int[numBins];
  unsigned int *binScan      = new unsigned int[numBins];

  unsigned int *vals_src = h_inputVals;
  unsigned int *pos_src  = h_inputPos;

  unsigned int *vals_dst = h_outputVals;
  unsigned int *pos_dst  = h_outputPos;

  //a simple radix sort - only guaranteed to work for numBits that are multiples of 2
  for (unsigned int i = 0; i < 8 * sizeof(unsigned int); i += numBits) {
    unsigned int mask = (numBins - 1) << i;

    memset(binHistogram, 0, sizeof(unsigned int) * numBins); //zero out the bins
    memset(binScan, 0, sizeof(unsigned int) * numBins); //zero out the bins

    //perform histogram of data & mask into bins
    for (unsigned int j = 0; j < numElems; ++j) {
      unsigned int bin = (vals_src[j] & mask) >> i;
      binHistogram[bin]++;
    }

    //perform exclusive prefix sum (scan) on binHistogram to get starting
    //location for each bin
    for (unsigned int j = 1; j < numBins; ++j) {
      binScan[j] = binScan[j - 1] + binHistogram[j - 1];
    }

    //Gather everything into the correct location
    //need to move vals and positions
    for (unsigned int j = 0; j < numElems; ++j) {
      unsigned int bin = (vals_src[j] & mask) >> i;
      vals_dst[binScan[bin]] = vals_src[j];
      pos_dst[binScan[bin]]  = pos_src[j];
      binScan[bin]++;
    }

    //swap the buffers (pointers only)
    std::swap(vals_dst, vals_src);
    std::swap(pos_dst, pos_src);
  }

  //we did an even number of iterations, need to copy from input buffer into output
  std::copy(h_inputVals, h_inputVals + numElems, h_outputVals);
  std::copy(h_inputPos, h_inputPos + numElems, h_outputPos);
	
	checkCudaErrors(cudaMemcpy(d_outputPos, h_outputPos, (sizeof(unsigned int) * numElems), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_outputVals, h_outputVals, (sizeof(unsigned int) * numElems), cudaMemcpyHostToDevice));

  delete[] binHistogram;
  delete[] binScan;

	checkCudaErrors(cudaFree(d_intermediate));
}

