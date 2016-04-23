 //Udacity HW 4
//Radix Sorting
// Needs work, slower than serial implementation!?!?! gosh

#include "utils.h"
#include <thrust/host_vector.h>
#include <stdio.h>
#include <stdlib.h>
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

__global__ void local_bucket_sort(unsigned int* const inputChannel,
																	unsigned int* const inputRef,
																	unsigned int* const outputChannel,
																	unsigned int* const outputRef,
																	unsigned int const numBits,
																	unsigned int startBit,
																	unsigned int const numBuckets,
																	unsigned int const numElems
																	)
{
	//PREP LOCAL VALS	
	extern __shared__ unsigned int s_data[];
	unsigned int myId = threadIdx.x + (blockIdx.x * blockDim.x);
	if (myId < numElems) {
	unsigned int tid = threadIdx.x;
	unsigned int myVal = inputChannel[myId];
	unsigned int myBucket = 0;
	unsigned int bucketOccCount;

	// 1 - Assign value to a bucket
	// Ie with 8 bit passes need to make sure bucket nums are 0 - 255**
	for (unsigned int i = 0; i < numBits; i++) {
		myBucket += ((myVal >> startBit) & (1 << i));
	}
	s_data[tid] = myBucket;
	__syncthreads();

	// 2 - Total the number of each bucket- There is only 1024 elements per block so doing shortcut.
	// results to be written to global memory in  prep for stage 3
	// Needs to be a local scan to output bucket based scatter address for each elem & a total of elems in each buck to be sent to output channel.

	if (tid < numBuckets) {
	bucketOccCount = 0;
		for (unsigned int i = 0; i < blockDim.x; i++)
		{
			if (s_data[i] == tid)
			{
				s_data[blockDim.x + i] = bucketOccCount;				
				bucketOccCount += 1;
			}
		}
		outputChannel[blockIdx.x + (gridDim.x * tid)] = bucketOccCount; //ready for coallesced scan
	}
	__syncthreads();

/// *** KERNEL SHOULD END HERE **///
	// 3 - Thread 1 conducts a SCAN on the global offset array (single thread it for now)
	//assuming image is not larger than 1024 X 1024...
	
	if (blockIdx.x == 0) {
		if (tid == 0)
		{
			bucketOccCount = 0;	
			//scan global bucket totals to advise output locations for each block bucket
			for (unsigned int i = 0; i < (numBuckets * gridDim.x); i++)
			{
				outputRef[i] = bucketOccCount; 
				bucketOccCount += outputChannel[i];
			}
		}
	}
	__syncthreads();

	//4 - Write element to its new location in outputChannel and OutputPos
	unsigned int const myBlockBucketOffset = outputRef[blockIdx.x + (myBucket * gridDim.x)];
	unsigned int  mySortedAddress = s_data[tid + blockDim.x];
	outputChannel[myBlockBucketOffset + mySortedAddress] = myVal;
	outputRef[myBlockBucketOffset + mySortedAddress] = inputRef[myVal]; //updating pixel location ref
	__syncthreads();
	inputRef[myId] = outputRef[myId];
}
}

void your_sort(unsigned int* const d_inputVals,
               unsigned int* const d_inputPos,
               unsigned int* const d_outputVals,
               unsigned int* const d_outputPos,
               const size_t numElems)
{ 
	//CONSTANTS
	unsigned int const valBitCount = 32;	
	unsigned int const numBitsPerPass = 8;
	unsigned int const maxThreads = 1024;
	
	//PREP VALS
	unsigned int const numBuckets = pow(2,numBitsPerPass);	
	unsigned int numThreads = maxThreads;
	unsigned int numBlocks = (numElems % numThreads == 0) ? numElems / numThreads : (numElems / numThreads) + 1; //Assuming numElems % maxThreads===0 :/
	const dim3 blockSize(numThreads, 1, 1);
	const dim3 gridSize(numBlocks, 1, 1);

	for (unsigned int startBit = 0; startBit < valBitCount; startBit += numBitsPerPass)
	{
			local_bucket_sort<<<gridSize, blockSize, ((numThreads * 2) * sizeof(unsigned int))>>>
      (d_inputVals, d_inputPos, d_outputVals, d_outputPos, numBitsPerPass, startBit, numBuckets, numElems);
	}
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

}

