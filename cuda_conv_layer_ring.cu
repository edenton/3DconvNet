#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <cuda_profiler_api.h> 
#include "cuda_conv_layer_ring.cuh"
#include "cutil.cuh"



template <int K>
__global__ void convLayerKernel(float* inputMaps, float* weights, float* outputMaps, int N, int M, int F_in, int F_out, int S, int Nt, int Mt, int Y_offset) 
{
	/* 
	(1) Determine which output feature and sample this block is working on
	*/
	const int f_out = blockIdx.x; 
	const int s = blockIdx.y;

	const int id = threadIdx.x;

	extern __shared__ float sharedArr[]; 
	float *sharedWeights = sharedArr; //dimension (K, K, K)
	float *sharedInput = &sharedArr[K*K*K]; //dimension (N, Nt)
	float *sharedOutput = &sharedArr[K*K*K + N*Nt]; //dimension (M, Mt, K)

	register float inpBuffer[K];

	int f_in;
	int i;
	int Z_offset;
	int X_offset;
	int sharedOutputBufferStart; // because we're using a ring buffer
	int inpBufferStart; // because we're using a ring buffer
	int y;
	int z;
	int toutZ;

	// To keep track of where we are in weight matrix in x, y, z dimensions
	int wx;
	int wy;
	int wz;

	

	// Loop over input maps
	for (f_in = 0; f_in < F_in; f_in++) {
		
		// First bring weights into shared memory - spread work accross threads
		for (i = 0; i < K*K*K*blockDim.x; i+= blockDim.x) {
			if (i + id < K*K*K) //dont run over end of filter
				sharedWeights[i + id] = weights[(K*K*K*F_out)*f_in + (K*K*K)*f_out + i+ id];
		}

		// Initialize shared output map to 0 - spread work accross threads
		for (i = 0; i < M*Mt*K*blockDim.x; i+= blockDim.x) {
			if (i + id < M*Mt*K) //dont run over end of filter
				sharedOutput[i + id] = 0;
		}

		Z_offset = 0;
		sharedOutputBufferStart = 0; 
		
		// Loop over z dimensions
		for (z = 0; z < N; z++) {
			
			// For z > K, every iteration means we've finished an entire output map plane
			if (z >= K) {

				__syncthreads(); // make sure all threads are here before we touch the shared memory output array

				// Move outputmap plane to global memory and then reset it to 0
				for (i = 0; i < M*Mt*blockDim.x; i+= blockDim.x) {
					if (i + id < M*Mt) {
						if (f_in == 0) //fill with 0's
						 	outputMaps[(M*M*M*F_out)*s + (M*M*M)*f_out + (M*M)*(z - K) + (Y_offset*M + i + id)] = 0;

						outputMaps[(M*M*M*F_out)*s + (M*M*M)*f_out + (M*M)*(z - K) + (Y_offset*M + i + id)] 
							+= sharedOutput[(M*Mt)*RBINDX(0, sharedOutputBufferStart, K) + (i + id)];

						sharedOutput[(M*Mt)*RBINDX(0, sharedOutputBufferStart, K) + (i + id)] = 0;
					}
				}

				Z_offset++;
				sharedOutputBufferStart = sharedOutputBufferStart == K-1 ? 0 : sharedOutputBufferStart + 1;
				
			}

			// Bring input map slice into shared memory 
			for (i = 0; i < N*Nt*blockDim.x; i+= blockDim.x) {
				if (i + id < N*Nt)
					sharedInput[i + id] = inputMaps[(N*N*N*F_in)*s + (N*N*N)*f_in + (N*N)*z + (Y_offset*N + i + id)];
			}

			__syncthreads(); // make sure all threads catch up before we start to use the shared input array

			// shift accross x dimension
			X_offset = 0;
			for (X_offset = 0; X_offset < M; X_offset += blockDim.x) {

				for (wx = 0; wx < K; wx++) {
				
					if (id + X_offset < M) { // otherwise this thread is over the X dim bounds of input map
						//now bring initial input map chunk data into registers

						#pragma unroll
						for (i = 0; i < K; i++) {
							inpBuffer[i] = sharedInput[i*N + (id + X_offset + wx)];
						}

						inpBufferStart = 0;
						//Now do the 1D convolution, looping vertically down each thread's strip of data
						for (y = 0; y < Nt - K + 1; y++) {
							// Loop over weight in z dimension
							#pragma unroll
							for (wz = 0; wz < K; wz++) {
								//if (z < K && wz > z) continue;
								if (!(z < K && wz > z)) {
									toutZ = z - wz - Z_offset; // Where we are in the internal partial output map

									#pragma unroll
									for (wy = 0; wy < K; wy++) {
										sharedOutput[(M*Mt)*RBINDX(toutZ, sharedOutputBufferStart, K) + (M)*y + (id + X_offset)] 
											+= inpBuffer[RBINDX(wy, inpBufferStart, K)] * sharedWeights[(K*K)*wz + (K)*wy + wx];
									}
								}
							}

							// bring next input map pixel into register
							inpBuffer[inpBufferStart] = sharedInput[(y + K)*N + (id + X_offset + wx)];
							inpBufferStart = inpBufferStart == K-1 ? 0 : inpBufferStart + 1;


						}				

					}
				}

				
			}
		}

		__syncthreads();

		z = N;
		// Move last outputmap place to global memory
		for (i = 0; i < M*Mt*blockDim.x; i+= blockDim.x) {
			if (i + id < M*Mt) {
				if (f_in == 0) //fill with 0's
				 	outputMaps[(M*M*M*F_out)*s + (M*M*M)*f_out + (M*M)*(z - K) + (Y_offset*M + i + id)] = 0;

				outputMaps[(M*M*M*F_out)*s + (M*M*M)*f_out + (M*M)*(z - K) + (Y_offset*M + i + id)] 
					+= sharedOutput[(M*Mt)*RBINDX(0, sharedOutputBufferStart, K) + (i + id)];
			}
		}
		// reset output map plane to 0


	} // end f_in loop
	// Shift horizontally


}

/*
input_maps:		(N, N, N, F_in, S) input feature maps
weights:		(K, K, K, F_out, F_in) filter weights
output_maps:	(M, M, M, F_out, S) output feature maps

N:		input feature map dimension (same in x, y, z direction)
M:		output map dimension
F_in:	number of channels in input layer
F_out: 	number of channels in output layer
K:		filter size (same in x, y, z direction)
S: 		batch size

access input_maps[i, j, k, f_in, s] by input_maps[(N*N*N*F_in)*s + (N*N*N)*f_in + (N*N)*k + (N)*j + i ]
access weights[i, j, k, f_out, f_in] by weights[(K*K*K*F_out)*f_in + (K*K*K)*f_out + (K*K)*k + (K)*j + i ]
access output_maps[i, j, k, f_out, s] by output_maps[(M*M*M*F_out)*s + (M*M*M)*f_out + (M*M)*k + (M)*j + i ]
*/
__host__ void convLayerRing(float* inputMaps, float* weights, float* outputMaps, int N, int M, int F_in, int F_out, int K, int S)
{

	cudaDeviceProp devProp;
    cudaGetDeviceProperties(&devProp, 0);


    // Ensure have large enough grid size
    int G_x = F_out;
    int G_y = S;
    assert(G_x <= devProp.maxGridSize[0]);
    assert(G_y <= devProp.maxGridSize[1]);
    
    dim3 dimGrid(G_x, G_y);
     
    /* 
    Determine block sizes. We have 2 contraints:
    	- regsPerBlock / threadsPerBlock > K +  (want to hold width of kernel in registers)
    	- Nt * N * sizeof(float) + K*K*K*sizeof(float) < sharedMemory (want to hold whole X x Y plane in shared memory)
    Nt = width of input matrix worked on at a given time (ideally N)
    */
    int threadsPerBlock = MIN(M, devProp.warpSize);
    int Nt = MIN( (devProp.sharedMemPerBlock - K*K*K*sizeof(float)) / ( (M*K + N)*sizeof(float)), N); // fix this
    int Mt = Nt - K + 1;
    assert(Mt > 0);
    //printf("Nt = %d, Mt = %d, threadsPerBlock = %d\n", Nt, Mt, threadsPerBlock);
    int sizeSharedMem = M*Mt*K*sizeof(float) + N*Nt*sizeof(float) + K*K*K*sizeof(float); //output + input + weights
    assert(sizeSharedMem < devProp.sharedMemPerBlock);

    //copy input maps and weights to global memory
    int sizeIn = N*N*N*F_in*S*sizeof(float);
    int sizeOut = M*M*M*F_out*S*sizeof(float);
    int sizeW = K*K*K*F_out*F_in*sizeof(float);

    //printf("sizeIn + sizeOut + sizeW = %d\ndevProp.totalGlobalMem = %u\n", sizeIn + sizeOut + sizeW, devProp.totalGlobalMem);
    assert((uint) (sizeIn + sizeOut + sizeW) < devProp.totalGlobalMem);
    float *d_inputMaps, *d_outputMaps, *d_weights;
    HANDLE_ERROR( cudaMalloc((void **)&d_inputMaps, sizeIn) );
    HANDLE_ERROR( cudaMalloc((void **)&d_outputMaps, sizeOut) );
    HANDLE_ERROR( cudaMalloc((void **)&d_weights, sizeW) );
    HANDLE_ERROR( cudaMemcpy(d_inputMaps, inputMaps, sizeIn, cudaMemcpyHostToDevice) );
    HANDLE_ERROR( cudaMemcpy(d_weights, weights, sizeW, cudaMemcpyHostToDevice) );

    // Run the kernel
    for (int Y_offset = 0; Y_offset < M; Y_offset += Mt) {
    	//printf("Y_offset = %d\n", Y_offset);
    	Nt = MIN(Nt, N - Y_offset);
    	Mt = Nt - K + 1;

	    if (K == 2) {
		    convLayerKernel<2><<<dimGrid, threadsPerBlock, sizeSharedMem>>>(d_inputMaps, d_weights, d_outputMaps, N, M, F_in, F_out, S, Nt, Mt, Y_offset);
		} else if (K == 3) {	
			convLayerKernel<3><<<dimGrid, threadsPerBlock, sizeSharedMem>>>(d_inputMaps, d_weights, d_outputMaps, N, M, F_in, F_out, S, Nt, Mt, Y_offset);
		} else if (K == 4) {	
			convLayerKernel<4><<<dimGrid, threadsPerBlock, sizeSharedMem>>>(d_inputMaps, d_weights, d_outputMaps, N, M, F_in, F_out, S, Nt, Mt, Y_offset);
		} else if (K == 5) {	
			convLayerKernel<5><<<dimGrid, threadsPerBlock, sizeSharedMem>>>(d_inputMaps, d_weights, d_outputMaps, N, M, F_in, F_out, S, Nt, Mt, Y_offset);
		} else if (K == 6) {	
			convLayerKernel<6><<<dimGrid, threadsPerBlock, sizeSharedMem>>>(d_inputMaps, d_weights, d_outputMaps, N, M, F_in, F_out, S, Nt, Mt, Y_offset);
		} else if (K == 7) {	
			convLayerKernel<7><<<dimGrid, threadsPerBlock, sizeSharedMem>>>(d_inputMaps, d_weights, d_outputMaps, N, M, F_in, F_out, S, Nt, Mt, Y_offset);
		} else if (K == 8) {	
			convLayerKernel<8><<<dimGrid, threadsPerBlock, sizeSharedMem>>>(d_inputMaps, d_weights, d_outputMaps, N, M, F_in, F_out, S, Nt, Mt, Y_offset);
		} else if (K == 9) {	
			convLayerKernel<9><<<dimGrid, threadsPerBlock, sizeSharedMem>>>(d_inputMaps, d_weights, d_outputMaps, N, M, F_in, F_out, S, Nt, Mt, Y_offset);
		} else if (K == 10) {	
			convLayerKernel<10><<<dimGrid, threadsPerBlock, sizeSharedMem>>>(d_inputMaps, d_weights, d_outputMaps, N, M, F_in, F_out, S, Nt, Mt, Y_offset);
		} else if (K == 11) {	
			convLayerKernel<11><<<dimGrid, threadsPerBlock, sizeSharedMem>>>(d_inputMaps, d_weights, d_outputMaps, N, M, F_in, F_out, S, Nt, Mt, Y_offset);
		} else if (K == 12) {	
			convLayerKernel<12><<<dimGrid, threadsPerBlock, sizeSharedMem>>>(d_inputMaps, d_weights, d_outputMaps, N, M, F_in, F_out, S, Nt, Mt, Y_offset);
		}
	}
    //copy output maps back to host
    HANDLE_ERROR( cudaMemcpy(outputMaps, d_outputMaps, sizeOut, cudaMemcpyDeviceToHost) );

    //free items in global memory
    HANDLE_ERROR( cudaFree(d_inputMaps) );     
    HANDLE_ERROR( cudaFree(d_outputMaps) ); 
    HANDLE_ERROR( cudaFree(d_weights) ); 


}




