#include <stdio.h>
#include <assert.h>
#include <math.h>
#include "cuda_conv_layer.cuh"
#include "cutil.cuh"


texture<float> t_weights;



__global__ void convLayerKernel(float* inputMaps, float* outputMaps, int N, int M, int F_in, int F_out, int K, int S) 
{

	/* 
	(1) Determine which (x, y, z, f_out, s) location in output map this thread is
	working on. Recall: 
		G_x = M*M*M
		G_y = F_out / featuresPerThread (rounded up)
		G_z = S

		B_x = featuresPerThread
		B_y = K*K*K
	*/
	int outMapIndx = blockIdx.x; //which spatial location are we working on (in output map)?
	int f_out = blockIdx.y*blockDim.x + threadIdx.x;//which output feature are we working on?
	int internal_f_out = threadIdx.x; //which feature, within this block are we working on?
	int s = blockIdx.z; //which sample are we working on?

	//now fins specific (x, y, z) location in output map
	int out_z = (int) floor((float) outMapIndx / (M*M));
	int out_y = (int) floor((float) (outMapIndx - out_z*M*M) / M);
	int out_x = outMapIndx - out_z*M*M - out_y*M;

	/*
	(2) Determine which (i, j, k, f_out, :) location of filter this thread will be working on
	*/
	int filterIndx = threadIdx.y;
	int filt_z = floor((float) filterIndx / (K*K));
	int filt_y = floor((float) (filterIndx - filt_z*K*K) / K);
	int filt_x = filterIndx - filt_z*K*K - filt_y*K;


	/*
	(3) Declare shared variables to hold the blockDim.x 3-D tensors of size (K, K, K). And Initialize
	elements to be all zeros. Size of 	shared memory is dynamically allocated and specified when 
	kernel function is called. Size should be featuresPerThread*K*K*K*sizeof(float).
	*/
	extern __shared__ float T[];
	T[filterIndx*blockDim.x + internal_f_out] = 0;

	if (f_out < F_out) {
		/*
		(4) Iterate through input features and fill T so that:
		T(f_out, p, q, r) = \sum_{f_in} inputMaps(out_x+p, out_y+q, out_z+r, f_in)*weights(p, q, r, f_out, f_in)
		*/
		if (blockDim.y == K*K*K) {
			for (int f_in = 0; f_in < F_in; f_in++) {
				__syncthreads();
				T[filterIndx*blockDim.x + internal_f_out] += 
					inputMaps[(N*N*N*F_in)*s + (N*N*N)*f_in + (N*N)*(out_z+filt_z) + (N)*(out_y+filt_y) + out_x+filt_x]*tex1Dfetch(t_weights, (K*K*K*F_out)*f_in + (K*K*K)*f_out + (K*K)*filt_z + (K)*filt_y + filt_x);
			}
		} else if (blockDim.y == K*K) {
			for (int f_in = 0; f_in < F_in; f_in++) {
				__syncthreads();
				for (int z = 0; z < K; z++) {
					T[filterIndx*blockDim.x + internal_f_out] += 
						inputMaps[(N*N*N*F_in)*s + (N*N*N)*f_in + (N*N)*(out_z+z) + (N)*(out_y+filt_y) + out_x+filt_x]*tex1Dfetch(t_weights, (K*K*K*F_out)*f_in + (K*K*K)*f_out + (K*K)*z + (K)*filt_y + filt_x);
				}
			}

		} else {
			for (int f_in = 0; f_in < F_in; f_in++) {
				__syncthreads();
				for (int z = 0; z < K; z++) {
					for (int y = 0; y < K; y++) {
						T[filterIndx*blockDim.x + internal_f_out] += 
							inputMaps[(N*N*N*F_in)*s + (N*N*N)*f_in + (N*N)*(out_z+z) + (N)*(out_y+y) + out_x+filt_x]*tex1Dfetch(t_weights, (K*K*K*F_out)*f_in + (K*K*K)*f_out + (K*K)*z + (K)*y + filt_x);
					}
				}
			}

		}
		__syncthreads();

		/*
		(5) Sum all the elements in the blockDim.x 3-D tensors. The summed values will be the output map values.
		Summation is done using reduction pattern. f_out acts as offset to tell us where in array we shoud be working
		 */
		for (int stride = 2; stride <= exp2f((float)ceil(log2((float)blockDim.y))); stride *= 2) {
				
			int step = ceil((float) blockDim.y / stride);
			if (filterIndx < step && filterIndx + step < ceil((float)2*blockDim.y / stride)) {
				T[filterIndx*blockDim.x + internal_f_out] += T[(filterIndx+step)*blockDim.x + internal_f_out];
			}
			__syncthreads();

		}
	}

	/*
	(6) Store the value of the reduction sum in the output map
	*/
	if (f_out < F_out && filterIndx == 0) {
		outputMaps[(M*M*M*F_out)*s + (M*M*M)*f_out + (M*M)*out_z + (M)*out_y + out_x] = T[internal_f_out];
	}
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
__host__ void convLayer(float* inputMaps, float* weights, float* outputMaps, int N, int M, int F_in, int F_out, int K, int S)
{

	cudaDeviceProp devProp;
    cudaGetDeviceProperties(&devProp, 0);

    //define some things
    int featuresPerThread = 32;

    // Ensure have large enough grid size
    assert(M*M*M <= devProp.maxGridSize[0]);
    assert(DIVUP(F_out, featuresPerThread) <= devProp.maxGridSize[1]);
    assert(S <= devProp.maxGridSize[2]);
    int G_x = M*M*M;
    int G_y = DIVUP(F_out, featuresPerThread);
    int G_z = S;
    dim3 dimGrid(G_x, G_y, G_z);
     
    //determine block sizes
    int B_y;
    if (featuresPerThread*K*K*K <= devProp.maxThreadsPerBlock && sizeof(float)*featuresPerThread*K*K*K <= devProp.sharedMemPerBlock) {
    	printf("Thread block case 1\n");
    	B_y = K*K*K;
    } else if (featuresPerThread*K*K <= devProp.maxThreadsPerBlock && sizeof(float)*featuresPerThread*K*K <= devProp.sharedMemPerBlock) {
    	printf("Thread block case 2\n");
    	B_y = K*K;
    } else if (featuresPerThread*K <= devProp.maxThreadsPerBlock && sizeof(float)*featuresPerThread*K <= devProp.sharedMemPerBlock) {
    	printf("Thread block case 3\n");
    	B_y = K;
    } else {
    	printf("ERROR: block size or shared memory too small\n");
    }
    int B_x = featuresPerThread;
    assert(B_x*B_y <= devProp.maxThreadsPerBlock);
    dim3 dimBlock(B_x, B_y);

	printf("dimGrid(%d, %d, %d)\n", G_x, G_y, G_z);
    printf("dimBlock(%d, %d)\n", B_x, B_y);
    

    //copy weights to global memory and bind to texture
    int sizeW = K*K*K*F_out*F_in*sizeof(float);
    float *d_weights;
    HANDLE_ERROR( cudaMalloc( (void**)&d_weights, sizeW ) );
    HANDLE_ERROR( cudaMemcpy(d_weights, weights, sizeW, cudaMemcpyHostToDevice) );
    HANDLE_ERROR( cudaBindTexture( NULL, t_weights, d_weights, sizeW ) );

    //copy input maps to global memory
    int sizeIn = N*N*N*F_in*S*sizeof(float);
    int sizeOut = M*M*M*F_out*S*sizeof(float);
    float *d_inputMaps, *d_outputMaps;
    HANDLE_ERROR( cudaMalloc((void **)&d_inputMaps, sizeIn) );
    HANDLE_ERROR( cudaMalloc((void **)&d_outputMaps, sizeOut) );
    HANDLE_ERROR( cudaMemcpy(d_inputMaps, inputMaps, sizeIn, cudaMemcpyHostToDevice) );

    //run kernel 
    int sizeSharedMem = featuresPerThread*B_y*sizeof(float);
    convLayerKernel<<<dimGrid, dimBlock, sizeSharedMem>>>(d_inputMaps, d_outputMaps, N, M, F_in, F_out, K, S);

    //copy output maps back to host
    HANDLE_ERROR( cudaMemcpy(outputMaps, d_outputMaps, sizeOut, cudaMemcpyDeviceToHost) );

    //free items in global memory
    HANDLE_ERROR( cudaFree(d_inputMaps) ); HANDLE_ERROR( cudaFree(d_outputMaps) ); //cudaFree(c_weights);


}




