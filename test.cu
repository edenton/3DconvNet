#include <stdio.h>
#include <stdlib.h>
#include "CPUconvLayer.cuh"
#include "cuda_conv_layer_ring.cuh"
#include "cuda_conv_layer.cuh"


#define device 0



__host__ void printArray(float* A, int n) 
{
	for (int i = 0; i< n; i++) {
		printf("%f ", A[i]);
	}
	printf("\n\n");

}

// Make sure relative error is small
__host__ int arrEq(float* a, float* b, int n)
{
	int diff = 0;
	for (int i = 0; i < n; i++) {
		if ( abs(a[i] - b[i]) / abs(b[i]) > pow((double) 10, (double)-5)) {
			diff += 1;
			//printf("a[%d] != b[%d]: |%f = %f| / %f = %f \n", i, i, a[i], b[i], abs(a[i] - b[i]) / abs(b[i]));
		}
	}
	return diff;
}

void printMap(float* map, int N, int F, int S) {
    for (int s = 0; s < S; s++) {
		printf("\n ====== sample %d =========\n", s);
        for (int f = 0; f < F; f++) {
    		printf("\n --- feature %d -----\n", f);
            for (int k = 0; k < N; k++) {
                for (int j = 0; j < N; j++) {
                    for (int i = 0; i < N; i++) {
                    	printf("%f ", map[(N*N*N*F)*s + (N*N*N)*f + (N*N)*k + (N)*j + i ]);
					}
					printf("\n");
                }
				printf("\n");
            }
        }
    }
}


int main(int argc, char **argv)
{ 
	cudaSetDevice(device);
	int N = 32;
	int K = 3;
	int M = N - K + 1;
	int S = 1;
	int F_in = 3; 
	int F_out = 16;
	srand(time(NULL));
	float *inputMaps = (float *) malloc(N*N*N*F_in*S*sizeof(float));
	float *outputMaps = (float *) malloc(M*M*M*F_out*S*sizeof(float));
	float *weights = (float *) malloc(K*K*K*F_in*F_out*sizeof(float));
	float *outputMapsCPU = (float *) malloc(M*M*M*F_out*S*sizeof(float));

	//fill the input maps
	for (int s = 0; s < S; s++) {
		for (int f_in = 0; f_in < F_in; f_in++) {
			for (int i = 0; i < N; i++) {
				for (int j = 0; j < N; j++) {
					for (int k = 0; k < N; k++) {
						inputMaps[(N*N*N*F_in)*s + (N*N*N)*f_in + (N*N)*k + (N)*j + i ] = (float) (double)rand()/(double)RAND_MAX ;
					}
				}
			}
		}
	}

	//fill the output maps
	for (int s = 0; s < S; s++) {
		for (int f_out = 0; f_out < F_out; f_out++) {
			for (int i = 0; i < M; i++) {
				for (int j = 0; j < M; j++) {
					for (int k = 0; k < M; k++) {
						outputMaps[(M*M*M*F_out)*s + (M*M*M)*f_out + (M*M)*k + (M)*j + i ] = 0;
						outputMapsCPU[(M*M*M*F_out)*s + (M*M*M)*f_out + (M*M)*k + (M)*j + i ] = 0;
					}
				}
			}
		}
	}

	//fill the weights
	for (int f_out = 0; f_out < F_out; f_out++) {
		for (int f_in = 0; f_in < F_in; f_in++) {
			for (int i = 0; i < K; i++) {
				for (int j = 0; j < K; j++) {
					for (int k = 0; k < K; k++) {
						weights[(K*K*K*F_out)*f_in + (K*K*K)*f_out + (K*K)*k + (K)*j + i ] = 1;
					}
				}
			}
		}
	}

	//do convolution
	clock_t start_t, end_t;
	double total_t;
	int num_runs = 1;
	start_t = clock();
	for (int t = 0; t < num_runs; t++) {
		convLayerRing(inputMaps, weights, outputMaps, N, M, F_in, F_out, K, S);

	}
	end_t = clock();
	total_t = (float)(end_t - start_t) / (CLOCKS_PER_SEC*num_runs); 
	printf("\nTotal time (in seconds) on GPU averaged over %d runs: %f \n", num_runs, total_t);

	//make sure it matches CPU version
	start_t = clock();
	for (int t = 0; t < num_runs; t++) {
		convLayerCPU(inputMaps, weights, outputMapsCPU, N, M, F_in, F_out, K, S);

	}
	end_t = clock();
	total_t = (float)(end_t - start_t) / (CLOCKS_PER_SEC*num_runs); 
	printf("\nTotal time (in seconds) on CPU averaged over %d runs: %f \n\n", num_runs, total_t);

	

	if (arrEq(outputMaps, outputMapsCPU, M*M*M*F_out*S)) {
		printf("%d: outputMaps != outputMapsCPU\n\n", N);
	} else {
		printf("%d: 1 outputMaps == outputMapsCPU\n\n", N);
	}

	//print input maps
	/*
	printArray(outputMaps, 20);//M*M*M*F_out*S);
	printArray(outputMapsCPU, 20);//M*M*M*F_out*S);


  	printMap(outputMaps, M, F_out, S);
	printMap(outputMapsCPU, M, F_out, S);
	*/


	free(inputMaps); free(weights); free(outputMaps); free(outputMapsCPU);

}