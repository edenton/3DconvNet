#include <stdio.h>
#include "CPUconvLayer.cuh"
#include "cuda_conv_layer.cuh"


#define device 0



__host__ void printArray(float* A, int n) 
{
	for (int i = 0; i< n; i++) {
		printf("%d ", (int)A[i]);
	}
	printf("\n\n");

}

__host__ int arrEq(float* a, float* b, int n)
{
	int diff = 0;
	for (int i = 0; i < n; i++) {
		if (a[i] != b[i]) {
			diff += 1;
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
                    	printf("%d ", (int) map[(N*N*N*F)*s + (N*N*N)*f + (N*N)*k + (N)*j + i ]);
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
	int N = 16;
	int K = 5;
	int M = N - K + 1;
	int S = 100;
	int F_in = 64; 
	int F_out = 32;

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
						inputMaps[(N*N*N*F_in)*s + (N*N*N)*f_in + (N*N)*k + (N)*j + i ] = s + f_in;
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
		convLayer(inputMaps, weights, outputMaps, N, M, F_in, F_out, K, S);

	}
	end_t = clock();
	total_t = (float)(end_t - start_t); 
	printf("\nTotal time (in seconds) on GPU averaged over %d runs: %f \n\n", num_runs, 0.001*total_t / num_runs);

	//make sure it matches CPU version
	start_t = clock();
	for (int t = 0; t < num_runs; t++) {
		convLayerCPU(inputMaps, weights, outputMapsCPU, N, M, F_in, F_out, K, S);

	}
	end_t = clock();
	total_t = (float)(end_t - start_t); 
	printf("\nTotal time (in seconds) on CPU averaged over %d runs: %f \n\n", num_runs, 0.001*total_t / num_runs);

	

	if (arrEq(outputMaps, outputMapsCPU, M*M*M*F_out*S)) {
		printf("UH OH: outputMaps != outputMapsCPU\n");
	} else {
		printf("yay! outputMaps == outputMapsCPU\n");
	}

	//print input maps
	//printArray(outputMaps, M*M*M*F_out*S);
	//printArray(outputMapsCPU, M*M*M*F_out*S);
    //printMap(outputMaps, M, F_out, S);
	//printf("\n\n**************************\n\n");


	free(inputMaps); free(weights); free(outputMaps);

}