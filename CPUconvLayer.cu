#include <stdio.h>
#include <assert.h>
#include <math.h>
#include "CPUconvLayer.cuh"

__host__ void dotProduct(float* A, float* B, float* C, int n) 
{
    for (int i = 0; i < n; i++) {
        *C += A[i] * B[i];
    }
}

/*
input_maps:     (N, N, N, F_in, S) input feature maps
weights:        (K, K, K, F_out, F_in) filter weights
output_maps:    (M, M, M, F_out, S) output feature maps

N:      input feature map dimension (same in x, y, z direction)
M:      output map dimension
F_in:   number of channels in input layer
F_out:  number of channels in output layer
K:      filter size (same in x, y, z direction)
S:      batch size

access input_maps[i, j, k, f_in, s] by input_maps[(N*N*N*F_in)*s + (N*N*N)*f_in + (N*N)*k + (N)*j + i ]
access weights[i, j, k, f_out, f_in] by weights[(K*K*K*F_out)*f_in + (K*K*K)*f_out + (K*K)*k + (K)*j + i ]
access output_maps[i, j, k, f_out, s] by input_maps[(M*M*M*F_out)*s + (N*N*N)*f_out + (N*N)*k + (N)*j + i ]
*/
void convLayerCPU(float* input_maps, float* weights, float* output_maps, int N, int M, int F_in, int F_out, int K, int S)
{
    int size = K*K*K * sizeof(float);
    float *filter = (float *) malloc(size);
    float *sig = (float *) malloc(size);

    // iterate through samples
    for (int s = 0; s < S; s++) {
        //iterate through input feature maps
        for (int f_out = 0; f_out < F_out; f_out++) {
            //iterate through output feature maps
            for (int f_in = 0; f_in < F_in; f_in++) {
                
                //set filter for this f_out - f_in combo (filter will be same for all i, j, k in this output map section)
                for (int kx = 0; kx < K; kx++) {
                    for (int ky = 0; ky < K; ky++) {
                        for (int kz = 0; kz < K; kz++) {
                            filter[(K*K)*kz + (K)*ky + kx] = weights[(K*K*K*F_out)*f_in + (K*K*K)*f_out + (K*K)*kz + (K)*ky + kx];

                        }
                    }
                } //end filter setup loop

                

                //go through all spatial locations of output map
                for (int i = 0; i < M; i++) {
                    for (int j = 0; j < M; j++) {
                        for (int k = 0; k < M; k++) {
                            //set input signal for this spatial location
                            for (int a = 0; a < K; a++) {
                                for (int b = 0; b< K; b++) {
                                    for (int c = 0; c < K; c++) {
                                        //set input signal for this spatial location
                                        sig[(K*K)*c + (K)*b + a] = input_maps[(N*N*N*F_in)*s + (N*N*N)*f_in + (N*N)*(k+c) + (N)*(j+b) + (i+a)];                                     
                                    }
                                }
                            } //end signal setup loop

                            //finally compute value of outtput map
                            dotProduct(filter, sig, &output_maps[(M*M*M*F_out)*s + (M*M*M)*f_out + (M*M)*k + (M)*j + i], K*K*K);


                        }
                    }
                }



            

            } //end f_in loop
        } //end f_out loop
    } //end samples loop

}