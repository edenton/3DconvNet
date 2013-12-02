__global__ void computePartialMaps(float* inputMaps, float* outputMaps, int N, int M, int F_in, int F_out, int K, int S) ;
__global__ void sumPartialMaps(float* partialOutputMaps, float* outputMaps, int M, int F_in, int F_out, int S) ;
__host__ void convLayerRing(float* inputMaps, float* weights, float* outputMaps, int N, int M, int F_in, int F_out, int K, int S);