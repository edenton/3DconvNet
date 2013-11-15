__host__ void dotProduct(float* A, float* B, float* C, int n) ;
__host__ void convLayerCPU(float* input_maps, float* weights, float* output_maps, int N, int M, int F_in, int F_out, int K, int S);