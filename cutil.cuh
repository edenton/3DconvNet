static void HandleError( cudaError_t err,
                         const char *file,
                         int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

#define DIVUP(x, y) (((x) + (y) - 1) / (y))

#define MIN(x, y) ((x) < (y) ? (x) : (y))

#define RBINDX(i, s, n) (((i) + (s)) % (n)) // indexing into ring buffer
//#define RBINDX(i, s, n) ( (i) + (s) < n ? (i) + (s) : (i) + (s) - (n) )