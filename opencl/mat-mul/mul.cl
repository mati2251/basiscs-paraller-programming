__kernel void mat_mul (const int n, __global int *A, __global int *B, __global int *C){
    int i = get_global_id(0);
    int j = get_global_id(1);
    float tmp = 0.0;
    for (int k = 0; k < n; k++){
        tmp += A[i*n + k] * B[i*n + k];
    }
    C[i*n + j] = tmp;
}