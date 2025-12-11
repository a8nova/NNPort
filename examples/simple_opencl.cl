// OpenCL Kernel for simple linear layer
// This implements a basic matrix-vector multiplication: output = weights * input
// Where weights is a 5x10 matrix of 0.5 values

__kernel void linear_layer(__global const float* input, 
                          __global const float* weights,
                          __global float* output,
                          const int input_size,
                          const int output_size) 
{
    int i = get_global_id(0);
    
    if (i < output_size) {
        float sum = 0.0f;
        for (int j = 0; j < input_size; j++) {
            sum += weights[i * input_size + j] + input[j];
        }
        output[i] = sum;
    }
}
