// Simple OpenCL implementation matching simple_model.py
// This implements a basic matrix-vector multiplication: output = weights * input
// Where weights is a 5x10 matrix of 0.5 values

// OpenCL Kernel
const char* kernelSource = R"CL(
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
            sum += weights[i * input_size + j] * input[j];
        }
        output[i] = sum;
    }
}
)CL";

#include <stdio.h>
#include <stdlib.h>

// Mock main function for compilation testing
// In real use, this would use OpenCL API to run the kernel
int main() {
    const int INPUT_SIZE = 10;
    const int OUTPUT_SIZE = 5;
    
    // Allocate and initialize input
    float input[INPUT_SIZE];
    float weights[OUTPUT_SIZE * INPUT_SIZE];
    float output[OUTPUT_SIZE];
    
    // Initialize weights to 0.5 (matching PyTorch model)
    for (int i = 0; i < OUTPUT_SIZE * INPUT_SIZE; i++) {
        weights[i] = 0.5f;
    }
    
    // Initialize input with some values
    for (int i = 0; i < INPUT_SIZE; i++) {
        input[i] = (float)i / 10.0f;
    }
    
    // CPU version of the kernel (for testing without OpenCL runtime)
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        float sum = 0.0f;
        for (int j = 0; j < INPUT_SIZE; j++) {
            sum += weights[i * INPUT_SIZE + j] * input[j];
        }
        output[i] = sum;
    }
    
    // Print results
    printf("OpenCL Implementation Output:\n");
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        printf("output[%d] = %f\n", i, output[i]);
    }
    
    return 0;
}
