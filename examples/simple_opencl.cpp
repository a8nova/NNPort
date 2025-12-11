// Simple OpenCL implementation matching simple_model.py
// This implements a basic matrix-vector multiplication: output = weights * input
// Where weights is a 5x10 matrix of 0.5 values
// 
// This version uses REAL OpenCL API calls with full GPU logging

#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK_ERROR(err, msg) \
    if (err != CL_SUCCESS) { \
        printf("ERROR: %s (code %d)\n", msg, err); \
        return 1; \
    }

// Load kernel source from file
char* load_kernel_source(const char* filename, size_t* size) {
    FILE* fp = fopen(filename, "r");
    if (!fp) {
        printf("ERROR: Could not open %s\n", filename);
        return NULL;
    }
    
    fseek(fp, 0, SEEK_END);
    *size = ftell(fp);
    rewind(fp);
    
    char* kernel_source = (char*)malloc(*size + 1);
    if (!kernel_source) {
        fclose(fp);
        return NULL;
    }
    
    fread(kernel_source, 1, *size, fp);
    kernel_source[*size] = '\0';
    fclose(fp);
    
    return kernel_source;
}

int main() {
    const int INPUT_SIZE = 10;
    const int OUTPUT_SIZE = 5;
    cl_int err;
    
    printf("=== OpenCL Initialization ===\n");
    
    // Get platforms
    cl_uint num_platforms;
    err = clGetPlatformIDs(0, NULL, &num_platforms);
    CHECK_ERROR(err, "Failed to get platform count");
    printf("Found %u OpenCL platform(s)\n", num_platforms);
    
    cl_platform_id* platforms = (cl_platform_id*)malloc(sizeof(cl_platform_id) * num_platforms);
    err = clGetPlatformIDs(num_platforms, platforms, NULL);
    CHECK_ERROR(err, "Failed to get platforms");
    
    // Get platform name
    char platform_name[128];
    clGetPlatformInfo(platforms[0], CL_PLATFORM_NAME, sizeof(platform_name), platform_name, NULL);
    printf("Using platform: %s\n", platform_name);
    
    // Get devices
    cl_uint num_devices;
    err = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, 0, NULL, &num_devices);
    if (err != CL_SUCCESS) {
        printf("No GPU found, trying all devices...\n");
        err = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);
    }
    CHECK_ERROR(err, "Failed to get device count");
    printf("Found %u device(s)\n", num_devices);
    
    cl_device_id* devices = (cl_device_id*)malloc(sizeof(cl_device_id) * num_devices);
    err = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, num_devices, devices, NULL);
    if (err != CL_SUCCESS) {
        err = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_ALL, num_devices, devices, NULL);
    }
    CHECK_ERROR(err, "Failed to get devices");
    
    // Get device name
    char device_name[128];
    clGetDeviceInfo(devices[0], CL_DEVICE_NAME, sizeof(device_name), device_name, NULL);
    printf("Using device: %s\n", device_name);
    
    // Create context
    cl_context context = clCreateContext(NULL, 1, devices, NULL, NULL, &err);
    CHECK_ERROR(err, "Failed to create context");
    printf("✓ Context created\n");
    
    // Create command queue
    cl_command_queue queue = clCreateCommandQueue(context, devices[0], 0, &err);
    CHECK_ERROR(err, "Failed to create command queue");
    printf("✓ Command queue created\n");
    
    // Load kernel source
    printf("\n=== Loading Kernel ===\n");
    size_t kernel_size;
    char* kernel_source = load_kernel_source("kernel.cl", &kernel_size);
    if (!kernel_source) {
        printf("Failed to load kernel source\n");
        return 1;
    }
    printf("Loaded kernel (%zu bytes)\n", kernel_size);
    
    // Create program
    cl_program program = clCreateProgramWithSource(context, 1, (const char**)&kernel_source, &kernel_size, &err);
    CHECK_ERROR(err, "Failed to create program");
    
    // Build program
    printf("Building kernel...\n");
    err = clBuildProgram(program, 1, devices, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        printf("ERROR: Kernel build failed (code %d)\n", err);
        char build_log[4096];
        clGetProgramBuildInfo(program, devices[0], CL_PROGRAM_BUILD_LOG, sizeof(build_log), build_log, NULL);
        printf("Build log:\n%s\n", build_log);
        return 1;
    }
    printf("✓ Kernel compiled successfully\n");
    
    // Create kernel
    cl_kernel kernel = clCreateKernel(program, "linear_layer", &err);
    CHECK_ERROR(err, "Failed to create kernel");
    printf("✓ Kernel 'linear_layer' created\n");
    
    // Read input
    printf("\n=== Processing Data ===\n");
    FILE* input_fp = fopen("input.bin", "rb");
    if (!input_fp) {
        printf("ERROR: Could not open input.bin\n");
        return 1;
    }
    
    float input[INPUT_SIZE];
    fread(input, sizeof(float), INPUT_SIZE, input_fp);
    fclose(input_fp);
    printf("Read %d input values\n", INPUT_SIZE);
    
    // Initialize weights
    float weights[OUTPUT_SIZE * INPUT_SIZE];
    for (int i = 0; i < OUTPUT_SIZE * INPUT_SIZE; i++) {
        weights[i] = 0.5f;
    }
    
    float output[OUTPUT_SIZE];
    
    // Create buffers
    cl_mem input_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                                      sizeof(float) * INPUT_SIZE, input, &err);
    CHECK_ERROR(err, "Failed to create input buffer");
    
    cl_mem weights_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                        sizeof(float) * OUTPUT_SIZE * INPUT_SIZE, weights, &err);
    CHECK_ERROR(err, "Failed to create weights buffer");
    
    cl_mem output_buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                       sizeof(float) * OUTPUT_SIZE, NULL, &err);
    CHECK_ERROR(err, "Failed to create output buffer");
    
    printf("✓ GPU buffers allocated\n");
    
    // Set kernel arguments
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &input_buf);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &weights_buf);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &output_buf);
    err |= clSetKernelArg(kernel, 3, sizeof(int), &INPUT_SIZE);
    CHECK_ERROR(err, "Failed to set kernel arguments");
    
    // Execute kernel
    printf("Executing kernel on GPU...\n");
    size_t global_size = OUTPUT_SIZE;
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size, NULL, 0, NULL, NULL);
    CHECK_ERROR(err, "Failed to execute kernel");
    
    // Wait for completion
    err = clFinish(queue);
    CHECK_ERROR(err, "Failed to wait for kernel completion");
    printf("✓ Kernel execution complete\n");
    
    // Read results
    err = clEnqueueReadBuffer(queue, output_buf, CL_TRUE, 0, sizeof(float) * OUTPUT_SIZE, output, 0, NULL, NULL);
    CHECK_ERROR(err, "Failed to read output buffer");
    printf("✓ Results copied from GPU\n");
    
    // Write output
    printf("\n=== Results ===\n");
    FILE* output_fp = fopen("output.bin", "wb");
    if (!output_fp) {
        printf("ERROR: Could not open output.bin for writing\n");
        return 1;
    }
    fwrite(output, sizeof(float), OUTPUT_SIZE, output_fp);
    fclose(output_fp);
    
    // Print results
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        printf("Output[%d]: %f\n", i, output[i]);
    }
    
    // Cleanup
    clReleaseMemObject(input_buf);
    clReleaseMemObject(weights_buf);
    clReleaseMemObject(output_buf);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    free(kernel_source);
    free(platforms);
    free(devices);
    
    printf("\n✓ OpenCL execution complete!\n");
    return 0;
}
