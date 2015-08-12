#define PROGRAM_FILE "ic_kernel.cl"
#define KERNEL_FUNC "ic_compute"
#include<stdio.h>
#include<stdlib.h>
#include<sys/types.h>
#ifdef MAC
#include<OpenCL/cl.h>
#else
#include<CL/cl.h>
#endif

#define N 1000
#define T 1000
#define Pr 0.5
#define I 1.0
#define Kappa 2.0
#define Tau 100.0

void initialize(float *u, float *w, float *z, float *result)
{
  int i, j;

  u = (float *)malloc(N*sizeof(float));
  w = (float *)malloc(N*N*sizeof(float));
  z = (float *)malloc(N*sizeof(float));
  result = (float *)malloc(T*N*sizeof(float));

  for(i = 0; i < N; i++){
    u[i] = I;
    z[i] = 0;
  }

  srand(23);

  for(i = 0; i < N; i++){
    for(j = 0; j < N; j++){
      if (i == j){
	w[j+N*i] = 0;
      }else{
	if ((float)rand()/(float)RAND_MAX < Pr){
	  w[j+N*i] = 1;
	}else{
	  w[j+N*i] = 0;
	}
      }
    }
  }
}
void finalize(float *u, float *w, float *z, float *result)
{
  free(u);
  free(w);
  free(z);
  free(result);
}
void output(const char *prefix, const float *result)
{
  FILE *f;
  int t, i;
  char fn[1024];

  sprintf(fn, "%s.r", prefix);
  f = fopen(fn, "w");
  for(t = 0; t < T; t++){
    for(i = 0; i < N; i++){
      if (result[i+N*t] > 0){
	fprintf(f, "%d %d\n", t, i);
      }
    }
  }

  fclose(f);
}

int main(void)
{
  cl_platform_id platform;
  cl_device_id device;
  cl_context context;
  cl_command_queue queue;
  cl_int i, j, err;

  cl_program program;
  FILE *program_handle;
  char *program_buffer, *program_log;
  size_t program_size, log_size;
  cl_kernel kernel;
  size_t work_units_per_kernel;

  float *w, *z, *u, *result;
  cl_mem w_buff, z_buff, u_buff, res_buff;

  ////initialize(u, w, z, result);

  w = (float *)malloc(N*N*sizeof(float));
  u = (float *)malloc(N*sizeof(float));
  z = (float *)malloc(N*sizeof(float));
  result = (float *)malloc(T*N*sizeof(float));

  for(i = 0; i < N; i++){
    u[i] = I;
    z[i] = 0;
  }

  srand(23);

  for(i = 0; i < N; i++){
    for(j = 0; j < N; j++){
      if (i == j){
	w[j+N*i] = 0;
      }else{
	if ((float)rand()/(float)RAND_MAX < Pr){
	  w[j+N*i] = 1;
	}else{
	  w[j+N*i] = 0;
	}
      }
    }
  }

  clGetPlatformIDs(1, &platform, NULL);
  clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &device, NULL);
  context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);

  program_handle = fopen(PROGRAM_FILE, "r");
  fseek(program_handle, 0, SEEK_END);
  program_size = ftell(program_handle);
  rewind(program_handle);
  program_buffer = (char *)malloc(program_size + 1);
  program_buffer[program_size] = '\0';
  fread(program_buffer, sizeof(char), program_size, program_handle);
  fclose(program_handle);

  program = clCreateProgramWithSource(context, 1, (const char **)&program_buffer, &program_size, &err);
  free(program_buffer);
  clBuildProgram(program, 0, NULL, NULL, NULL, NULL);

  kernel = clCreateKernel(program, KERNEL_FUNC, &err);
  queue = clCreateCommandQueue(context, device, 0, &err);

  w_buff = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float)*N*N, w, &err);
  u_buff = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float)*N, u, &err);
  z_buff = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float)*N, z, &err);
  res_buff = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float)*N*T, NULL, &err);

  work_units_per_kernel = N;
  for(int t = 0; t < T; t++){
    clSetKernelArg(kernel, 0, sizeof(int), &t);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &w_buff);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &u_buff);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), &z_buff);
    clSetKernelArg(kernel, 4, sizeof(cl_mem), &res_buff);
    clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &work_units_per_kernel, NULL, 0, NULL, NULL);
  }

  clEnqueueReadBuffer(queue, res_buff, CL_TRUE, 0, sizeof(float)*N*T, result, 0, NULL, NULL);

  output("cl.out", result);

  ////finalize(u, w, z, result);

  free(w);
  free(u);
  free(z);
  free(result);

  clReleaseMemObject(u_buff);
  clReleaseMemObject(w_buff);
  clReleaseMemObject(z_buff);
  clReleaseMemObject(res_buff);
  clReleaseKernel(kernel);
  clReleaseCommandQueue(queue);
  clReleaseProgram(program);
  clReleaseContext(context);

  return 0;
}
