#pragma once

#include <CL/cl.h>

struct ComputeContext
{
	cl_platform_id platform_id;
	cl_device_id device_id;

	cl_context context;
	cl_command_queue cmd_queue;
};

void checkOpenCLError(cl_int return_code, const char* file, int line);
#define CHECK(x) checkOpenCLError(x, __FILE__, __LINE__)

bool initCompute(ComputeContext& ctx, bool profiling);
void deinitCompute(ComputeContext& ctx);
cl_program loadProgram(const ComputeContext& ctx, const char* fname);
cl_kernel createKernel(cl_program program, const char* kernel_name);

extern bool verboseComputeLogging;