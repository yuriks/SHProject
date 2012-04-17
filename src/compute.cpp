#include <CL/cl.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <numeric>
#include <cmath>

#include "compute.hpp"

void checkOpenCLError(cl_int return_code, const char* file, int line)
{
	if (return_code == CL_SUCCESS)
		return;

	const char* enum_name;
	switch (return_code) {
#define ENUM_ENTRY(x) case x: enum_name = #x; break;
	ENUM_ENTRY(CL_SUCCESS                                  )
	ENUM_ENTRY(CL_DEVICE_NOT_FOUND                         )
	ENUM_ENTRY(CL_DEVICE_NOT_AVAILABLE                     )
	ENUM_ENTRY(CL_COMPILER_NOT_AVAILABLE                   )
	ENUM_ENTRY(CL_MEM_OBJECT_ALLOCATION_FAILURE            )
	ENUM_ENTRY(CL_OUT_OF_RESOURCES                         )
	ENUM_ENTRY(CL_OUT_OF_HOST_MEMORY                       )
	ENUM_ENTRY(CL_PROFILING_INFO_NOT_AVAILABLE             )
	ENUM_ENTRY(CL_MEM_COPY_OVERLAP                         )
	ENUM_ENTRY(CL_IMAGE_FORMAT_MISMATCH                    )
	ENUM_ENTRY(CL_IMAGE_FORMAT_NOT_SUPPORTED               )
	ENUM_ENTRY(CL_BUILD_PROGRAM_FAILURE                    )
	ENUM_ENTRY(CL_MAP_FAILURE                              )
	ENUM_ENTRY(CL_MISALIGNED_SUB_BUFFER_OFFSET             )
	ENUM_ENTRY(CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST)

	ENUM_ENTRY(CL_INVALID_VALUE                            )
	ENUM_ENTRY(CL_INVALID_DEVICE_TYPE                      )
	ENUM_ENTRY(CL_INVALID_PLATFORM                         )
	ENUM_ENTRY(CL_INVALID_DEVICE                           )
	ENUM_ENTRY(CL_INVALID_CONTEXT                          )
	ENUM_ENTRY(CL_INVALID_QUEUE_PROPERTIES                 )
	ENUM_ENTRY(CL_INVALID_COMMAND_QUEUE                    )
	ENUM_ENTRY(CL_INVALID_HOST_PTR                         )
	ENUM_ENTRY(CL_INVALID_MEM_OBJECT                       )
	ENUM_ENTRY(CL_INVALID_IMAGE_FORMAT_DESCRIPTOR          )
	ENUM_ENTRY(CL_INVALID_IMAGE_SIZE                       )
	ENUM_ENTRY(CL_INVALID_SAMPLER                          )
	ENUM_ENTRY(CL_INVALID_BINARY                           )
	ENUM_ENTRY(CL_INVALID_BUILD_OPTIONS                    )
	ENUM_ENTRY(CL_INVALID_PROGRAM                          )
	ENUM_ENTRY(CL_INVALID_PROGRAM_EXECUTABLE               )
	ENUM_ENTRY(CL_INVALID_KERNEL_NAME                      )
	ENUM_ENTRY(CL_INVALID_KERNEL_DEFINITION                )
	ENUM_ENTRY(CL_INVALID_KERNEL                           )
	ENUM_ENTRY(CL_INVALID_ARG_INDEX                        )
	ENUM_ENTRY(CL_INVALID_ARG_VALUE                        )
	ENUM_ENTRY(CL_INVALID_ARG_SIZE                         )
	ENUM_ENTRY(CL_INVALID_KERNEL_ARGS                      )
	ENUM_ENTRY(CL_INVALID_WORK_DIMENSION                   )
	ENUM_ENTRY(CL_INVALID_WORK_GROUP_SIZE                  )
	ENUM_ENTRY(CL_INVALID_WORK_ITEM_SIZE                   )
	ENUM_ENTRY(CL_INVALID_GLOBAL_OFFSET                    )
	ENUM_ENTRY(CL_INVALID_EVENT_WAIT_LIST                  )
	ENUM_ENTRY(CL_INVALID_EVENT                            )
	ENUM_ENTRY(CL_INVALID_OPERATION                        )
	ENUM_ENTRY(CL_INVALID_GL_OBJECT                        )
	ENUM_ENTRY(CL_INVALID_BUFFER_SIZE                      )
	ENUM_ENTRY(CL_INVALID_MIP_LEVEL                        )
	ENUM_ENTRY(CL_INVALID_GLOBAL_WORK_SIZE                 )
	ENUM_ENTRY(CL_INVALID_PROPERTY                         )
#undef ENUM_ENTRY
	default: enum_name = "Unknown error"; break;
	}

	std::cerr << file << ':' << line << ": OpenCL returned error " << enum_name << std::endl;
}

bool enumerateDevices(cl_platform_id& platform_id, cl_device_id& device_id)
{
	bool found_device = false;

	cl_platform_id platform_ids[8];
	cl_uint num_platforms;

	CHECK(clGetPlatformIDs(8, platform_ids, &num_platforms));
	std::cout << "Found " << num_platforms << " platforms:\n";

	for (unsigned int i = 0; i < num_platforms; ++i) {
		char vendor_buffer[64];
		char name_buffer[64];
		char version_buffer[64];

		CHECK(clGetPlatformInfo(platform_ids[i], CL_PLATFORM_VENDOR,  sizeof(vendor_buffer),  static_cast<void*>(vendor_buffer),  nullptr));
		CHECK(clGetPlatformInfo(platform_ids[i], CL_PLATFORM_NAME,    sizeof(name_buffer),    static_cast<void*>(name_buffer),    nullptr));
		CHECK(clGetPlatformInfo(platform_ids[i], CL_PLATFORM_VERSION, sizeof(version_buffer), static_cast<void*>(version_buffer), nullptr));
		std::cout << "    " << vendor_buffer << " - " << name_buffer << " - " << version_buffer << '\n';

		cl_device_id device_ids[8];
		cl_uint num_devices;

		CHECK(clGetDeviceIDs(platform_ids[i], CL_DEVICE_TYPE_ALL, 8, device_ids, &num_devices));

		for (unsigned int j = 0; j < num_devices; ++j) {
			char vendor_buffer[64];
			char name_buffer[64];

			CHECK(clGetDeviceInfo(device_ids[j], CL_DEVICE_VENDOR, sizeof(vendor_buffer), static_cast<void*>(vendor_buffer), nullptr));
			CHECK(clGetDeviceInfo(device_ids[j], CL_DEVICE_NAME,   sizeof(name_buffer),   static_cast<void*>(name_buffer),   nullptr));
			std::cout << "        " << vendor_buffer << " - " << name_buffer << '\n';

			cl_device_type device_type;
			CHECK(clGetDeviceInfo(device_ids[j], CL_DEVICE_TYPE,   sizeof(device_type),   static_cast<void*>(&device_type),  nullptr));

			if (device_type == CL_DEVICE_TYPE_GPU) {
				platform_id = platform_ids[i];
				device_id = device_ids[j];
				found_device = true;
			}
		}
	}

	if (found_device) {
		char name_buffer[64];
		char version_buffer[64];

		CHECK(clGetDeviceInfo(device_id, CL_DEVICE_NAME,    sizeof(name_buffer),    static_cast<void*>(name_buffer),    nullptr));
		CHECK(clGetDeviceInfo(device_id, CL_DEVICE_VERSION, sizeof(version_buffer), static_cast<void*>(version_buffer), nullptr));
		std::cout << "Using device " << name_buffer << " - " << version_buffer << '\n';
	} else {
		std::cout << "No appropriate device found. :(\n";
	}

	return found_device;
}

cl_context createContext(cl_platform_id platform_id, cl_device_id device_id)
{
	cl_context_properties context_properties[] = {
		CL_CONTEXT_PLATFORM,
		(cl_context_properties)platform_id,
		0
	};

	std::cout << "Creating context...\n";

	cl_int error_code;
	cl_context context = clCreateContext(context_properties, 1, &device_id, nullptr, nullptr, &error_code);
	CHECK(error_code);

	return context;
}

cl_command_queue createCommandQueue(cl_context context, cl_device_id device_id)
{
	std::cout << "Creating command queue...\n";

	cl_int error_code;
	cl_command_queue cmd_queue = clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, &error_code);
	CHECK(error_code);

	return cmd_queue;
}

cl_program loadProgramFromFile(cl_context context, const char* fname)
{
	std::ifstream f(fname);

	if (!f)
		return nullptr;

	std::vector<char> str;

	f.seekg(0, std::ios::end);
	str.reserve((unsigned int)f.tellg() + 1);
	f.seekg(0);

	str.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
	str.push_back('\0');

	const char* str_ptr = str.data();

	cl_int error_code;
	cl_program program = clCreateProgramWithSource(context, 1, &str_ptr, nullptr, &error_code);
	CHECK(error_code);

	return program;
}

bool checkProgramLog(cl_program program, cl_device_id device_id)
{
	cl_build_status status;
	CHECK(clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_STATUS, sizeof(status), &status, nullptr));

	if (status == CL_BUILD_SUCCESS) {
		std::cout << "Program built successfully!\n";
		return true;
	} else if (status == CL_BUILD_ERROR) {
		std::cout << "Program failed to build:\n";

		size_t log_size;
		CHECK(clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size));

		std::vector<char> log_buffer(log_size);
		CHECK(clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, log_buffer.size(), log_buffer.data(), nullptr));

		std::cout << log_buffer.data();
	} else if (status == CL_BUILD_NONE) {
		std::cout << "No build done.\n";
	} else if (status == CL_BUILD_IN_PROGRESS) {
		std::cout << "Build is still in progress.\n";
	}

	return false;
}

cl_kernel createKernel(cl_program program, const char* kernel_name) {
	std::cout << "Creating \"" << kernel_name << "\" kernel...\n";

	cl_int error_code;
	cl_kernel kernel = clCreateKernel(program, kernel_name, &error_code); CHECK(error_code);
	return kernel;
}

void fillWithRandomData(std::vector<float>& v)
{
	for (size_t i = 0; i < v.size(); ++i) {
		v[i] = rand() / (float)RAND_MAX * 10.0f;
	}
}

// GPU floating point precision is pretty horrible :P
// Thanks to Bruce Dawson (http://bit.ly/xIW2Mf)
bool compareFloat(float a, float b, float max_rel_diff = FLT_EPSILON*100.f)
{
    // Calculate the difference.
    float diff = std::fabs(a - b);
    a = std::fabs(a);
    b = std::fabs(b);
    // Find the largest
    float largest = (b > a) ? b : a;

    return diff <= largest * max_rel_diff;
}

bool compareResults(float* v_a, float* v_b, size_t size)
{
	for (size_t i = 0; i < size; ++i) {
		if (!compareFloat(v_a[i], v_b[i])) {
			std::cout << i << '\n';
			return false;
		}
	}

	return true;
}

void printTimingStats(const double* timing_samples, int n)
{
	double total_ms = std::accumulate(timing_samples, timing_samples + n, 0.0);
	double mean_ms = total_ms / n;

	double sqr_mean_ms = std::accumulate(timing_samples, timing_samples + n, 0.0, [](double a, double b) { return a + b*b; });
	sqr_mean_ms /= n;

	double std_dev_ms = std::sqrt(sqr_mean_ms - mean_ms*mean_ms);

	std::cout << "Timing stats:\n" <<
		"    Total: "     << total_ms    << "ms\n" <<
		"    Mean: "      << mean_ms     << "ms\n" <<
		"    Std. dev.: " << std_dev_ms  << "ms\n";
}

bool verboseComputeLogging = false;

bool initCompute(ComputeContext& ctx)
{
	if (!enumerateDevices(ctx.platform_id, ctx.device_id))
		return false;

	ctx.context = createContext(ctx.platform_id, ctx.device_id);
	ctx.cmd_queue = createCommandQueue(ctx.context, ctx.device_id);

	return true;
}

void deinitCompute(ComputeContext& ctx)
{
	std::cout << "Releasing command queue...\n";
	CHECK(clReleaseCommandQueue(ctx.cmd_queue));

	std::cout << "Releasing context...\n";
	CHECK(clReleaseContext(ctx.context));
}

cl_program loadProgram(const ComputeContext& ctx, const char* fname)
{
	std::cout << "Loading program from " << fname << '\n';
	cl_program program = loadProgramFromFile(ctx.context, fname);

	std::cout << "Building program...\n";
	CHECK(clBuildProgram(program, 0, nullptr, "-Werror -cl-mad-enable -cl-fast-relaxed-math", nullptr, nullptr));
	if (!checkProgramLog(program, ctx.device_id)) {
		return nullptr;
	}

	return program;
}

/*
{
	cl_kernel sum_kernel = createKernel(program, "sum");
	cl_kernel reduce_pass1_kernel = createKernel(program, "reduce_pass1");
	cl_kernel reduce_pass2_kernel = createKernel(program, "reduce_pass2");

	std::cout << "Allocating host buffers...\n";

	static const size_t SUM_DATA_SIZE = 256 * 1024;
	static const size_t SUM_GROUP_SIZE = 128;
	static const size_t REDUCTION_WORK_ITEMS = 4 * 1024;
	static const size_t REDUCTION_WORK_GROUP_SIZE = 512; // Max. supported by 9600GT
	static const size_t PARTIAL_SUMS_SIZE = REDUCTION_WORK_ITEMS / REDUCTION_WORK_GROUP_SIZE;

	Array<float> src_a(SUM_DATA_SIZE);
	Array<float> src_b(SUM_DATA_SIZE);

	srand(0);
	fillWithRandomData(src_a);
	fillWithRandomData(src_b);

	std::cout << "Creating OpenCL buffers...\n";

	cl_mem src_a_buf     = clCreateBuffer(context, CL_MEM_READ_ONLY,  src_a.size        * sizeof(float), nullptr, &error_code); CHECK(error_code);
	cl_mem src_b_buf     = clCreateBuffer(context, CL_MEM_READ_ONLY,  src_b.size        * sizeof(float), nullptr, &error_code); CHECK(error_code);
	cl_mem dst_buf       = clCreateBuffer(context, CL_MEM_READ_WRITE, SUM_DATA_SIZE     * sizeof(float), nullptr, &error_code); CHECK(error_code);
	cl_mem sums_buf      = clCreateBuffer(context, CL_MEM_READ_WRITE, PARTIAL_SUMS_SIZE * sizeof(float), nullptr, &error_code); CHECK(error_code);
	cl_mem final_sum_buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float),                     nullptr, &error_code); CHECK(error_code);

	std::cout << "Setting kernel arguments...\n";
	CHECK(clSetKernelArg(sum_kernel, 0, sizeof(src_a_buf), &src_a_buf));
	CHECK(clSetKernelArg(sum_kernel, 1, sizeof(src_b_buf), &src_b_buf));
	CHECK(clSetKernelArg(sum_kernel, 2, sizeof(dst_buf),   &dst_buf));

	CHECK(clSetKernelArg(reduce_pass1_kernel, 0, sizeof(dst_buf),  &dst_buf));
	CHECK(clSetKernelArg(reduce_pass1_kernel, 1, sizeof(sums_buf), &sums_buf));
	CHECK(clSetKernelArg(reduce_pass1_kernel, 2, REDUCTION_WORK_GROUP_SIZE * sizeof(float), nullptr));
	CHECK(clSetKernelArg(reduce_pass1_kernel, 3, sizeof(size_t),   &SUM_DATA_SIZE));

	CHECK(clSetKernelArg(reduce_pass2_kernel, 0, sizeof(sums_buf),      &sums_buf));
	CHECK(clSetKernelArg(reduce_pass2_kernel, 1, sizeof(final_sum_buf), &final_sum_buf));
	CHECK(clSetKernelArg(reduce_pass2_kernel, 2, sizeof(size_t),        &PARTIAL_SUMS_SIZE));

	static const int NUM_TIMING_SAMPLES = 4096;
	double timing_samples[NUM_TIMING_SAMPLES];

	float cpu_result;
	Array<float> dst_reference(SUM_DATA_SIZE);
	std::cout << "\nRunning CPU algorithm...\n";
	for (int run = 0; run < NUM_TIMING_SAMPLES; ++run) {
		startPerfTimer();

		sumKernelCpu(dst_reference.size, src_a, src_b, dst_reference);
		cpu_result = reduceKernelCpu(dst_reference.size, dst_reference);

		timing_samples[run] = stopPerfTimer();
	}
	printTimingStats(timing_samples, NUM_TIMING_SAMPLES);
	std::cout << "Done!\n\n";
	dst_reference.free();

	std::cout << "Copying from host to OpenCL buffers...\n";
	CHECK(clFinish(cmd_queue));
	CHECK(clEnqueueWriteBuffer(cmd_queue, src_a_buf, CL_FALSE, 0, src_a.size * sizeof(float), src_a, 0, nullptr, nullptr));
	CHECK(clEnqueueWriteBuffer(cmd_queue, src_b_buf, CL_FALSE, 0, src_b.size * sizeof(float), src_b, 0, nullptr, nullptr));
	CHECK(clFinish(cmd_queue));

	std::cout << "Executing OpenCL kernels...\n";
	for (int run = 0; run < NUM_TIMING_SAMPLES; ++run) {
		cl_event first_kernel_event;
		cl_event last_kernel_event;

		const cl_uint sum_work_dim[1] = { SUM_DATA_SIZE };
		const cl_uint sum_work_group_dim[1] = { SUM_GROUP_SIZE };
		CHECK(clEnqueueNDRangeKernel(cmd_queue, sum_kernel, 1, nullptr, sum_work_dim, sum_work_group_dim, 0, nullptr, &first_kernel_event));

		const cl_uint reduce_work_items_dim[1] = { REDUCTION_WORK_ITEMS };
		const cl_uint reduce_work_group_dim[1] = { REDUCTION_WORK_GROUP_SIZE };
		CHECK(clEnqueueNDRangeKernel(cmd_queue, reduce_pass1_kernel, 1, nullptr, reduce_work_items_dim, reduce_work_group_dim, 0, nullptr, nullptr));

		CHECK(clEnqueueTask(cmd_queue, reduce_pass2_kernel, 0, nullptr, &last_kernel_event));

		CHECK(clFinish(cmd_queue));

		cl_ulong kernels_start_time;
		cl_ulong kernels_end_time;
		clGetEventProfilingInfo(first_kernel_event, CL_PROFILING_COMMAND_START, sizeof(kernels_start_time), &kernels_start_time, nullptr);
		clGetEventProfilingInfo(last_kernel_event,  CL_PROFILING_COMMAND_END,   sizeof(kernels_end_time),   &kernels_end_time,   nullptr);
		cl_ulong kernels_duration = kernels_end_time - kernels_start_time;

		clReleaseEvent(first_kernel_event);
		clReleaseEvent(last_kernel_event);

		timing_samples[run] = (double)kernels_duration / 1000000.0;
	}
	printTimingStats(timing_samples, NUM_TIMING_SAMPLES);


	std::cout << "Reading computation results back to host...\n";
	float gpu_result = 0.f;
	CHECK(clEnqueueReadBuffer(cmd_queue, final_sum_buf, CL_FALSE, 0, sizeof(gpu_result), &gpu_result, 0, nullptr, nullptr));
	CHECK(clFinish(cmd_queue));
	std::cout << "Done!\n\n";

	std::cout << "Comparing results...\n";
	std::cout << "CPU: " << cpu_result << " - GPU: " << gpu_result << '\n';
	if (compareFloat(cpu_result, gpu_result)) {
		std::cout << "Results match!\n\n";
	} else {
		std::cout << "Results don't match! :(\n\n";
	}

	std::cout << "Releasing OpenCL buffers...\n";
	CHECK(clReleaseMemObject(src_a_buf));
	CHECK(clReleaseMemObject(src_b_buf));
	CHECK(clReleaseMemObject(dst_buf));
	CHECK(clReleaseMemObject(sums_buf));
	CHECK(clReleaseMemObject(final_sum_buf));

	std::cout << "Freeing host buffers...\n";
	src_a.free();
	src_b.free();

	std::cout << "Releasing kernel...\n";
	CHECK(clReleaseKernel(sum_kernel));

	std::cout << "Releasing program...\n";
	CHECK(clReleaseProgram(program));

	return 0;
}
*/
