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
	if (verboseComputeLogging)
		std::cout << "Found " << num_platforms << " platforms:\n";

	for (unsigned int i = 0; i < num_platforms; ++i) {
		char vendor_buffer[64];
		char name_buffer[64];
		char version_buffer[64];

		CHECK(clGetPlatformInfo(platform_ids[i], CL_PLATFORM_VENDOR,  sizeof(vendor_buffer),  static_cast<void*>(vendor_buffer),  nullptr));
		CHECK(clGetPlatformInfo(platform_ids[i], CL_PLATFORM_NAME,    sizeof(name_buffer),    static_cast<void*>(name_buffer),    nullptr));
		CHECK(clGetPlatformInfo(platform_ids[i], CL_PLATFORM_VERSION, sizeof(version_buffer), static_cast<void*>(version_buffer), nullptr));
		if (verboseComputeLogging)
			std::cout << "    " << vendor_buffer << " - " << name_buffer << " - " << version_buffer << '\n';

		cl_device_id device_ids[8];
		cl_uint num_devices;

		CHECK(clGetDeviceIDs(platform_ids[i], CL_DEVICE_TYPE_ALL, 8, device_ids, &num_devices));

		for (unsigned int j = 0; j < num_devices; ++j) {
			char vendor_buffer[64];
			char name_buffer[64];

			CHECK(clGetDeviceInfo(device_ids[j], CL_DEVICE_VENDOR, sizeof(vendor_buffer), static_cast<void*>(vendor_buffer), nullptr));
			CHECK(clGetDeviceInfo(device_ids[j], CL_DEVICE_NAME,   sizeof(name_buffer),   static_cast<void*>(name_buffer),   nullptr));
			if (verboseComputeLogging)
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
		std::cout << "Using OpenCL device " << name_buffer << " - " << version_buffer << '\n';
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

	if (verboseComputeLogging)
		std::cout << "Creating context...\n";

	cl_int error_code;
	cl_context context = clCreateContext(context_properties, 1, &device_id, nullptr, nullptr, &error_code);
	CHECK(error_code);

	return context;
}

cl_command_queue createCommandQueue(cl_context context, cl_device_id device_id, bool profiling)
{
	if (verboseComputeLogging)
		std::cout << "Creating command queue...\n";

	cl_int error_code;

	cl_command_queue_properties queue_props = CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE;
	if (profiling)
		queue_props |= CL_QUEUE_PROFILING_ENABLE;

	cl_command_queue cmd_queue = clCreateCommandQueue(context, device_id, queue_props, &error_code);
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
		if (verboseComputeLogging)
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
	if (verboseComputeLogging)
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

bool verboseComputeLogging = false;

bool initCompute(ComputeContext& ctx, bool profiling)
{
	if (!enumerateDevices(ctx.platform_id, ctx.device_id))
		return false;

	ctx.context = createContext(ctx.platform_id, ctx.device_id);
	ctx.cmd_queue = createCommandQueue(ctx.context, ctx.device_id, profiling);

	return true;
}

void deinitCompute(ComputeContext& ctx)
{
	if (verboseComputeLogging)
		std::cout << "Releasing command queue...\n";
	CHECK(clReleaseCommandQueue(ctx.cmd_queue));

	if (verboseComputeLogging)
		std::cout << "Releasing context...\n";
	CHECK(clReleaseContext(ctx.context));
}

cl_program loadProgram(const ComputeContext& ctx, const char* fname)
{
	if (verboseComputeLogging)
		std::cout << "Loading program from " << fname << '\n';
	cl_program program = loadProgramFromFile(ctx.context, fname);

	if (verboseComputeLogging)
		std::cout << "Building program...\n";
	CHECK(clBuildProgram(program, 0, nullptr, "-Werror -cl-mad-enable -cl-fast-relaxed-math", nullptr, nullptr));
	if (!checkProgramLog(program, ctx.device_id)) {
		return nullptr;
	}

	return program;
}
