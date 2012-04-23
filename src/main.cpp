#include <cassert>
#include <cmath>
#include <cstdint>
#include <functional>
#include <iostream>
#include <iterator>
#include <memory>
#include <string>
#include <vector>
#include <numeric>

#include "stb_image.hpp"
#include "stb_image_write.hpp"
#include "compute.hpp"
#include "timer.hpp"

typedef uint8_t u8;
typedef uint32_t u32;

inline void splitColor(u32 col, u8& r, u8& g, u8& b) {
	r = col >> 0  & 0xFF;
	g = col >> 8  & 0xFF;
	b = col >> 16 & 0xFF;
}

inline u32 makeColor(u8 r, u8 g, u8 b) {
	return r | g << 8 | b << 16 | 0xFF << 24;
}

inline float lerp(float a, float b, float t) {
	return a * (1.f - t) + b * t;
}

inline float areaIntegral(float x, float y) {
	return std::atan2f(x*y, std::sqrtf(x*x + y*y + 1));
}

inline float unlerp(int val, int max) {
	return (val + 0.5f) / max;
}

struct Image {
	int width, height;
	std::unique_ptr<u8, std::function<void(u8*)>> data;

	Image() :
		width(-1), height(-1)
	{}

	Image(const std::string& filename) :
		data(stbi_load(filename.c_str(), &width, &height, nullptr, 4), stbi_image_free)
	{
		if (data == nullptr) {
			std::cerr << "Failed to load " << filename << ".\n";
		}
	}

	Image& operator= (Image&& o) {
		width = o.width;
		height = o.height;
		data.swap(o.data);

		return *this;
	}

private:
	Image& operator= (const Image&);
};

struct Colorf {
	float r, g, b;

	Colorf() { }
	Colorf(float r, float g, float b) : r(r), g(g), b(b) { }

	explicit Colorf(u32 col) {
		u8 rb, gb, bb;
		splitColor(col, rb, gb, bb);
		r = rb / 255.f;
		g = gb / 255.f;
		b = bb / 255.f;
	}

	u32 toU32() const {
		return makeColor(
			static_cast<u8>(r * 255),
			static_cast<u8>(g * 255),
			static_cast<u8>(b * 255));
	}

	Colorf& operator += (const Colorf& o) {
		r += o.r;
		g += o.g;
		b += o.b;
		return *this;
	}

	static Colorf mix(const Colorf& a, const Colorf& b, float t) {
		return Colorf(
			lerp(a.r, b.r, t),
			lerp(a.g, b.g, t),
			lerp(a.b, b.b, t));
	}
};

Colorf operator * (const Colorf& a, const Colorf& b) {
	return Colorf(a.r * b.r, a.g * b.g, a.b * b.b);
}

Colorf operator * (const Colorf& a, const float f) {
	return Colorf(a.r * f, a.g * f, a.b * f);
}

struct Cubemap {
	enum CubeFace {
		FACE_POS_X, FACE_NEG_X,
		FACE_POS_Y, FACE_NEG_Y,
		FACE_POS_Z, FACE_NEG_Z,
		NUM_FACES
	};

	Image faces[NUM_FACES];

	Cubemap(const std::string& fname_prefix, const std::string& fname_extension) {
		faces[FACE_POS_X] = Image(fname_prefix + "_right."  + fname_extension);
		faces[FACE_NEG_X] = Image(fname_prefix + "_left."   + fname_extension);
		faces[FACE_POS_Y] = Image(fname_prefix + "_top."    + fname_extension);
		faces[FACE_NEG_Y] = Image(fname_prefix + "_bottom." + fname_extension);
		faces[FACE_POS_Z] = Image(fname_prefix + "_front."  + fname_extension);
		faces[FACE_NEG_Z] = Image(fname_prefix + "_back."   + fname_extension);
	}

	u32 readTexel(CubeFace face, int x, int y) const {
		assert(face < NUM_FACES);
		const Image& face_img = faces[face];

		assert(x < face_img.width);
		assert(y < face_img.height);

		const u32* img_data = reinterpret_cast<const u32*>(face_img.data.get());
		return img_data[y * face_img.width + x];
	}

	void computeTexCoords(float x, float y, float z, CubeFace& out_face, float& out_s, float& out_t) {
		int major_axis;

		float v[3] = { x, y, z };
		float a[3] = { std::abs(x), std::abs(y), std::abs(z) };

		if (a[0] >= a[1] && a[0] >= a[2]) {
			major_axis = 0;
		} else if (a[1] >= a[0] && a[1] >= a[2]) {
			major_axis = 1;
		} else if (a[2] >= a[0] && a[2] >= a[1]) {
			major_axis = 2;
		}

		if (v[major_axis] < 0.0f)
			major_axis = major_axis * 2 + 1;
		else
			major_axis *= 2;

		float tmp_s, tmp_t, m;
		switch (major_axis) {
			/* +X */ case 0: tmp_s = -z; tmp_t = -y; m = a[0]; break;
			/* -X */ case 1: tmp_s =  z; tmp_t = -y; m = a[0]; break;
			/* +Y */ case 2: tmp_s =  x; tmp_t =  z; m = a[1]; break;
			/* -Y */ case 3: tmp_s =  x; tmp_t = -z; m = a[1]; break;
			/* +Z */ case 4: tmp_s =  x; tmp_t = -y; m = a[2]; break;
			/* -Z */ case 5: tmp_s = -x; tmp_t = -y; m = a[2]; break;
		}

		out_face = CubeFace(major_axis);
		out_s = 0.5f * (tmp_s / m + 1.0f);
		out_t = 0.5f * (tmp_t / m + 1.0f);
	}

	u32 readTexelClamped(CubeFace face, int x, int y) {
		const Image& face_img = faces[face];

		x = std::max(std::min(x, face_img.width  - 1), 0);
		y = std::max(std::min(y, face_img.height - 1), 0);
		return readTexel(face, x, y);
	}

	Colorf sampleFace(CubeFace face, float s, float t) {
		const Image& face_img = faces[face];

		const float x = s * face_img.width;
		const float y = t * face_img.height;

		const int x_base = static_cast<int>(x);
		const int y_base = static_cast<int>(y);
		const float x_fract = x - x_base;
		const float y_fract = y - y_base;

		const Colorf sample_00(readTexelClamped(face, x_base,     y_base));
		const Colorf sample_10(readTexelClamped(face, x_base + 1, y_base));
		const Colorf sample_01(readTexelClamped(face, x_base,     y_base + 1));
		const Colorf sample_11(readTexelClamped(face, x_base + 1, y_base + 1));

		const Colorf mix_0 = Colorf::mix(sample_00, sample_10, x_fract);
		const Colorf mix_1 = Colorf::mix(sample_01, sample_11, x_fract);

		return Colorf::mix(mix_0, mix_1, y_fract);
	}

	float calcSolidAngle(CubeFace face, int x, int y) const {
		float s = unlerp(x, faces[face].width)  * 2.0f - 1.0f;
		float t = unlerp(y, faces[face].height) * 2.0f - 1.0f;

		// assumes square face
		float half_texel_size = 1.0f / faces[face].width;

		float x0 = s - half_texel_size;
		float y0 = t - half_texel_size;
		float x1 = s + half_texel_size;
		float y1 = t + half_texel_size;

		return areaIntegral(x0, y0) - areaIntegral(x0, y1) - areaIntegral(x1, y0) + areaIntegral(x1, y1);
	}

	void calcDirectionVector(CubeFace face, int face_x, int face_y, float& out_x, float& out_y, float& out_z) const {
		float s = unlerp(face_x, faces[face].width)  * 2.0f - 1.0f;
		float t = unlerp(face_y, faces[face].height) * 2.0f - 1.0f;

		float x, y, z;

		switch (face) {
		case FACE_POS_Z: x =  s; y = -t; z =  1; break;
		case FACE_NEG_Z: x = -s; y = -t; z = -1; break;
		case FACE_NEG_X: x = -1; y = -t; z =  s; break;
		case FACE_POS_X: x =  1; y = -t; z = -s; break;
		case FACE_POS_Y: x =  s; y =  1; z =  t; break;
		case FACE_NEG_Y: x =  s; y = -1; z = -t; break;
		}

		// Normalize vector
		float inv_len = 1.0f / std::sqrtf(x*x + y*y + z*z);
		out_x = x * inv_len;
		out_y = y * inv_len;
		out_z = z * inv_len;
	}
};

template <typename T>
inline typename T::value_type pop_from(T& container) {
	T::value_type val = container.back();
	container.pop_back();
	return val;
}

void printProgramUsage() {
	std::cerr <<
		"SHProject v.GIT\n"
		"\n"
		"Usage:\n"
		"  SHProject [opts] [-] input_prefix input_extension\n"
		"\n"
		"Available options:\n"
		"  -c / -no-opencl  Disable OpenCL support.\n"
		"  -v / -verbose    Enables all sorts of debug garbage.\n"
		"  -p / -profile    Enable timing statistics.\n"
		"  -h / -help       Print this help text.\n"
		"\n";
}

std::ostream& printColorf(std::ostream& s, const Colorf& col) {
	return s << col.r << "f, " << col.g << "f, " << col.b << "f,\n";
}

struct ProgramOptions
{
	std::vector<std::string> positional_params;
	int return_code;

	bool use_opencl;
	bool verbose;
	bool profile;
};

ProgramOptions parseOptions(int argc, char* argv[]) {
	ProgramOptions opts;

	// Set defaults;
	opts.return_code = 0;
	opts.use_opencl = true;
	opts.verbose = false;
	opts.profile = false;

	if (argc < 1) {
		printProgramUsage();
		opts.return_code = 1;
		return opts;
	}

	// Parse options
	std::vector<std::string> input_params(std::reverse_iterator<char**>(argv+argc), std::reverse_iterator<char**>(argv+1));

	while (!input_params.empty()) {
		std::string opt = pop_from(input_params);

		if (!opt.empty() && opt[0] == '-') {
			if (opt == "-") {
				std::copy(input_params.rbegin(), input_params.rend(), std::back_inserter(opts.positional_params));
				break;
			} else if (opt == "-c" || opt == "-no-opencl") {
				opts.use_opencl = false;
			} else if (opt == "-v" || opt == "-verbose") {
				opts.verbose = true;
			} else if (opt == "-p" || opt == "-profile") {
				opts.profile = true;
			} else if (opt == "-h" || opt == "-help") {
				printProgramUsage();
				opts.return_code = 1;
				break;
			} else {
				std::cerr << "Unknown option " << opt << ". Try -help.\n";
				opts.return_code = 1;
				break;
			}
		} else {
			opts.positional_params.push_back(opt);
		}
	}

	return opts;
}

static ProgramOptions opts;

void shproject_cpu(Colorf sh_coeffs[9], const Cubemap& input_cubemap)
{
	for (int i = 0; i < 9; ++i)
		sh_coeffs[i] = Colorf(0.f, 0.f, 0.f);

	int in_width = input_cubemap.faces[0].width;
	int in_height = input_cubemap.faces[0].height;

	for (int face_i = 0; face_i < Cubemap::NUM_FACES; ++face_i) {
		for (int y = 0; y < in_height; ++y) {
			for (int x = 0; x < in_width; ++x) {
				static const float c_SHconst_0 = 0.28209479177387814347f; // 1 / (2*sqrt(pi))
				static const float c_SHconst_1 = 0.48860251190291992159f; // sqrt(3 /(4pi))
				static const float c_SHconst_2 = 1.09254843059207907054f; // 1/2 * sqrt(15/pi)
				static const float c_SHconst_3 = 0.31539156525252000603f; // 1/4 * sqrt(5/pi)
				static const float c_SHconst_4 = 0.54627421529603953527f; // 1/4 * sqrt(15/pi)

				const Cubemap::CubeFace face = static_cast<Cubemap::CubeFace>(face_i);
				const Colorf texel(input_cubemap.readTexel(face, x, y));
				const float solid_angle = input_cubemap.calcSolidAngle(face, x, y);

				float dir_x;
				float dir_y;
				float dir_z;
				input_cubemap.calcDirectionVector(face, x, y, dir_x, dir_y, dir_z);

				// l, m = 0, 0
				sh_coeffs[0] += texel * c_SHconst_0 * solid_angle;

				// l, m = 1, -1
				sh_coeffs[1] += texel * c_SHconst_1 * dir_y * solid_angle;
				// l, m = 1, 0
				sh_coeffs[2] += texel * c_SHconst_1 * dir_z * solid_angle;
				// l, m = 1, 1
				sh_coeffs[3] += texel * c_SHconst_1 * dir_x * solid_angle;

				// l, m = 2, -2
				sh_coeffs[4] += texel * c_SHconst_2 * (dir_x*dir_y) * solid_angle;
				// l, m = 2, -1
				sh_coeffs[5] += texel * c_SHconst_2 * (dir_y*dir_z) * solid_angle;
				// l, m = 2, 0
				sh_coeffs[6] += texel * c_SHconst_3 * (3.0f*dir_z*dir_z - 1.0f) * solid_angle;
				// l, m = 2, 1
				sh_coeffs[7] += texel * c_SHconst_2 * (dir_x*dir_z) * solid_angle;
				// l, m = 2, 2
				sh_coeffs[8] += texel * c_SHconst_4 * (dir_x*dir_x - dir_y*dir_y) * solid_angle;
			}
		}
	}
}

void shproject_compute(Colorf sh_coeffs[9], const Cubemap& input_cubemap, const ComputeContext& ctx)
{
	cl_int error_code;

	cl_program program = loadProgram(ctx, "shproject.cl");
	cl_kernel pass1_kernel = createKernel(program, "shTransform");
	cl_kernel pass2_kernel = createKernel(program, "shReduce");

	// Number of items processed by each pass1
	size_t data_size = input_cubemap.faces[0].width * input_cubemap.faces[0].height;
	// Number of items processed in parallel by each pass1
	size_t work_items = data_size / 64;
	// Local work group size
	size_t work_group_size = 64;
	size_t partial_results_size = work_items / work_group_size;
	size_t partial_buffer_size = partial_results_size * 9 * 4 * sizeof(float);
	// Scratch space for pass1 intermediary results. Must fit in local memory
	size_t partial_scratch_size = work_group_size * 9 * 4 * sizeof(float);

	cl_event ev_face_upload[6];
	cl_event ev_pass1[6];
	cl_event ev_pass2;
	cl_event ev_download;

	// Image data
	cl_mem cube_faces[6];
	// Partial results calculated by each pass1 to be summed by pass2
	cl_mem partial_sh_buf = clCreateBuffer(ctx.context, CL_MEM_READ_WRITE, 6 * partial_buffer_size, nullptr, &error_code); CHECK(error_code);
	// Subviews from partial_sh_buf for eahc cube face
	cl_mem partial_face_buffers[6];
	// Holds final results
	cl_mem final_buffer = clCreateBuffer(ctx.context, CL_MEM_WRITE_ONLY, 9 * 4 * sizeof(float), nullptr, &error_code); CHECK(error_code);

	static const cl_image_format img_format = {
		/*.image_channel_order = */ CL_RGBA,
		/*.image_channel_data_type = */ CL_UNORM_INT8
	};

	// Create image buffers
	for (int i = 0; i < 6; ++i) {
		const Image& face = input_cubemap.faces[i];
		cube_faces[i] = clCreateImage2D(ctx.context, CL_MEM_READ_ONLY, &img_format, face.width, face.height, 0, nullptr, &error_code); CHECK(error_code);

		const cl_buffer_region region = {
			/* .origin = */ i * partial_buffer_size,
			/* .size = */ partial_buffer_size
		};
		partial_face_buffers[i] = clCreateSubBuffer(partial_sh_buf, CL_MEM_WRITE_ONLY, CL_BUFFER_CREATE_TYPE_REGION, &region, &error_code); CHECK(error_code);
	}

	if (opts.verbose)
		std::cerr << "Buffers created\n";

	CHECK(clSetKernelArg(pass1_kernel, 2, partial_scratch_size, nullptr));
	float inv_size = 1.f / input_cubemap.faces[0].width;
	CHECK(clSetKernelArg(pass1_kernel, 4, sizeof(inv_size), &inv_size));

	static const size_t img_copy_offsets[3] = { 0, 0, 0 };
	for (int i = 0 ; i < 6; ++i) {
		const Image& face = input_cubemap.faces[i];
		const size_t img_copy_region[3] = { face.width, face.height, 1 };

		CHECK(clEnqueueWriteImage(ctx.cmd_queue, cube_faces[i], CL_FALSE, img_copy_offsets, img_copy_region, 0, 0, face.data.get(), 0, nullptr, &ev_face_upload[i]));
		if (opts.verbose)
			std::cerr << "Enqueued upload " << i+1 << '\n';

		CHECK(clSetKernelArg(pass1_kernel, 0, sizeof(cube_faces[i]), &cube_faces[i]));
		CHECK(clSetKernelArg(pass1_kernel, 1, sizeof(partial_face_buffers[i]), &partial_face_buffers[i]));
		CHECK(clSetKernelArg(pass1_kernel, 3, sizeof(i), &i));

		CHECK(clEnqueueNDRangeKernel(ctx.cmd_queue, pass1_kernel, 1, nullptr, &work_items, &work_group_size, 1, &ev_face_upload[i], &ev_pass1[i]));
		if (opts.verbose)
			std::cerr << "Enqueued pass1 " << i+1 << '\n';

		clReleaseMemObject(cube_faces[i]);
		clReleaseMemObject(partial_face_buffers[i]);
	}

	CHECK(clSetKernelArg(pass2_kernel, 0, sizeof(partial_sh_buf), &partial_sh_buf));
	CHECK(clSetKernelArg(pass2_kernel, 1, sizeof(final_buffer), &final_buffer));
	int reduction_n = 6 * partial_results_size;
	CHECK(clSetKernelArg(pass2_kernel, 2, sizeof(reduction_n), &reduction_n));

	size_t reduction_size = 9;
	CHECK(clEnqueueNDRangeKernel(ctx.cmd_queue, pass2_kernel, 1, nullptr, &reduction_size, nullptr, 6, ev_pass1, &ev_pass2));

	if (opts.verbose)
		std::cerr << "Enqueued pass2\n";

	float results[9 * 4];
	CHECK(clEnqueueReadBuffer(ctx.cmd_queue, final_buffer, CL_FALSE, 0, sizeof(results), results, 1, &ev_pass2, &ev_download));

	if (opts.verbose)
		std::cerr << "Enqueued coefficients read\n";

	CHECK(clFinish(ctx.cmd_queue));

	// Free memory objects
	clReleaseMemObject(partial_sh_buf);
	clReleaseMemObject(final_buffer);

	if (opts.verbose)
		std::cerr << "Finished\n";

	for (int i = 0; i < 9; ++i) {
		sh_coeffs[i] = Colorf(
			results[i*4 + 0],
			results[i*4 + 1],
			results[i*4 + 2]);
	}
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

int main(int argc, char* argv[]) {
	opts = parseOptions(argc, argv);

	if (opts.return_code != 0)
		return opts.return_code;

	if (opts.verbose)
		verboseComputeLogging = true;

	if (opts.positional_params.size() != 2) {
		printProgramUsage();
		return 1;
	}

	std::string fname_prefix = opts.positional_params[0];
	std::string fname_extension = opts.positional_params[1];

	std::cout << "Loading images... " << std::flush;

	Cubemap input_cubemap(fname_prefix, fname_extension);
	for (int i = 0; i < Cubemap::NUM_FACES; ++i) {
		if (input_cubemap.faces[i].data == nullptr) {
			std::cerr << "Failed to open input files.\n";
			return 1;
		}
	}

	std::cout << "Done.\n";

	Colorf sh_coeffs[9];

	ComputeContext compute_ctx;
	if (opts.use_opencl) {
		opts.use_opencl = initCompute(compute_ctx, opts.profile);
		if (!opts.use_opencl) {
			std::cerr << "OpenCL not available.\n";
		}
	}

	if (!opts.use_opencl) {
		std::cout << "Using CPU algorithm.\n";
	}

	std::cout << "Processing..." << std::flush;

	static const int number_of_runs = 16;
	double run_times[number_of_runs];
	for (int run = 0; run < (opts.profile ? number_of_runs : 1); ++run) {
		if (opts.profile)
			startPerfTimer();

		if (opts.use_opencl) {
			shproject_compute(sh_coeffs, input_cubemap, compute_ctx);
		} else {
			shproject_cpu(sh_coeffs, input_cubemap);
		}

		if (opts.profile)
			run_times[run] = stopPerfTimer();
	}

	std::cout << "Done!\n";

	if (opts.profile) {
		printTimingStats(run_times, number_of_runs);
	}

	if (opts.use_opencl) {
		deinitCompute(compute_ctx);
	}

	std::cout << "\n\n// l = 0\n";
	printColorf(std::cout, sh_coeffs[0]);
	std::cout << "\n// l = 1\n";
	printColorf(std::cout, sh_coeffs[1]);
	printColorf(std::cout, sh_coeffs[2]);
	printColorf(std::cout, sh_coeffs[3]);
	std::cout << "\n// l = 2\n";
	printColorf(std::cout, sh_coeffs[4]);
	printColorf(std::cout, sh_coeffs[5]);
	printColorf(std::cout, sh_coeffs[6]);
	printColorf(std::cout, sh_coeffs[7]);
	printColorf(std::cout, sh_coeffs[8]);
}
