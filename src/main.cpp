#include <cassert>
#include <cmath>
#include <cstdint>
#include <functional>
#include <iostream>
#include <iterator>
#include <memory>
#include <string>
#include <vector>

#include "stb_image.hpp"
#include "stb_image_write.hpp"

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
	{}

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

	bool no_opencl;
	bool verbose;
};

ProgramOptions parseOptions(int argc, char* argv[]) {
	ProgramOptions opts;

	// Set defaults;
	opts.return_code = 0;
	opts.no_opencl = false;
	opts.verbose = false;

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
				opts.no_opencl = true;
			} else if (opt == "-v" || opt == "-verbose") {
				opts.verbose = true;
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

int main(int argc, char* argv[]) {
	ProgramOptions opts = parseOptions(argc, argv);

	if (opts.return_code != 0)
		return opts.return_code;

	if (opts.positional_params.size() != 2) {
		printProgramUsage();
		return 1;
	}

	std::string fname_prefix = opts.positional_params[0];
	std::string fname_extension = opts.positional_params[1];

	Cubemap input_cubemap(fname_prefix, fname_extension);

	Colorf sh_coeffs[9];
	shproject_cpu(sh_coeffs, input_cubemap);

	std::cout << "// l = 0\n";
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
