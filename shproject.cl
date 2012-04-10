const sampler_t cube_sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;

enum CubeFace {
	FACE_POS_X, FACE_NEG_X,
	FACE_POS_Y, FACE_NEG_Y,
	FACE_POS_Z, FACE_NEG_Z
};

float areaIntegral(float2 pos)
{
	return atan2(pos.x * pos.y, sqrt(dot(pos, pos) + 1.f));
}

float calcSolidAngle(float2 st, float half_texel_size)
{
	float2 xy0 = st - half_texel_size;
	float2 xy1 = st + half_texel_size;

	return areaIntegral(xy0)
	     - areaIntegral((float2)(xy0.x, xy1.y))
	     - areaIntegral((float2)(xy1.x, xy0.y))
	     + areaIntegral(xy1);
}

float3 calcDirectionVector(enum CubeFace face, float2 st)
{
	float3 dir;

	switch (face) {
		case FACE_POS_Z: dir = (float3)( st.x, -st.y,  1.f);  break;
		case FACE_NEG_Z: dir = (float3)(-st.x, -st.y, -1.f);  break;
		case FACE_NEG_X: dir = (float3)(-1.f,  -st.y,  st.x); break;
		case FACE_POS_X: dir = (float3)( 1.f,  -st.y, -st.x); break;
		case FACE_POS_Y: dir = (float3)( st.x,  1.f,   st.y); break;
		case FACE_NEG_Y: dir = (float3)( st.x, -1.f,  -st.y); break;
	}

	//return normalize(dir); Fucking nVidia and it's buggy drivers
	return dir / sqrt(dot(dir, dir));
}

// Calculates the SH transform for a single texel.
void shTransformTexel(
	int2 pos,
	read_only image2d_t cube_face,
	float3 sh_coeffs[9],
	const enum CubeFace face_id,
	const float inv_size)
{
	float2 st = (convert_float2(pos) + (float2)0.5f) * (float2)inv_size * 2.f - (float2)1.f;

	float3 texel = read_imagef(cube_face, cube_sampler, pos).xyz;
	float solid_angle = calcSolidAngle(st, inv_size);
	float3 dir = calcDirectionVector(face_id, st);

	static const float c_SHconst_0 = 0.28209479177387814347f; // 1 / (2*sqrt(pi))
	static const float c_SHconst_1 = 0.48860251190291992159f; // sqrt(3 /(4pi))
	static const float c_SHconst_2 = 1.09254843059207907054f; // 1/2 * sqrt(15/pi)
	static const float c_SHconst_3 = 0.31539156525252000603f; // 1/4 * sqrt(5/pi)
	static const float c_SHconst_4 = 0.54627421529603953527f; // 1/4 * sqrt(15/pi)

	// l, m = 0, 0
	sh_coeffs[0] += texel * c_SHconst_0 * solid_angle;

	// l, m = 1, -1
	sh_coeffs[1] += texel * c_SHconst_1 * dir.y * solid_angle;
	// l, m = 1, 0
	sh_coeffs[2] += texel * c_SHconst_1 * dir.z * solid_angle;
	// l, m = 1, 1
	sh_coeffs[3] += texel * c_SHconst_1 * dir.x * solid_angle;

	// l, m = 2, -2
	sh_coeffs[4] += texel * c_SHconst_2 * (dir.x*dir.y) * solid_angle;
	// l, m = 2, -1
	sh_coeffs[5] += texel * c_SHconst_2 * (dir.y*dir.z) * solid_angle;
	// l, m = 2, 0
	sh_coeffs[6] += texel * c_SHconst_3 * (3.0f*dir.z*dir.z - 1.0f) * solid_angle;
	// l, m = 2, 1
	sh_coeffs[7] += texel * c_SHconst_2 * (dir.x*dir.z) * solid_angle;
	// l, m = 2, 2
	sh_coeffs[8] += texel * c_SHconst_4 * (dir.x*dir.x - dir.y*dir.y) * solid_angle;
}

// Calculates the SH projection for the texels in cube_face, partically summing
// (integrating) them to partial_sh_coeffs.
kernel void shTransform(
	read_only image2d_t cube_face,
	global float3 partial_sh_coeffs[/* global_size / local_size */][9],
	local float3 scratch_coeffs[/* local_size */][9],
	const enum CubeFace face_id,
	const float inv_size)
{
	float3 coeff_accum[9];
	for (int i = 0; i < 9; ++i)
		coeff_accum[i] = 0.f;

	int2 img_dims = get_image_dim(cube_face);
	for (int i = get_global_id(0); i < img_dims.x * img_dims.y; i += get_global_size(0)) {
		int2 pos = (int2)(i % img_dims.x, i / img_dims.y);
		shTransformTexel(pos, cube_face, coeff_accum, face_id, inv_size);
	}

	int lid = get_local_id(0);
	for (int i = 0; i < 9; ++i)
		scratch_coeffs[lid][i] = coeff_accum[i];
	barrier(CLK_LOCAL_MEM_FENCE);

	for (int offset = get_local_size(0) / 2; offset > 0; offset /= 2) {
		if (lid < offset) {
			for (int i = 0; i < 9; ++i)
				scratch_coeffs[lid][i] += scratch_coeffs[lid+offset][i];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	if (lid == 0) {
		for (int i = 0; i < 9; ++i)
			partial_sh_coeffs[get_group_id(0)][i] = scratch_coeffs[0][i];
	}
}

// Reduce-sums SH coefficients in partial_sh_coeffs, outputting final set of
// coefficients to final_coeffs.
kernel void shReduce(
	global const float3 partial_sh_coeffs[/* n */][9],
	global       float3 final_coeffs[9],
	const int n)
{
	size_t idx = get_global_id(0);
	float3 coeff_accum = 0.f;

	for (int i = 0; i < n; ++i) {
		coeff_accum += partial_sh_coeffs[i][idx];
	}

	final_coeffs[idx] = coeff_accum;
}
