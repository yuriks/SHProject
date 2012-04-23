#include "timer.hpp"

#include <Windows.h>

#ifdef _WIN32
static LARGE_INTEGER clock_start;
#else
#error Unsupported system
#endif

void startPerfTimer()
{
#ifdef _WIN32
	QueryPerformanceCounter(&clock_start);
#endif
}

double stopPerfTimer()
{
#ifdef _WIN32
	LARGE_INTEGER clock_end;
	QueryPerformanceCounter(&clock_end);

	LARGE_INTEGER clock_freq;
	QueryPerformanceFrequency(&clock_freq);

	return (clock_end.QuadPart - clock_start.QuadPart) / (double)(clock_freq.QuadPart) * 1000.0;
#endif
}