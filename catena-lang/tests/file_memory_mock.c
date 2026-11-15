#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>

static int allocated, freed, registered, unregistered, copies;
static int integrated(void) { return atoi(getenv("FILE_INTEGRATED")); }
int hipGetDevice(int *d) { *d = 7; return 0; }
int hipDeviceGetAttribute(int *v, int a, int d) { assert(a == 16 && d == 7); *v = integrated(); return 0; }
int cudaDeviceGetAttribute(int *v, int a, int d) { assert(a == 18 && d == 7); *v = integrated(); return 0; }
int hipMalloc(void **p, size_t n) { allocated++; *p = malloc(n); return !*p; }
int hipFree(void *p) { freed++; free(p); return 0; }
int hipMemcpy(void *d, const void *s, size_t n, int kind) { assert(kind == 1 && n <= 1024 * 1024); copies++; memcpy(d, s, n); return 0; }
int hipHostRegister(void *p, size_t n, unsigned f) { assert(p && n && f == 10); registered++; return 0; }
int hipHostGetDevicePointer(void **d, void *p, unsigned f) { assert(f == 0); *d = p; return 0; }
// The mapping must still exist when the driver unregisters it.
int hipHostUnregister(void *p) { unsigned char resident; assert(mincore(p, 1, &resident) == 0); unregistered++; return 0; }
void file_test_check(void) {
    assert(allocated == !integrated() && freed == allocated);
    assert(registered == integrated() && unregistered == registered);
    assert(copies == 2 * !integrated());
}
#define CUDA_ALIAS(name) __typeof__(hip##name) cuda##name __attribute__((alias("hip" #name)));
CUDA_ALIAS(GetDevice) CUDA_ALIAS(Malloc) CUDA_ALIAS(Free) CUDA_ALIAS(Memcpy)
CUDA_ALIAS(HostRegister) CUDA_ALIAS(HostGetDevicePointer) CUDA_ALIAS(HostUnregister)
