#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static size_t stats[6]; // malloc, free, register, unregister, copies, largest copy
size_t file_test_stat(int i) { return stats[i]; }
static int failure(const char *name) {
    const char *value = getenv("FILE_TEST_FAILURE");
    return value && strcmp(value, name) == 0;
}
int hipGetDevice(int *device) { *device = 7; return 0; }
int cudaGetDevice(int *device) { return hipGetDevice(device); }
static int attribute(int *value, int attr, int device, int expected) {
    assert(attr == expected && device == 7);
    *value = atoi(getenv("FILE_TEST_INTEGRATED"));
    if (failure("truncate")) assert(truncate(getenv("FILE_TEST_PATH"), 1) == 0);
    return failure("attribute");
}
int hipDeviceGetAttribute(int *v, int a, int d) { return attribute(v, a, d, 16); }
int cudaDeviceGetAttribute(int *v, int a, int d) { return attribute(v, a, d, 18); }
int hipMalloc(void **p, size_t n) { stats[0]++; *p = malloc(n); return !*p; }
int cudaMalloc(void **p, size_t n) { return hipMalloc(p, n); }
int hipFree(void *p) { stats[1]++; free(p); return 0; }
int cudaFree(void *p) { return hipFree(p); }
int hipHostRegister(void *p, size_t n, unsigned flags) {
    assert(p && n && flags == 10);
    stats[2]++;
    return failure("register");
}
int cudaHostRegister(void *p, size_t n, unsigned flags) { return hipHostRegister(p, n, flags); }
int hipHostGetDevicePointer(void **d, void *p, unsigned flags) {
    assert(flags == 0); *d = failure("null") ? NULL : p;
    return failure("pointer");
}
int cudaHostGetDevicePointer(void **d, void *p, unsigned f) { return hipHostGetDevicePointer(d, p, f); }
int hipHostUnregister(void *p) { assert(p); stats[3]++; return 0; }
int cudaHostUnregister(void *p) { return hipHostUnregister(p); }
int hipMemcpy(void *d, const void *s, size_t n, int kind) {
    assert(kind == 1); stats[4]++;
    if (n > stats[5]) stats[5] = n;
    if (failure("copy")) return 1;
    memcpy(d, s, n); return 0;
}
int cudaMemcpy(void *d, const void *s, size_t n, int kind) { return hipMemcpy(d, s, n, kind); }
