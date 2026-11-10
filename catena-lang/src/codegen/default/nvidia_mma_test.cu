// Appended to production CUDA support by the ignored NVIDIA hardware test.
#include <vector>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#define CUDA_CHECK(call) do { cudaError_t e=(call); if(e!=cudaSuccess) { fprintf(stderr,"%s: %s\n",#call,cudaGetErrorString(e)); exit(1); } } while(0)

// Four warps per block, each computing an independent row-major tile.
__global__ void verify_mma(const catena_bf16_t* a, const catena_bf16_t* b,
                           const float* c, float* d, int tiles, int steps) {
    int lane=threadIdx.x%32, tile=blockIdx.x*4+threadIdx.x/32;
    if(tile>=tiles) return; // Uniform for every lane of the warp.
    int g=lane/4, t=lane%4;
    a+=tile*256; b+=tile*128; c+=tile*128; d+=tile*128;
    auto af=catena_mma_a_pack(
        catena_bf16_pack(a[g*16+2*t],a[g*16+2*t+1]),
        catena_bf16_pack(a[(g+8)*16+2*t],a[(g+8)*16+2*t+1]),
        catena_bf16_pack(a[g*16+2*t+8],a[g*16+2*t+9]),
        catena_bf16_pack(a[(g+8)*16+2*t+8],a[(g+8)*16+2*t+9]));
    auto bf=catena_mma_b_pack(
        catena_bf16_pack(b[2*t*8+g],b[(2*t+1)*8+g]),
        catena_bf16_pack(b[(2*t+8)*8+g],b[(2*t+9)*8+g]));
    catena_mma_f32_acc_t acc;
    for(int i=0;i<4;++i) acc.elt[i]=c[(g+8*(i/2))*8+2*t+i%2];
    for(int step=0;step<steps;++step) acc=catena_mma_bf16_f32(af,bf,acc);
    for(int i=0;i<4;++i) d[(g+8*(i/2))*8+2*t+i%2]=catena_mma_acc_get(acc,i);
}

int main() {
    // Every A coordinate and every B/output column are isolated by one-hot tiles.
    // Extra dense tiles exercise signed, exactly representable BF16 fractions.
    const int directed=16*16*8, tiles=directed+33;
    std::vector<catena_bf16_t> a(tiles*256), b(tiles*128);
    for(int tile=0;tile<tiles;++tile) {
        for(int i=0;i<256;++i) a[tile*256+i]=catena_bf16_from_f32(tile<directed ? 0.0f : float((i*7+tile*3)%17-8)/8);
        for(int i=0;i<128;++i) b[tile*128+i]=catena_bf16_from_f32(tile<directed ? 0.0f : float((i*3+tile*5)%19-9)/8);
        if(tile<directed) {
            int row=tile/128, k=(tile/8)%16, col=tile%8;
            a[tile*256+row*16+k]=catena_bf16_from_f32(1.0f);
            b[tile*128+k*8+col]=catena_bf16_from_f32(1.0f);
        }
    }
    catena_bf16_t *da,*db; float *dc,*dd;
    CUDA_CHECK(cudaMalloc(&da,a.size()*sizeof(a[0])));
    CUDA_CHECK(cudaMalloc(&db,b.size()*sizeof(b[0])));
    CUDA_CHECK(cudaMalloc(&dc,tiles*128*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dd,tiles*128*sizeof(float)));
    CUDA_CHECK(cudaMemcpy(da,a.data(),a.size()*sizeof(a[0]),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(db,b.data(),b.size()*sizeof(b[0]),cudaMemcpyHostToDevice));
    std::vector<float> c(tiles*128), d(tiles*128);
    for(int nonzero=0;nonzero<2;++nonzero) for(int steps : {1,3}) {
        for(size_t i=0;i<c.size();++i) c[i]=nonzero ? float(int(i%29)-14)/4 : 0.0f;
        CUDA_CHECK(cudaMemcpy(dc,c.data(),c.size()*sizeof(float),cudaMemcpyHostToDevice));
        verify_mma<<<(tiles+3)/4,128>>>(da,db,dc,dd,tiles,steps);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(d.data(),dd,d.size()*sizeof(float),cudaMemcpyDeviceToHost));
        for(int tile=0;tile<tiles;++tile) for(int row=0;row<16;++row) for(int col=0;col<8;++col) {
            double dot=0;
            for(int k=0;k<16;++k) dot+=double(catena_bf16_to_f32(a[tile*256+row*16+k]))*catena_bf16_to_f32(b[tile*128+k*8+col]);
            int i=tile*128+row*8+col;
            double expected=c[i]+steps*dot;
            if(!std::isfinite(d[i]) || std::abs(d[i]-expected)>1e-5+1e-5*std::abs(expected)) {
                fprintf(stderr,"tile=%d row=%d col=%d initial=%d steps=%d got=%g expected=%g\n",tile,row,col,nonzero,steps,d[i],expected);
                return 1;
            }
        }
    }
    CUDA_CHECK(cudaFree(da)); CUDA_CHECK(cudaFree(db)); CUDA_CHECK(cudaFree(dc)); CUDA_CHECK(cudaFree(dd));
    puts("NVIDIA MMA one-hot, dense, accumulator, repeated-step and warp-isolation checks passed");
}
