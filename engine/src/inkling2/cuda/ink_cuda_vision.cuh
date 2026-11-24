// Inkling-2-Mini ViT: HIP/CPU-reference geometry, BF16 matrices and FP32 accumulation.
#pragma once
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <limits>
#include <new>
#include "ink_manifest.h"
#include "ink_vision.h"

namespace ink2 {
namespace cuda_vision_detail {

__device__ inline float warp_sum(float x) {
    for (int d = 16; d; d >>= 1) x += __shfl_down_sync(0xffffffffu, x, d);
    return x;
}

__global__ void layernorm(const float* x, const float* w, const float* b, float* out, int D, float eps) {
    const int row = blockIdx.x, tid = threadIdx.x;
    x += (size_t)row * D;
    out += (size_t)row * D;
    __shared__ float sums[8], mean, inv;
    float sum = 0;
    for (int d = tid; d < D; d += 256) sum += x[d];
    sum = warp_sum(sum);
    if ((tid & 31) == 0) sums[tid >> 5] = sum;
    __syncthreads();
    if (tid == 0) {
        float s = 0;
        for (int i = 0; i < 8; ++i) s += sums[i];
        mean = s / D;
    }
    __syncthreads();
    float var = 0;
    for (int d = tid; d < D; d += 256) { float a = x[d] - mean; var = fmaf(a, a, var); }
    var = warp_sum(var);
    if ((tid & 31) == 0) sums[tid >> 5] = var;
    __syncthreads();
    if (tid == 0) {
        float s = 0;
        for (int i = 0; i < 8; ++i) s += sums[i];
        inv = rsqrtf(s / D + eps);
    }
    __syncthreads();
    for (int d = tid; d < D; d += 256) out[d] = (x[d] - mean) * inv * w[d] + b[d];
}

__global__ void convert_bf16(const float* x, __nv_bfloat16* y, size_t n) {
    const size_t i = (size_t)blockIdx.x * 256 + threadIdx.x;
    if (i < n) y[i] = __float2bfloat16_rn(x[i]);
}

__global__ void bias(float* x, const float* b, size_t n, int D) {
    const size_t i = (size_t)blockIdx.x * 256 + threadIdx.x;
    if (i < n) x[i] += b[i % D];
}

__global__ void gelu(float* x, size_t n) {
    const size_t i = (size_t)blockIdx.x * 256 + threadIdx.x;
    if (i < n) { float v = x[i]; x[i] = 0.5f * v * (1.0f + erff(v * 0.7071067811865475244f)); }
}

__global__ void residual(float* x, const float* delta, const float* gamma, size_t n, int D) {
    const size_t i = (size_t)blockIdx.x * 256 + threadIdx.x;
    if (i < n) x[i] += delta[i] * gamma[i % D];
}

// Input is exactly prepare_picture's [merged cell, 40, 40, 3]. No host patch tensor or reordered image.
__global__ void patchify(const float* pixels, float* patches, size_t n) {
    const size_t i = (size_t)blockIdx.x * 256 + threadIdx.x;
    if (i >= n) return;
    const size_t p = i / 1200, cell = p / 4;
    const int sub = (int)(p % 4), v = (int)(i % 1200);
    const int py = v / 60, px = (v / 3) % 20, ch = v % 3;
    patches[i] = pixels[(cell * 1600 + (sub / 2 * 20 + py) * 40 + sub % 2 * 20 + px) * 3 + ch];
}

__global__ void position(float* cur, const float* cls, const float* cls_pos, const float* pos,
                         int S, int D, int gh, int gw, int G) {
    const size_t i = (size_t)blockIdx.x * 256 + threadIdx.x;
    if (i >= (size_t)S * D) return;
    const int t = (int)(i / D), d = (int)(i % D);
    if (t == 0) { cur[d] = cls[d] + cls_pos[d]; return; }
    const int p = t - 1, cell = p / 4, sub = p % 4;
    const int py = cell / gw * 2 + sub / 2, px = cell % gw * 2 + sub % 2;
    const float y = (py + 0.5f) * ((float)G / (gh * 2)) - 0.5f;
    const float x = (px + 0.5f) * ((float)G / (gw * 2)) - 0.5f;
    const int y0 = max(0, min(G - 1, (int)floorf(y))), y1 = min(G - 1, y0 + 1);
    const int x0 = max(0, min(G - 1, (int)floorf(x))), x1 = min(G - 1, x0 + 1);
    const float wy = fmaxf(0, fminf(1, y - y0)), wx = fmaxf(0, fminf(1, x - x0));
    cur[i] += (1-wy)*(1-wx)*pos[((size_t)y0*G+x0)*D+d] + (1-wy)*wx*pos[((size_t)y0*G+x1)*D+d]
              + wy*(1-wx)*pos[((size_t)y1*G+x0)*D+d] + wy*wx*pos[((size_t)y1*G+x1)*D+d];
}

__global__ void rope_table(float* rc, float* rs, int S, int gw, float theta) {
    const int i = blockIdx.x * 256 + threadIdx.x;
    if (i >= S * 32) return;
    const int t = i / 32, pair = i % 32;
    const int p = t - 1, cell = p / 4, sub = p % 4;
    const int ph = t ? cell / gw * 2 + sub / 2 + 1 : 0;
    const int pw = t ? cell % gw * 2 + sub % 2 + 1 : 0;
    const float freq = 1.0f / powf(theta, (float)(2 * (pair % 16)) / 32);
    const float a = (pair < 16 ? pw : ph) * freq;
    float s, c;
    sincosf(a, &s, &c);
    rc[t*64+pair*2] = rc[t*64+pair*2+1] = c;
    rs[t*64+pair*2] = rs[t*64+pair*2+1] = s;
}

__global__ void rope(float* q, float* k, const float* rc, const float* rs, int S, int H) {
    const int i = blockIdx.x * 256 + threadIdx.x;
    if (i >= S * H * 32) return;
    const int j = (i % 32) * 2, t = i / (H * 32);
    const size_t off = (size_t)(i / 32) * 64 + j;
    const float c = rc[t*64+j], s = rs[t*64+j];
    const float q0 = q[off], q1 = q[off+1], k0 = k[off], k1 = k[off+1];
    q[off] = q0*c-q1*s; q[off+1] = q1*c+q0*s;
    k[off] = k0*c-k1*s; k[off+1] = k1*c+k0*s;
}

// One CUDA warp per query/head, two channels per lane. Unlike HIP's two-wave block, no block-wide
// barriers or S-by-S storage are needed. Online softmax retains just two accumulators per lane.
__global__ void attention(const float* Q, const float* K, const float* V, float* O, int S, int H) {
    const int h = blockIdx.x, qi = blockIdx.y * 4 + threadIdx.x / 32, d = threadIdx.x & 31;
    if (qi >= S) return;
    const size_t qo = ((size_t)qi * H + h) * 64 + d;
    const float q0 = Q[qo] * 0.125f, q1 = Q[qo+32] * 0.125f;
    float mx = -1e30f, den = 0, o0 = 0, o1 = 0;
    for (int ki = 0; ki < S; ++ki) {
        const size_t off = ((size_t)ki * H + h) * 64 + d;
        float dot = warp_sum(q0*K[off] + q1*K[off+32]);
        const float score = __shfl_sync(0xffffffffu, dot, 0);
        const float nm = fmaxf(mx, score), a = expf(mx-nm), b = expf(score-nm);
        o0 = fmaf(o0, a, b*V[off]); o1 = fmaf(o1, a, b*V[off+32]);
        den = fmaf(den, a, b); mx = nm;
    }
    O[qo] = o0 / den; O[qo+32] = o1 / den;
}
} // namespace cuda_vision_detail

class CudaVisionEncoder {
public:
    CudaVisionEncoder() = default;
    CudaVisionEncoder(const CudaVisionEncoder&) = delete;
    CudaVisionEncoder& operator=(const CudaVisionEncoder&) = delete;
    ~CudaVisionEncoder() { clear(); }
    bool loaded() const noexcept { return loaded_; }
    size_t device_bytes() const noexcept { return weight_bytes_ + arena_bytes_ + blas_bytes_; }

    void clear() noexcept {
        int previous = 0;
        (void)cudaGetDevice(&previous);
        (void)cudaSetDevice(gpu_);
        if (stream_) (void)cudaStreamSynchronize(stream_);
        if (blas_) (void)cublasDestroy(blas_);
        if (arena_) (void)cudaFree(arena_);
        if (blas_workspace_) (void)cudaFree(blas_workspace_);
        for (void* p : weights_) if (p) (void)cudaFree(p);
        if (stream_) (void)cudaStreamDestroy(stream_);
        weights_.clear(); layers_.clear();
        stream_ = nullptr; blas_ = nullptr; arena_ = nullptr; blas_workspace_ = nullptr;
        weight_bytes_ = arena_bytes_ = blas_bytes_ = capacity_ = 0;
        loaded_ = false;
        (void)cudaGetLastError();
        (void)cudaSetDevice(previous);
    }

    bool load(const std::string& path, int gpu, std::string& err) {
        clear(); gpu_ = gpu;
        auto fail = [&]() { clear(); return false; };
        try {
            if (!ok(cudaSetDevice(gpu_), "select vision GPU", err)) return fail();
            cudaDeviceProp prop{};
            if (!ok(cudaGetDeviceProperties(&prop, gpu_), "query vision GPU", err)) return fail();
            if (prop.major < 8) { err = "CUDA BF16 vision requires sm80 or newer"; return fail(); }
            if (!vision_probe(path, info_, err)) return fail();
            const auto& c = info_.c;
            if (c.hidden != 1024 || c.inter != 4096 || c.layers != 23 || c.heads != 16 || c.head_dim != 64 ||
                c.patch != 20 || c.merge != 2 || c.out_hidden != 2304 || c.proj_hidden != 4096) {
                err = "unsupported Inkling vision geometry"; return fail();
            }
            if (info_.is_safetensors) { err = "CUDA vision requires the native .strata vision shard"; return fail(); }
            strata::artifact::StrataFileReader rd;
            std::vector<uint8_t> section;
            if (!rd.open(path, err) || !rd.read_section(1, section, err)) return fail();
            auto upload = [&](const std::string& name, size_t n, bool matrix) -> void* {
                const VitTensor* t = info_.find(name);
                if (!t) { err = "missing vision tensor " + name; return nullptr; }
                size_t count = 1;
                for (int64_t d : t->shape) { if (d <= 0 || count > SIZE_MAX/(size_t)d) {err = "invalid tensor shape"; return nullptr;} count *= (size_t)d; }
                const size_t elem = t->fmt == F_F32 ? 4 : 2;
                if (count != n || t->bytes != n*elem || t->off > section.size() || t->bytes > section.size()-t->off ||
                    (t->fmt != F_F32 && t->fmt != F_BF16 && t->fmt != F_F16)) {
                    err = "invalid vision tensor layout " + name; return nullptr;
                }
                const uint8_t* src = section.data() + t->off;
                std::vector<float> floats;
                std::vector<uint16_t> bf;
                const void* data = src;
                const size_t bytes = n * (matrix ? 2 : 4);
                if (!(matrix && t->fmt == F_BF16) && !(!matrix && t->fmt == F_F32)) {
                    if (matrix) bf.resize(n); else floats.resize(n);
                    for (size_t i = 0; i < n; ++i) {
                        float f;
                        if (t->fmt == F_BF16) f = bf16_to_f32(reinterpret_cast<const uint16_t*>(src)[i]);
                        else if (t->fmt == F_F16) f = fp16_to_f32(reinterpret_cast<const uint16_t*>(src)[i]);
                        else std::memcpy(&f, src+i*4, 4);
                        if (matrix) bf[i] = f32_to_bf16(f); else floats[i] = f;
                    }
                    data = matrix ? (const void*)bf.data() : (const void*)floats.data();
                }
                weights_.push_back(nullptr);
                if (!ok(cudaMalloc(&weights_.back(), bytes), "vision weights", err)) return nullptr;
                weight_bytes_ += bytes;
                if (!ok(cudaMemcpy(weights_.back(), data, bytes, cudaMemcpyHostToDevice), "upload vision weights", err)) return nullptr;
                return weights_.back();
            };
            const int D = c.hidden;
            const std::string base = "model.visual.backbone.";
            auto f = [&](const std::string& name, size_t n) { return (float*)upload(name, n, false); };
            auto w = [&](const std::string& name, size_t n) { return (__nv_bfloat16*)upload(name, n, true); };
            cls_ = f(base+"embeddings.class_embedding", D);
            cls_pos_ = f(base+"embeddings.cls_position_embedding", D);
            pos_ = f(base+"embeddings.position_embedding", (size_t)c.posemb_grid*c.posemb_grid*D);
            patch_ = w(base+"embeddings.patch_embedding.weight", (size_t)D*1200);
            pre_w_ = f(base+"ln_pre.weight", D); pre_b_ = f(base+"ln_pre.bias", D);
            proj_in_ = w("model.visual.proj_in.weight", (size_t)c.proj_hidden*4*D);
            proj_out_ = w("model.visual.proj_out.weight", (size_t)c.out_hidden*c.proj_hidden);
            if (!cls_ || !cls_pos_ || !pos_ || !patch_ || !pre_w_ || !pre_b_ || !proj_in_ || !proj_out_) return fail();
            layers_.resize(c.layers);
            for (int i = 0; i < c.layers; ++i) {
                const std::string p = base+"layers."+std::to_string(i);
                auto& l = layers_[i];
                l.n1w=f(p+".ln_1.weight",D); l.n1b=f(p+".ln_1.bias",D);
                l.n2w=f(p+".ln_2.weight",D); l.n2b=f(p+".ln_2.bias",D);
                l.q=w(p+".self_attn.q_proj.weight",(size_t)D*D); l.qb=f(p+".self_attn.q_proj.bias",D);
                l.k=w(p+".self_attn.k_proj.weight",(size_t)D*D); l.kb=f(p+".self_attn.k_proj.bias",D);
                l.v=w(p+".self_attn.v_proj.weight",(size_t)D*D); l.vb=f(p+".self_attn.v_proj.bias",D);
                l.o=w(p+".self_attn.out_proj.weight",(size_t)D*D); l.ob=f(p+".self_attn.out_proj.bias",D);
                l.fc=w(p+".mlp.c_fc.weight",(size_t)c.inter*D); l.fcb=f(p+".mlp.c_fc.bias",c.inter);
                l.down=w(p+".mlp.c_proj.weight",(size_t)D*c.inter); l.downb=f(p+".mlp.c_proj.bias",D);
                l.ls1=f(p+".ls_1.gamma",D); l.ls2=f(p+".ls_2.gamma",D);
                if (!l.n1w || !l.n1b || !l.n2w || !l.n2b || !l.q || !l.qb || !l.k || !l.kb || !l.v || !l.vb ||
                    !l.o || !l.ob || !l.fc || !l.fcb || !l.down || !l.downb || !l.ls1 || !l.ls2) return fail();
            }
            if (!ok(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking), "vision stream", err)) return fail();
            if (!blas_ok(cublasCreate(&blas_), "vision cuBLAS", err) ||
                !blas_ok(cublasSetStream(blas_, stream_), "vision cuBLAS stream", err)) return fail();
            const size_t bytes = 4u << 20;
            if (!ok(cudaMalloc(&blas_workspace_, bytes), "vision cuBLAS workspace", err)) return fail();
            blas_bytes_ = bytes;
            if (!blas_ok(cublasSetWorkspace(blas_, blas_workspace_, bytes), "set vision cuBLAS workspace", err)) return fail();
            loaded_ = true;
            return true;
        } catch (const std::exception& e) { err = std::string("vision load: ")+e.what(); return fail(); }
    }

    bool encode(const ImageRec& img, std::vector<float>& out, std::string& err) {
        using namespace cuda_vision_detail;
        if (!loaded_) { err = "CUDA vision encoder not loaded"; return false; }
        auto fail = [&]() {
            if (stream_) (void)cudaStreamSynchronize(stream_);
            if (arena_) (void)cudaFree(arena_);
            arena_ = nullptr; arena_bytes_ = capacity_ = 0;
            (void)cudaGetLastError();
            out.clear(); return false;
        };
        try {
            if (!ok(cudaSetDevice(gpu_), "select vision GPU", err)) return fail();
            const auto& c = info_.c;
            const int nt = img.num_lm_tokens(), D = c.hidden, H = c.heads;
            if (img.grid_h <= 0 || img.grid_w <= 0 || nt > 1048576 ||
                img.pixel_values.size() != (size_t)nt*4800) {err="invalid prepared vision image"; return false;}
            const int S = nt*4+1, np = S-1;
            const size_t sd = (size_t)S*D, patch_n = (size_t)np*1200;
            const size_t convert_n = std::max({sd, (size_t)S*c.inter, patch_n, (size_t)nt*4096});
            // A single arena grows only for a larger image. No LLM scratch or image-sized host intermediates.
            const size_t floats = (size_t)nt*4800 + patch_n + 7*sd + (size_t)S*c.inter +
                                  (size_t)S*128 + (size_t)nt*(c.proj_hidden+c.out_hidden);
            const size_t need = floats*4 + convert_n*2;
            if (need > capacity_) {
                if (arena_) { if (!ok(cudaFree(arena_), "free vision workspace", err)) return fail(); arena_ = nullptr; }
                arena_bytes_ = capacity_ = 0;
                if (!ok(cudaMalloc(&arena_, need), "vision image workspace", err)) return fail();
                arena_bytes_ = capacity_ = need;
            }
            float* ptr = (float*)arena_;
            auto take = [&](size_t n) {float* p=ptr; ptr+=n; return p;};
            float* pixels=take((size_t)nt*4800), *patches=take(patch_n), *cur=take(sd), *norm=take(sd);
            float* q=take(sd), *k=take(sd), *v=take(sd), *ao=take(sd), *proj=take(sd);
            float* mlp=take((size_t)S*c.inter), *rc=take((size_t)S*64), *rs=take((size_t)S*64);
            float* mid=take((size_t)nt*c.proj_hidden), *output=take((size_t)nt*c.out_hidden);
            auto* converted = (__nv_bfloat16*)ptr;
            if (!ok(cudaMemcpyAsync(pixels, img.pixel_values.data(), img.pixel_values.size()*4, cudaMemcpyHostToDevice, stream_), "upload image pixels", err)) return fail();
            patchify<<<grid(patch_n),256,0,stream_>>>(pixels,patches,patch_n);
            if (!linear(patch_,patches,nullptr,cur+D,np,D,1200,converted,err)) return fail();
            position<<<grid(sd),256,0,stream_>>>(cur,cls_,cls_pos_,pos_,S,D,img.grid_h,img.grid_w,c.posemb_grid);
            layernorm<<<S,256,0,stream_>>>(cur,pre_w_,pre_b_,cur,D,c.ln_eps);
            rope_table<<<grid((size_t)S*32),256,0,stream_>>>(rc,rs,S,img.grid_w,c.rope_theta);
            for (const Layer& l : layers_) {
                layernorm<<<S,256,0,stream_>>>(cur,l.n1w,l.n1b,norm,D,c.ln_eps);
                // One normalized input conversion feeds Q, K and V.
                convert_bf16<<<grid(sd),256,0,stream_>>>(norm,converted,sd);
                if (!gemm(l.q,converted,l.qb,q,S,D,D,err) || !gemm(l.k,converted,l.kb,k,S,D,D,err) ||
                    !gemm(l.v,converted,l.vb,v,S,D,D,err)) return fail();
                rope<<<grid((size_t)S*H*32),256,0,stream_>>>(q,k,rc,rs,S,H);
                attention<<<dim3(H,(S+3)/4),128,0,stream_>>>(q,k,v,ao,S,H);
                if (!linear(l.o,ao,l.ob,proj,S,D,D,converted,err)) return fail();
                residual<<<grid(sd),256,0,stream_>>>(cur,proj,l.ls1,sd,D);
                layernorm<<<S,256,0,stream_>>>(cur,l.n2w,l.n2b,norm,D,c.ln_eps);
                if (!linear(l.fc,norm,l.fcb,mlp,S,c.inter,D,converted,err)) return fail();
                gelu<<<grid((size_t)S*c.inter),256,0,stream_>>>(mlp,(size_t)S*c.inter);
                if (!linear(l.down,mlp,l.downb,proj,S,D,c.inter,converted,err)) return fail();
                residual<<<grid(sd),256,0,stream_>>>(cur,proj,l.ls2,sd,D);
            }
            if (!linear(proj_in_,cur+D,nullptr,mid,nt,c.proj_hidden,4*D,converted,err)) return fail();
            gelu<<<grid((size_t)nt*c.proj_hidden),256,0,stream_>>>(mid,(size_t)nt*c.proj_hidden);
            if (!linear(proj_out_,mid,nullptr,output,nt,c.out_hidden,c.proj_hidden,converted,err)) return fail();
            if (!ok(cudaGetLastError(), "vision kernel launch", err)) return fail();
            out.resize((size_t)nt*c.out_hidden);
            if (!ok(cudaMemcpyAsync(out.data(),output,out.size()*4,cudaMemcpyDeviceToHost,stream_), "vision embeddings D2H", err) ||
                !ok(cudaStreamSynchronize(stream_), "vision encode synchronization", err)) return fail();
            return true;
        } catch (const std::exception& e) {err=std::string("vision encode: ")+e.what(); return fail();}
    }

private:
    struct Layer {
        float *n1w=nullptr,*n1b=nullptr,*n2w=nullptr,*n2b=nullptr,*qb=nullptr,*kb=nullptr,*vb=nullptr,*ob=nullptr;
        float *fcb=nullptr,*downb=nullptr,*ls1=nullptr,*ls2=nullptr;
        __nv_bfloat16 *q=nullptr,*k=nullptr,*v=nullptr,*o=nullptr,*fc=nullptr,*down=nullptr;
    };
    static unsigned grid(size_t n) { return (unsigned)((n+255)/256); }
    static bool ok(cudaError_t status, const char* what, std::string& err) {
        if (status == cudaSuccess) return true;
        err=std::string(what)+": "+cudaGetErrorString(status); return false;
    }
    static bool blas_ok(cublasStatus_t status, const char* what, std::string& err) {
        if (status == CUBLAS_STATUS_SUCCESS) return true;
        err=std::string(what)+": cuBLAS status "+std::to_string((int)status); return false;
    }
    bool gemm(const __nv_bfloat16* w, const __nv_bfloat16* x, const float* b, float* y,
              int M, int N, int K, std::string& err) {
        const float alpha=1, beta=0;
        if (!blas_ok(cublasGemmEx(blas_,CUBLAS_OP_T,CUBLAS_OP_N,N,M,K,&alpha,w,CUDA_R_16BF,K,
                                  x,CUDA_R_16BF,K,&beta,y,CUDA_R_32F,N,CUBLAS_COMPUTE_32F,
                                  CUBLAS_GEMM_DEFAULT_TENSOR_OP),"vision BF16 GEMM",err)) return false;
        if (b) cuda_vision_detail::bias<<<grid((size_t)M*N),256,0,stream_>>>(y,b,(size_t)M*N,N);
        return true;
    }
    bool linear(const __nv_bfloat16* w, const float* x, const float* b, float* y,
                int M, int N, int K, __nv_bfloat16* converted, std::string& err) {
        cuda_vision_detail::convert_bf16<<<grid((size_t)M*K),256,0,stream_>>>(x,converted,(size_t)M*K);
        return gemm(w,converted,b,y,M,N,K,err);
    }
    int gpu_=0;
    bool loaded_=false;
    VisionShardInfo info_;
    cudaStream_t stream_=nullptr;
    cublasHandle_t blas_=nullptr;
    void* arena_=nullptr;
    void* blas_workspace_=nullptr;
    size_t weight_bytes_=0,arena_bytes_=0,blas_bytes_=0,capacity_=0;
    std::vector<void*> weights_;
    std::vector<Layer> layers_;
    float *cls_=nullptr,*cls_pos_=nullptr,*pos_=nullptr,*pre_w_=nullptr,*pre_b_=nullptr;
    __nv_bfloat16 *patch_=nullptr,*proj_in_=nullptr,*proj_out_=nullptr;
};
} // namespace ink2
