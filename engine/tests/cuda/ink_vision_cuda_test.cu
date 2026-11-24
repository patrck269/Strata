#include "cuda/ink_cuda_vision.cuh"
#include <array>
#include <cstdio>
#include <stdexcept>

namespace {
void require(bool value, const std::string& why) { if (!value) throw std::runtime_error(why); }
ink2::ImageRec fixture(int width, int height) {
    ink2::ImageRec image;
    image.orig_w = image.padded_w = width;
    image.orig_h = image.padded_h = height;
    image.grid_w = width / 40;
    image.grid_h = height / 40;
    image.pixel_values.resize((size_t)image.num_lm_tokens() * 4800);
    for (int gr = 0; gr < image.grid_h; ++gr)
        for (int gc = 0; gc < image.grid_w; ++gc)
            for (int y = 0; y < 40; ++y)
                for (int x = 0; x < 40; ++x) {
                    const int gx = gc*40+x, gy = gr*40+y;
                    const size_t i = (((size_t)gr*image.grid_w+gc)*1600+y*40+x)*3;
                    image.pixel_values[i] = 2.0f*gx/(width-1)-1.0f;
                    image.pixel_values[i+1] = 2.0f*gy/(height-1)-1.0f;
                    image.pixel_values[i+2] = ((gx/10+gy/10)%2) ? 0.75f : -0.5f;
                }
    return image;
}
void parity(const std::vector<float>& ref, const std::vector<float>& actual) {
    require(ref.size() == actual.size() && !ref.empty(), "embedding shape mismatch");
    double dot=0, rr=0, aa=0, diff=0, max_abs=0;
    for (size_t i=0; i<ref.size(); ++i) {
        require(std::isfinite(ref[i]) && std::isfinite(actual[i]), "nonfinite vision embedding");
        const double r=ref[i], a=actual[i], d=r-a;
        dot+=r*a; rr+=r*r; aa+=a*a; diff+=d*d; max_abs=std::max(max_abs,std::abs(d));
    }
    require(rr>0 && aa>0, "zero vision embedding norm");
    const double cosine=dot/std::sqrt(rr*aa), relative_l2=std::sqrt(diff/rr);
    std::printf("VISION_PARITY elements=%zu cosine=%.9f relative_l2=%.9f max_abs=%.9f\n",ref.size(),cosine,relative_l2,max_abs);
    require(cosine>=0.999 && relative_l2<=0.03, "CUDA/CPU numerical parity gate failed");
}
}
int main(int argc, char** argv) {
    if (argc != 2) { std::fprintf(stderr,"SKIP: supply actual Inkling-2-Mini-Vision.strata\n"); return 77; }
    try {
        std::string err;
        ink2::CpuVisionEncoder cpu;
        ink2::CudaVisionEncoder gpu;
        require(cpu.load(argv[1],err),err);
        require(gpu.load(argv[1],0,err),err);
        const size_t weights=gpu.device_bytes();
        const std::array<std::array<int,2>,3> shapes{{{{40,40}},{{80,120}},{{160,120}}}};
        std::array<std::vector<float>,3> references, outputs;
        for (size_t i=0; i<shapes.size(); ++i) {
            auto image=fixture(shapes[i][0],shapes[i][1]);
            require(cpu.encode(image,references[i],err),err);
            require(gpu.encode(image,outputs[i],err),err);
            require(outputs[i].size()==(size_t)image.num_lm_tokens()*2304,"wrong LM embedding shape");
            std::printf("VISION_CASE width=%d height=%d bytes=%zu\n",shapes[i][0],shapes[i][1],gpu.device_bytes());
            parity(references[i],outputs[i]);
        }
        const size_t retained=gpu.device_bytes();
        require(retained>weights,"vision workspace missing");
        for (int i : {0,2,1,0}) {
            std::vector<float> repeated;
            require(gpu.encode(fixture(shapes[i][0],shapes[i][1]),repeated,err),err);
            parity(references[i],repeated);
            require(repeated==outputs[i],"reused CUDA image workspace changed embeddings");
            require(gpu.device_bytes()==retained,"workspace grew without a larger image");
        }
        gpu.clear();
        require(gpu.device_bytes()==0,"vision clear retained device memory");
        std::puts("ink_vision_cuda_test: PASS");
        return 0;
    } catch (const std::exception& e) { std::fprintf(stderr,"ink_vision_cuda_test: %s\n",e.what()); return 1; }
}
