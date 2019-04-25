// Implementation (nearly) that from Arbor.

#include <cstddef>
#include <cstdint>
#include <memory>

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 600
#error "Require compute capability >= 6.0"
#endif

namespace arb {
    constexpr unsigned mask_all = 0xFFFFFFFF;
    constexpr unsigned threads_per_warp = 32;

    // Return largest n = 2^k s.t. n <= i.
    // Precondition: i>0
    __device__ __inline__
    unsigned rounddown_power_of_2(std::uint32_t i) {
        return 1u<<(31u - __clz(i));
    }

    struct run_length {
        unsigned left;
        unsigned right;
        unsigned shift;
        unsigned lane_id;
        unsigned key_mask;

        __device__ __inline__
        bool is_root() const {
            return left == lane_id;
        }

        __device__
        run_length(int idx, unsigned mask) {
            key_mask = mask;
            lane_id = threadIdx.x%threads_per_warp;
            unsigned num_lanes = threads_per_warp-__clz(key_mask);

            auto right_limit = [num_lanes] (unsigned roots, unsigned shift) {
                unsigned zeros_right  = __ffs(roots>>shift);
                return zeros_right ? shift -1 + zeros_right : num_lanes;
            };

            // determine if this thread is the root (i.e. first thread with this key)
            int left_idx  = __shfl_up_sync(key_mask, idx, lane_id? 1: 0);
            int is_root = 1;
            if(lane_id>0) {
                is_root = (left_idx != idx);
            }

            // determine the range this thread contributes to
            unsigned roots = __ballot_sync(key_mask, is_root);

            // determine the bounds of the lanes with the same key as idx
            right = right_limit(roots, lane_id+1);
            left  = threads_per_warp-1-right_limit(__brev(roots), threads_per_warp-1-lane_id);

            // find the largest power of two that is less than or equal to the run length
            shift = rounddown_power_of_2(right - left);
        }
    };

    template <typename T, typename I>
    __global__
    void reduce_impl(std::size_t N, T* p, const T* v, const I* index) {
        unsigned tid = threadIdx.x+blockIdx.x*blockDim.x;
        unsigned mask = __ballot_sync(mask_all, tid<N);

        if (tid<N) {
            auto contribution = v[tid];
            auto i = index[tid];

            run_length run(i, mask);
            unsigned shift = run.shift;
            const unsigned key_lane = run.lane_id - run.left;

            bool participate = run.shift && run.lane_id+shift<run.right;

            while (__any_sync(run.key_mask, shift)) {
                const unsigned w = participate? shift: 0;
                const unsigned source_lane = run.lane_id + w;

                T source_value = __shfl_sync(run.key_mask, contribution, source_lane);
                if (participate) {
                    contribution += source_value;
                }

                shift >>= 1;
                participate = key_lane<shift;
            }

            if (run.is_root()) {
                atomicAdd(p+i, contribution);
            }
        }
    }
} // namespace arb

namespace expr1 {
    constexpr unsigned mask_all = 0xFFFFFFFF;
    constexpr unsigned threads_per_warp = 32;
    constexpr unsigned log2_tpw = 5;

    template <typename T, typename I>
    __global__
    void reduce_impl(std::size_t N, T* p, const T* v, const I* index) {
        unsigned tid = threadIdx.x+blockIdx.x*blockDim.x;

        unsigned mask = __ballot_sync(mask_all, tid<N);
        if (tid>=N) return;

	unsigned lane_id = threadIdx.x%threads_per_warp;

	I i = index[tid];
	auto i_prev = __shfl_up_sync(mask, i, 1);

	bool root = lane_id==0 || i!=i_prev;
	unsigned roots = __ballot_sync(mask, root);
	roots |= ~mask;

	unsigned width = __ffs(roots>>lane_id>>1);
	if (!width) width=threads_per_warp-lane_id;

	auto v_local = v[tid];
	auto accum = v_local;
	unsigned shift = 1;

	for (unsigned k = 0; k<log2_tpw; ++k) {
	    if (shift>=width) shift = 0;
	    T v_shift = __shfl_down_sync(mask, accum, shift);

	    if (shift) accum += v_shift;
	    shift *= 2;
	}

	if (root) atomicAdd(p+i, accum);
    }
}

namespace expr2 {
    constexpr unsigned mask_all = 0xFFFFFFFF;
    constexpr unsigned threads_per_warp = 32;
    constexpr unsigned log2_tpw = 5;

    template <typename T, typename I>
    __global__
    void reduce_impl(std::size_t N, T* p, const T* v, const I* index) {
        unsigned tid = threadIdx.x+blockIdx.x*blockDim.x;

        unsigned mask = __ballot_sync(mask_all, tid<N);
        if (tid>=N) return;

	unsigned lane_id = threadIdx.x%threads_per_warp;

	I i = index[tid];
	auto i_prev = __shfl_up_sync(mask, i, 1);

	bool root = lane_id==0 || i!=i_prev;
	unsigned roots = __ballot_sync(mask, root);
	roots |= ~mask;

        auto v_local = v[tid];

        if (roots!=-1) {
            unsigned width = __ffs(roots>>lane_id>>1);
            if (!width) width=threads_per_warp-lane_id;

            auto accum = v_local;
            unsigned shift = 1;

            while (__any_sync(mask, shift)) {
                if (shift>=width) shift = 0;
                T v_shift = __shfl_down_sync(mask, accum, shift);

                if (shift) accum += v_shift;
                shift *= 2;
            }

            if (root) atomicAdd(p+i, accum);
        }
        else {
            atomicAdd(p+i, v_local);
        }
    }
}


template <typename T>
struct block {
    T* data = nullptr;
    std::size_t n = 0;
};

struct gpu_block_delete {
    template <typename T>
    void operator()(block<T>* bp) {
        cudaFree((void*)bp->data);
        delete bp;
    }
};

template <typename T>
using gpu_block_ptr = std::unique_ptr<block<T>, gpu_block_delete>;

template <typename T>
gpu_block_ptr<T> on_gpu(block<T> b) {
    void* gpu_data = nullptr;
    cudaMalloc(&gpu_data, b.n*sizeof(T));
    cudaMemcpy(gpu_data, b.data, b.n*sizeof(T), cudaMemcpyHostToDevice);

    auto p = gpu_block_ptr<T>(new block<T>);
    p->data = (T*)gpu_data;
    p->n = b.n;
    return p;
}

template <typename T>
void from_gpu(block<T> b, const gpu_block_ptr<T>& p) {
    if (b.n!=p->n) throw std::runtime_error("block size mismatch");
    cudaMemcpy(b.data, p->data, b.n*sizeof(T), cudaMemcpyDeviceToHost);
}

struct cuda_event_delete {
    void operator()(cudaEvent_t* evptr) {
	if (evptr) cudaEventDestroy(*evptr);
	delete evptr;
    }
};

using cuda_event = std::unique_ptr<cudaEvent_t, cuda_event_delete>;
cuda_event make_cuda_event() {
    cudaEvent_t* p = new cudaEvent_t;
    cudaEventCreate(p);
    return cuda_event(p);
}

float elapsed_time_s(const cuda_event& e0, const cuda_event& e1) {
    float t = 0;
    cudaEventElapsedTime(&t, *e0, *e1);
    return t*0.001;
}

using impl_fn_ptr = void (*)(std::size_t, double*, const double*, const int*);

float impl_harness(impl_fn_ptr fn, std::size_t N, double* p, const double* v, const int* index, int reps) {
    unsigned bwidth = 128;
    unsigned bcount = (N+bwidth-1)/bwidth;

    block<double> p_view{p, N};
    block<const double> v_view{v, N};
    block<const int> i_view{index, N};

    auto p_gpu = on_gpu(p_view);
    auto v_gpu = on_gpu(v_view);
    auto i_gpu = on_gpu(i_view);

    auto e0 = make_cuda_event();
    auto e1 = make_cuda_event();

    p_gpu = on_gpu(p_view);
    cudaEventRecord(*e0);
    for (int c = 0; c<reps; ++c) {
        (*fn)<<<bcount, bwidth>>>(N, p_gpu->data, v_gpu->data, i_gpu->data);
    }
    cudaEventRecord(*e1);

    from_gpu(p_view, p_gpu);
    return elapsed_time_s(e0, e1);
}

float arbor_cuda_reduce_impl(std::size_t N, double* p, const double* v, const int* index, int reps) {
    return impl_harness(&arb::reduce_impl<double, int>, N, p, v, index, reps);
}

float expr1_cuda_reduce_impl(std::size_t N, double* p, const double* v, const int* index, int reps) {
    return impl_harness(&expr1::reduce_impl<double, int>, N, p, v, index, reps);
}

float expr2_cuda_reduce_impl(std::size_t N, double* p, const double* v, const int* index, int reps) {
    return impl_harness(&expr2::reduce_impl<double, int>, N, p, v, index, reps);
}
