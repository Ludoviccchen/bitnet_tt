// Data read kernel (data movement kernel 0)
#include <dataflow_api.h>

void kernel_main() {
    // Read parameters from the kernel run-time arguments
    const uint32_t a_addr = get_arg_val<uint32_t>(0);
    const uint32_t b_addr = get_arg_val<uint32_t>(1);
    const uint32_t n_tiles = get_arg_val<uint32_t>(2);

    constexpr auto cb_in0 = tt::CBIndex::c_0;
    constexpr auto cb_in1 = tt::CBIndex::c_1;

    const uint32_t tile_size_bytes = get_tile_size(cb_in0);

    constexpr auto args_a = TensorAccessorArgs<0>();
    const auto a = TensorAccessor(args_a, a_addr, tile_size_bytes);

    constexpr auto args_b = TensorAccessorArgs<args_a.next_compile_time_args_offset()>();
    const auto b = TensorAccessor(args_b, b_addr, tile_size_bytes);

    // Read inputs from DRAM into circular buffers
    for (uint32_t i = 0; i < n_tiles; i++) {
        cb_reserve_back(cb_in0, 1);
        cb_reserve_back(cb_in1, 1);

        const uint32_t cb_in0_addr = get_write_ptr(cb_in0);
        const uint32_t cb_in1_addr = get_write_ptr(cb_in1);
        noc_async_read_page(i, a, cb_in0_addr);
        noc_async_read_page(i, b, cb_in1_addr);

        noc_async_read_barrier();  // Wait until tile reads are done

        cb_push_back(cb_in0, 1);
        cb_push_back(cb_in1, 1);
    }
}