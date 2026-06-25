// Data write kernel (data movement kernel 1)
#include <dataflow_api.h>

void kernel_main() {
    const uint32_t c_addr = get_arg_val<uint32_t>(0);
    const uint32_t n_tiles = get_arg_val<uint32_t>(1);

    constexpr auto cb_out = tt::CBIndex::c_16;

    const uint32_t tile_size_bytes = get_tile_size(cb_out);

    constexpr auto args_c = TensorAccessorArgs<0>();
    const auto c = TensorAccessor(args_c, c_addr, tile_size_bytes);

    // Read outputs from circular buffers into DRAM
    for (uint32_t i = 0; i < n_tiles; i++) {
        cb_wait_front(cb_out, 1);

        const uint32_t cb_out_addr = get_read_ptr(cb_out);
        noc_async_write_page(i, c, cb_out_addr);

        noc_async_write_barrier();

        cb_pop_front(cb_out, 1);
    }
}