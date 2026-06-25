// Compute kernel
#include <compute_kernel_api.h>
#include <compute_kernel_api/common.h>
#include <compute_kernel_api/eltwise_binary.h>

void kernel_main() {
    const uint32_t n_tiles = get_arg_val<uint32_t>(0);
    constexpr auto cb_in0 = tt::CBIndex::c_0;
    constexpr auto cb_in1 = tt::CBIndex::c_1;
    constexpr auto cb_out = tt::CBIndex::c_16;
    constexpr uint32_t dst_reg = 0;

    // Metalium API Calls                              Involved Cores
    binary_op_init_common(cb_in0, cb_in1, cb_out);  // Unpack, Math, Pack
    add_tiles_init(cb_in0, cb_in1, false);          // Unpack, Math

    for (uint32_t i = 0; i < n_tiles; i++) {
        cb_wait_front(cb_in0, 1);                   // Unpack
        cb_wait_front(cb_in1, 1);                   // Unpack

        tile_regs_acquire();                        // Math
        add_tiles(cb_in0, cb_in1, 0, 0, dst_reg);   // Unpack, Math
        tile_regs_commit();                         // Math

        cb_pop_front(cb_in0, 1);                    // Unpack
        cb_pop_front(cb_in1, 1);                    // Unpack

        cb_reserve_back(cb_out, 1);                 // Pack

        tile_regs_wait();                           // Pack
        pack_tile(dst_reg, cb_out, 0);              // Pack
        tile_regs_release();                        // Pack

        cb_push_back(cb_out, 1);                    // Pack
    }
}