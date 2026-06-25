#include "tt_metal/host_api.hpp"
#include "tt_metal/impl/device/device.hpp"

using namespace tt;
using namespace tt::tt_metal;

int main(int argc, char **argv) {
    // 1. Initialisation de la carte (Device 0)
    int device_id = 0;
    Device *device = CreateDevice(device_id);

    // 2. Création du programme (Chef d'orchestre)
    Program program = CreateProgram();

    // On va exécuter ce kernel sur un seul coeur de calcul pour le test
    CoreCoord core = {0, 0}; 

    // 3. Allocation de la mémoire en DRAM (Mémoire principale de la carte)
    uint32_t single_tile_size = 2 * 32 * 32; // Format bfloat16 (2 octets par valeur)
    uint32_t num_tiles = 1;
    uint32_t buffer_size = single_tile_size * num_tiles;

    Buffer a_buffer = CreateBuffer(device, buffer_size, buffer_size, BufferType::DRAM);
    Buffer b_buffer = CreateBuffer(device, buffer_size, buffer_size, BufferType::DRAM);
    Buffer c_buffer = CreateBuffer(device, buffer_size, buffer_size, BufferType::DRAM);

    // 4. Configuration des Circular Buffers (SRAM / L1)
    CircularBufferConfig cb_in0_config = CircularBufferConfig(buffer_size, {{CBIndex::c_0, tt::DataFormat::Float16_b}}).set_page_size(CBIndex::c_0, single_tile_size);
    CreateCircularBuffer(program, core, cb_in0_config);

    CircularBufferConfig cb_in1_config = CircularBufferConfig(buffer_size, {{CBIndex::c_1, tt::DataFormat::Float16_b}}).set_page_size(CBIndex::c_1, single_tile_size);
    CreateCircularBuffer(program, core, cb_in1_config);

    CircularBufferConfig cb_out_config = CircularBufferConfig(buffer_size, {{CBIndex::c_16, tt::DataFormat::Float16_b}}).set_page_size(CBIndex::c_16, single_tile_size);
    CreateCircularBuffer(program, core, cb_out_config);

    // 5. Déclaration et compilation des noyaux RISC-V (Ceux que tu as écrits)
    KernelHandle reader_kernel = CreateKernel(
        program, "kernels/vecadd/reader_1d.cpp", core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});

    KernelHandle writer_kernel = CreateKernel(
        program, "kernels/vecadd/writer_1d.cpp", core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    KernelHandle compute_kernel = CreateKernel(
        program, "kernels/vecadd/compute_add.cpp", core,
        ComputeConfig{});

    // 6. Envoi des arguments d'exécution (Adresses mémoire et taille) aux noyaux
    SetRuntimeArgs(program, reader_kernel, core, {a_buffer.address(), b_buffer.address(), num_tiles});
    SetRuntimeArgs(program, writer_kernel, core, {c_buffer.address(), num_tiles});
    SetRuntimeArgs(program, compute_kernel, core, {num_tiles});

    // 7. Exécution ! (Dans la réalité, on remplit 'a' et 'b' avec EnqueueWriteBuffer avant)
    EnqueueProgram(device->command_queue(), program, false);
    Finish(device->command_queue()); // On attend que la carte ait fini

    // 8. Fermeture
    CloseDevice(device);
    return 0;
}