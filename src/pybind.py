#include <pybind11/pybind11.h>
#include "tt_metal/host_api.hpp"

namespace py = pybind11;
using namespace tt::tt_metal;

// Fonction C++ simple qui encapsule l'appel à la carte
void run_vecadd_on_device() {
    int device_id = 0;
    Device *device = CreateDevice(device_id);
    
    // (Ici tu pourras copier la logique de ton host_vecadd.cpp plus tard)
    // Pour l'instant on fait juste un print depuis le C++
    std::cout << "Kernel VecAdd déclenché depuis le backend C++ !" << std::endl;
    
    CloseDevice(device);
}

// C'est cette macro qui crée le module Python
PYBIND11_MODULE(custom_bitnet_ops, m) {
    m.doc() = "Librairie custom BitNet pour Tenstorrent Blackhole p150a"; // Docstring optionnel

    // On expose notre fonction C++ sous le nom 'vecadd_1d' en Python
    m.def("vecadd_1d", &run_vecadd_on_device, "Exécute l'addition de vecteurs sur la carte");
}