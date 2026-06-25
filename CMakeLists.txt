cmake_minimum_required(VERSION 3.14)
project(bitnet_tt_blackhole CXX)

# Tenstorrent utilise le standard C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 1. Vérification de l'environnement
set(TT_METAL_HOME $ENV{TT_METAL_HOME})
if(NOT TT_METAL_HOME)
    message(FATAL_ERROR "La variable TT_METAL_HOME n'est pas définie. Fais un 'source ~/.bashrc' ou un export manuel.")
endif()

# 2. Recherche de Pybind11 (installé via Conda)
find_package(pybind11 REQUIRED)

# 3. Chemins vers les en-têtes (Headers) de Tenstorrent
# On inclut la racine et les dossiers d'API
include_directories(
    ${TT_METAL_HOME}
    ${TT_METAL_HOME}/tt_metal/include
    ${TT_METAL_HOME}/tt_metal/third_party/fmt/include # Très souvent requis par les logs TT
)

# 4. Chemins vers les bibliothèques compilées de Tenstorrent (.so / .a)
link_directories(${TT_METAL_HOME}/build/lib)

# ==========================================
# CIBLE 1 : L'exécutable C++ natif
# ==========================================
add_executable(host_vecadd src/host_vecadd.cpp)
# On lie notre exécutable à la librairie tt_metal
target_link_libraries(host_vecadd PRIVATE tt_metal)

# ==========================================
# CIBLE 2 : Le module Python (Pybind11)
# ==========================================
# Cela va créer un fichier custom_bitnet_ops.so importable en Python
pybind11_add_module(custom_bitnet_ops src/pybind_vecadd.cpp)
target_link_libraries(custom_bitnet_ops PRIVATE tt_metal)