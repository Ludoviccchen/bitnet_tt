import torch
import ttnn
# On importe la librairie C++ que tu auras compilée avec pybind11
import custom_bitnet_ops 

def test_custom_vector_addition():
    print("🚀 Test du Custom Kernel C++ depuis Python...")

    # 1. On ouvre la carte (géré par ttnn)
    device = ttnn.open_device(device_id=0)

    # 2. Création des données de test via PyTorch
    # On crée deux matrices de 32x32 remplies de 1.0 et 2.0
    tensor_a_pt = torch.ones((1, 1, 32, 32), dtype=torch.bfloat16)
    tensor_b_pt = torch.full((1, 1, 32, 32), 2.0, dtype=torch.bfloat16)

    # Transfert vers la carte Blackhole
    tensor_a_tt = ttnn.from_torch(tensor_a_pt, layout=ttnn.TILE_LAYOUT, device=device)
    tensor_b_tt = ttnn.from_torch(tensor_b_pt, layout=ttnn.TILE_LAYOUT, device=device)

    # 3. L'APPEL MAGIQUE : On appelle TON kernel C++ (qui orchestre les fichiers RISC-V)
    # custom_bitnet_ops.vecadd_1d fera le pont avec le code C++ vu plus haut
    tensor_c_tt = custom_bitnet_ops.vecadd_1d(tensor_a_tt, tensor_b_tt, device)

    # 4. Rapatriement et vérification
    tensor_c_pt = ttnn.to_torch(tensor_c_tt)
    
    # 1.0 + 2.0 devrait faire 3.0
    assert torch.allclose(tensor_c_pt, tensor_a_pt + tensor_b_pt), "Erreur de calcul dans le kernel !"
    print("✅ Succès ! Le kernel personnalisé a calculé l'addition parfaitement.")

    ttnn.close_device(device)

if __name__ == "__main__":
    test_custom_vector_addition()