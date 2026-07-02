import torch
import ttnn

def main():
    print("Initialisation de la carte Tenstorrent Blackhole...")
    
    # 1. Ouvrir la connexion avec le périphérique (Device 0)
    device_id = 0
    device = ttnn.open_device(device_id=device_id)
    
    try:
        # 2. Créer des données sur le processeur hôte (CPU) via PyTorch
        print("Création d'un tenseur de test sur le CPU...")
        # On utilise bfloat16, le format de prédilection des accélérateurs IA
        torch_input = torch.randn((1, 1, 32, 32), dtype=torch.bfloat16)
        
        # 3. Transférer les données vers la mémoire DRAM de la Blackhole
        print("Transfert du tenseur vers la carte (Host -> Device)...")
        # Le layout TILE_LAYOUT est requis par Tenstorrent pour les calculs matriciels
        tt_input = ttnn.from_torch(
            torch_input, 
            layout=ttnn.TILE_LAYOUT, 
            device=device
        )
        
        # 4. Exécuter une opération mathématique sur le matériel
        print("Exécution de l'opération GELU sur l'accélérateur...")
        tt_output = ttnn.gelu(tt_input)
        
        # 5. Récupérer le résultat sur le processeur hôte
        print("Récupération du résultat (Device -> Host)...")
        torch_output = ttnn.to_torch(tt_output)
        
        print("\n--- Succès ! ---")
        print(f"Shape du tenseur de sortie : {torch_output.shape}")
        print("La communication avec la Blackhole p150a est parfaitement fonctionnelle !")
        
    finally:
        # 6. Toujours refermer le périphérique pour éviter les fuites de mémoire
        print("\nLibération des ressources de la carte.")
        ttnn.close_device(device)

if __name__ == "__main__":
    main()