import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader
from SNN import SEENIC_SNN_Dataset, SpacecraftSNN

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Generating Uncertainty Plot on: {device}")

    # 1. Carica il Test Dataset (usa la sequenza MAI VISTA in fase di training)
    TEST_FOLDER = "hubble-approach-fast-lightboxdiffuser" 
    events_file = Path(TEST_FOLDER) / "events.csv"
    poses_file = Path(TEST_FOLDER) / "cam-poses.csv"

    test_dataset = SEENIC_SNN_Dataset(str(events_file), str(poses_file))
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # 2. Lista dei 5 modelli allenati in modo indipendente
    # Assicurati di aver rinominato i tuoi 5 salvataggi in questo modo
    model_paths = [
        
        "SNN_weights_run_1.pth",
        "SNN_weights_run_2.pth",
        "SNN_weights_run_3.pth",
        "SNN_weights_run_4.pth",
        "SNN_weights_run_5.pth",
        "SNN_weights_run_6.pth",
        "SNN_weights_run_7.pth",
        "SNN_weights_run_8.pth",
        "SNN_weights_run_9.pth",
        "SNN_weights_run_10.pth"
    ]

    all_predictions = []
    true_trajectory = []

    # 3. Raccogli le predizioni da tutti i 5 modelli
    for idx, path in enumerate(model_paths):
        print(f"Evaluating Model {idx+1}/10: {path}")
        model = SpacecraftSNN(beta=0.95, threshold=0.3).to(device)
        model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        model.eval()
        
        run_preds = []
        true_poses = []
        
        with torch.no_grad():
            for data, target in test_loader:
                data = data.to(device)
                pred, _ = model(data)
                
                run_preds.append(pred[0].cpu().numpy())
                if idx == 0: # Salviamo la Ground Truth solo al primo giro
                    true_poses.append(target[0].numpy())
                    
        all_predictions.append(run_preds)
        if idx == 0:
            true_trajectory = np.array(true_poses)

    # Converti in Numpy Array: shape sarà (5_runs, num_windows, 6_dof)
    all_predictions = np.array(all_predictions)

    # 4. Calcola la Media e la Deviazione Standard lungo l'asse dei RUN (axis=0)
    mean_preds = np.mean(all_predictions, axis=0)
    std_preds = np.std(all_predictions, axis=0)

    # 5. Creazione del Plot Definitivo per il Report
    print("\nPlotting Data...")
    fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    time_steps = np.arange(mean_preds.shape[0])

    # X-Axis Plot (Indice 3)
    axs[0].plot(time_steps, true_trajectory[:, 3], 'k--', label='True X', linewidth=2)
    axs[0].plot(time_steps, mean_preds[:, 3], 'r-', label='Mean Predicted X', linewidth=1.5)
    axs[0].fill_between(time_steps, mean_preds[:, 3] - std_preds[:, 3], mean_preds[:, 3] + std_preds[:, 3], color='red', alpha=0.3, label='±1 Std Dev')
    axs[0].set_ylabel('X Position (m)', fontsize=12)
    axs[0].grid(True, linestyle='--', alpha=0.7)
    axs[0].legend(loc='upper right')
    axs[0].set_title('SNN 6DoF Pose Estimation Uncertainty (5 Independent Runs)', fontsize=14, fontweight='bold')

    # Y-Axis Plot (Indice 4)
    axs[1].plot(time_steps, true_trajectory[:, 4], 'k--', label='True Y', linewidth=2)
    axs[1].plot(time_steps, mean_preds[:, 4], 'g-', label='Mean Predicted Y', linewidth=1.5)
    axs[1].fill_between(time_steps, mean_preds[:, 4] - std_preds[:, 4], mean_preds[:, 4] + std_preds[:, 4], color='green', alpha=0.3, label='±1 Std Dev')
    axs[1].set_ylabel('Y Position (m)', fontsize=12)
    axs[1].grid(True, linestyle='--', alpha=0.7)
    axs[1].legend(loc='upper right')

    # Z-Axis Plot (Indice 5)
    axs[2].plot(time_steps, true_trajectory[:, 5], 'k--', label='True Z', linewidth=2)
    axs[2].plot(time_steps, mean_preds[:, 5], 'b-', label='Mean Predicted Z', linewidth=1.5)
    axs[2].fill_between(time_steps, mean_preds[:, 5] - std_preds[:, 5], mean_preds[:, 5] + std_preds[:, 5], color='blue', alpha=0.3, label='±1 Std Dev')
    axs[2].set_ylabel('Z Position (m)', fontsize=12)
    axs[2].set_xlabel('Time (Overlapping Windows)', fontsize=12)
    axs[2].grid(True, linestyle='--', alpha=0.7)
    axs[2].legend(loc='upper right')

    plt.tight_layout()
    plt.savefig('snn_uncertainty_bands.png', dpi=300)
    print("Saved beautiful plot as 'snn_uncertainty_bands.png'")
    plt.show()