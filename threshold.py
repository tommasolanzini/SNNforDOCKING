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
    print(f"Running Threshold Sensitivity Analysis on: {device}")

    # Load Test Dataset
    TEST_FOLDER = "hubble-approach-fast-lightboxdiffuser" 
    events_file = Path(TEST_FOLDER) / "events.csv"
    poses_file = Path(TEST_FOLDER) / "cam-poses.csv"

    test_dataset = SEENIC_SNN_Dataset(str(events_file), str(poses_file))
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Define configurations for the 3 distinct trained models
    # UPDATE THESE PATHS to match your newly trained weight files
    models_info = [
        {"path": "NNs_weights/SNN_weights_run_1.pth", "thr": 0.3, "title": "a) Threshold $U_{th} = 0.3$\n(Optimal Propagation)", "color": "green"},
        {"path": "NNs_weights/SNN_weights_thr_06.pth", "thr": 0.6, "title": "b) Threshold $U_{th} = 0.6$\n(Partial Attenuation)", "color": "orange"},
        {"path": "NNs_weights/SNN_weights_thr_1.pth", "thr": 1.0, "title": "c) Threshold $U_{th} = 1.0$\n(Voltage Famine)", "color": "red"}
    ]
    
    all_predictions = []
    true_trajectory = []

    # Inference loop for each model
    for idx, info in enumerate(models_info):
        print(f"Loading Model trained with Threshold = {info['thr']}...")
        
        # Initialize model with its specific threshold
        model = SpacecraftSNN(beta=0.95, threshold=info['thr']).to(device)
        
        # Load the specific weights trained for this threshold
        model.load_state_dict(torch.load(info['path'], map_location=device, weights_only=True))
        model.eval()
        
        run_preds = []
        true_poses = []
        
        with torch.no_grad():
            for data, target in test_loader:
                data = data.to(device)
                pred, _ = model(data)
                run_preds.append(pred[0].cpu().numpy())
                
                # Save Ground Truth only during the first iteration
                if idx == 0:
                    true_poses.append(target[0].numpy())
                    
        all_predictions.append(np.array(run_preds))
        if idx == 0:
            true_trajectory = np.array(true_poses)

    # Plotting the 3-panel figure
    print("\nPlotting Sensitivity Analysis...")
    fig, axs = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    time_steps = np.arange(true_trajectory.shape[0])

    for i, info in enumerate(models_info):
        preds = all_predictions[i]
        
        # Plot Z axis (Index 5)
        axs[i].plot(time_steps, true_trajectory[:, 5], 'k--', label='True Z Position', linewidth=2)
        axs[i].plot(time_steps, preds[:, 5], color=info['color'], label=f'Predicted Z ($U_{{th}}={info["thr"]}$)', linewidth=1.5)
        
        # Panel formatting
        axs[i].set_title(info['title'], fontsize=14, fontweight='bold', pad=15)
        axs[i].set_xlabel('Time (Overlapping Windows)', fontsize=12)
        axs[i].grid(True, linestyle='--', alpha=0.7)
        axs[i].legend(loc='lower left' if i == 0 else 'upper right')
        
    # Set Y label only on the first shared axis
    axs[0].set_ylabel('Z Position (m)', fontsize=12)

    plt.tight_layout()
    plt.savefig('plots/threshold_sensitivity_Z.pdf', dpi=300, bbox_inches='tight')
    plt.show()