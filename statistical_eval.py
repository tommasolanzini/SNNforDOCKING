import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from torch.utils.data import DataLoader
from SNN import SEENIC_SNN_Dataset, SpacecraftSNN

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Calculating RMSE & MSE Uncertainties on: {device}")
    
    TEST_FOLDER = "testing/hubble-approach-fast-lightboxdiffuser"
    events_file = Path(TEST_FOLDER) / "events.csv"
    poses_file = Path(TEST_FOLDER) / "cam-poses.csv"

    test_dataset = SEENIC_SNN_Dataset(str(events_file), str(poses_file))
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    model_paths = [
        "NNs_weights/SNN_weights_run_1.pth", "NNs_weights/SNN_weights_run_2.pth", "NNs_weights/SNN_weights_run_3.pth",
        "NNs_weights/SNN_weights_run_4.pth", "NNs_weights/SNN_weights_run_5.pth", "NNs_weights/SNN_weights_run_6.pth",
        "NNs_weights/SNN_weights_run_7.pth", "NNs_weights/SNN_weights_run_8.pth", "NNs_weights/SNN_weights_run_9.pth",
        "NNs_weights/SNN_weights_run_10.pth"
    ]

    all_rmse = []
    all_mse = []
    all_predictions = []
    true_trajectory = []

    for idx, path in enumerate(model_paths):
        print(f"Evaluating Model {idx+1}/10...")
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
                if idx == 0:
                    true_poses.append(target[0].numpy())
        
        run_preds = np.array(run_preds)
        if idx == 0:
            true_trajectory = np.array(true_poses)
        
        all_predictions.append(run_preds)
        
        mse_per_dof = np.mean((run_preds - true_trajectory)**2, axis=0)
        rmse_per_dof = np.sqrt(mse_per_dof)
        
        all_mse.append(mse_per_dof)
        all_rmse.append(rmse_per_dof)

    all_predictions = np.array(all_predictions)
    mean_preds = np.mean(all_predictions, axis=0)
    std_preds = np.std(all_predictions, axis=0)

    all_mse = np.array(all_mse)
    all_rmse = np.array(all_rmse)
    
    mean_mse = np.mean(all_mse, axis=0)
    std_mse = np.std(all_mse, axis=0)
    mean_rmse = np.mean(all_rmse, axis=0)
    std_rmse = np.std(all_rmse, axis=0)
    
    dof_labels = ['X (m)', 'Y (m)', 'Z (m)', 'Rx (rad)', 'Ry (rad)', 'Rz (rad)']
    
    print("\n\n" + "="*60)
    print(f"{'DoF':<10} | {'RMSE Mean ± Std':<20} | {'MSE Mean ± Std':<20}")
    print("-" * 55)
    for i in range(6):
        print(f"{dof_labels[i]:<10} | {mean_rmse[i]:.4f} ± {std_rmse[i]:.4f} | {mean_mse[i]:.4f} ± {std_mse[i]:.4f}")
    print("="*60)
    
    print("\nGenerating Comprehensive Evaluation Plot...")
    
    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(6, 2, width_ratios=[1.5, 1])
    
    ax_x = fig.add_subplot(gs[0:2, 0])
    ax_y = fig.add_subplot(gs[2:4, 0], sharex=ax_x)
    ax_z = fig.add_subplot(gs[4:6, 0], sharex=ax_x)
    
    time_steps = np.arange(mean_preds.shape[0])

    ax_x.plot(time_steps, true_trajectory[:, 3], 'k--', label='True X', linewidth=2)
    ax_x.plot(time_steps, mean_preds[:, 3], 'r-', label='Mean Predicted X', linewidth=1.5)
    ax_x.fill_between(time_steps, mean_preds[:, 3] - std_preds[:, 3], mean_preds[:, 3] + std_preds[:, 3], color='red', alpha=0.3, label='±1 Std Dev')
    ax_x.set_ylabel('X Position (m)', fontsize=12)
    ax_x.grid(True, linestyle='--', alpha=0.7)
    ax_x.legend(loc='upper right')
    ax_x.set_title('SNN 6DoF Pose Estimation Uncertainty (10 Independent Runs)', fontsize=14, fontweight='bold')

    ax_y.plot(time_steps, true_trajectory[:, 4], 'k--', label='True Y', linewidth=2)
    ax_y.plot(time_steps, mean_preds[:, 4], 'g-', label='Mean Predicted Y', linewidth=1.5)
    ax_y.fill_between(time_steps, mean_preds[:, 4] - std_preds[:, 4], mean_preds[:, 4] + std_preds[:, 4], color='green', alpha=0.3, label='±1 Std Dev')
    ax_y.set_ylabel('Y Position (m)', fontsize=12)
    ax_y.grid(True, linestyle='--', alpha=0.7)
    ax_y.legend(loc='upper right')

    ax_z.plot(time_steps, true_trajectory[:, 5], 'k--', label='True Z', linewidth=2)
    ax_z.plot(time_steps, mean_preds[:, 5], 'b-', label='Mean Predicted Z', linewidth=1.5)
    ax_z.fill_between(time_steps, mean_preds[:, 5] - std_preds[:, 5], mean_preds[:, 5] + std_preds[:, 5], color='blue', alpha=0.3, label='±1 Std Dev')
    ax_z.set_ylabel('Z Position (m)', fontsize=12)
    ax_z.set_xlabel('Time (Overlapping Windows)', fontsize=12)
    ax_z.grid(True, linestyle='--', alpha=0.7)
    ax_z.legend(loc='upper right')

    ax_rmse = fig.add_subplot(gs[0:3, 1])
    ax_mse = fig.add_subplot(gs[3:6, 1])
    
    x_pos = np.arange(len(dof_labels))
    cap_size = 5
    colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#c2c2f0', '#ffb3e6']

    ax_rmse.bar(x_pos, mean_rmse, yerr=std_rmse, capsize=cap_size, color=colors, edgecolor='black')
    ax_rmse.set_xticks(x_pos)
    ax_rmse.set_xticklabels(dof_labels)
    ax_rmse.set_ylabel('RMSE', fontsize=12)
    ax_rmse.set_title('RMSE per Degree of Freedom', fontsize=14, fontweight='bold')
    ax_rmse.grid(axis='y', linestyle='--', alpha=0.7)

    ax_mse.bar(x_pos, mean_mse, yerr=std_mse, capsize=cap_size, color=colors, edgecolor='black')
    ax_mse.set_xticks(x_pos)
    ax_mse.set_xticklabels(dof_labels)
    ax_mse.set_ylabel('MSE', fontsize=12)
    ax_mse.set_title('MSE per Degree of Freedom', fontsize=14, fontweight='bold')
    ax_mse.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig('snn_comprehensive_evaluation.pdf', format='pdf', bbox_inches='tight', dpi=300)
    print("Saved beautiful comprehensive plot as 'snn_comprehensive_evaluation.pdf'")
    plt.show()