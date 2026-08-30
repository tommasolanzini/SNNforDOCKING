import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from SNN import SEENIC_SNN_Dataset, SpacecraftSNN
from ANN import SEENIC_ANN_Dataset, SpacecraftANN

def evaluate_model(model_type, model_path, test_folder_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*50}")
    print(f" EVALUATING {model_type.upper()} ARCHITECTURE")
    print(f"{'='*50}")
    print(f"Using device: {device}")

    # Initialize Model and Dataset based on type
    events_file = Path(test_folder_path) / "events.csv"
    poses_file = Path(test_folder_path) / "cam-poses.csv"

    if model_type.upper() == "SNN":
        model = SpacecraftSNN(beta=0.95, threshold=1).to(device)
        test_dataset = SEENIC_SNN_Dataset(str(events_file), str(poses_file))
    elif model_type.upper() == "ANN":
        model = SpacecraftANN().to(device)
        test_dataset = SEENIC_ANN_Dataset(str(events_file), str(poses_file))
    else:
        raise ValueError("model_type must be either 'SNN' or 'ANN'")

    # Load weights and lock the network
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    true_poses = []
    pred_poses = []

    print(f"Running inference on unseen data ({len(test_dataset)} sequential windows)...")
    
    with torch.no_grad(): 
        for data, true_pose in test_loader:
            data = data.to(device)
            
            # The SNN returns a tuple (prediction, timeline), the ANN returns just the prediction
            if model_type.upper() == "SNN":
                pred, _ = model(data)
            else:
                pred = model(data)
                
            pred_poses.append(pred[0].cpu().numpy())
            true_poses.append(true_pose[0].numpy())

    true_poses = np.array(true_poses)
    pred_poses = np.array(pred_poses)

    # Calculate Statistical Metrics
    errors = pred_poses - true_poses
    
    # Root Mean Square Error (RMSE) - heavily penalizes large deviations
    rmse_per_axis = np.sqrt(np.mean(errors**2, axis=0))
    mean_trans_rmse = np.sqrt(np.mean(errors[:, 3:]**2))
    mean_rot_rmse = np.sqrt(np.mean(errors[:, :3]**2))
    
    # Mean Absolute Error (MAE) - gives the average literal distance off-target
    mae_per_axis = np.mean(np.abs(errors), axis=0)
    
    print("\n--- STATISTICAL RESULTS ---")
    print(f"Overall Translation RMSE: {mean_trans_rmse:.4f} meters")
    print(f"Overall Rotation RMSE:    {mean_rot_rmse:.4f} radians")
    
    print("\nDetailed Mean Absolute Error (MAE):")
    print(f"  X: {mae_per_axis[3]:.4f} m   |  Roll (Rx):  {mae_per_axis[0]:.4f} rad")
    print(f"  Y: {mae_per_axis[4]:.4f} m   |  Pitch (Ry): {mae_per_axis[1]:.4f} rad")
    print(f"  Z: {mae_per_axis[5]:.4f} m   |  Yaw (Rz):   {mae_per_axis[2]:.4f} rad")

    # 3. Plotting the Results
    time_steps = range(len(true_poses))
    plt.figure(figsize=(10, 8))
    
    # Plot X
    plt.subplot(3, 1, 1)
    plt.plot(time_steps, true_poses[:, 3], label='True X', color='black', linestyle='--')
    plt.plot(time_steps, pred_poses[:, 3], label=f'Predicted X ({model_type})', color='red')
    plt.ylabel('X Position (m)')
    plt.legend()
    plt.grid(True)

    # Plot Y
    plt.subplot(3, 1, 2)
    plt.plot(time_steps, true_poses[:, 4], label='True Y', color='black', linestyle='--')
    plt.plot(time_steps, pred_poses[:, 4], label=f'Predicted Y ({model_type})', color='green')
    plt.ylabel('Y Position (m)')
    plt.legend()
    plt.grid(True)

    # Plot Z
    plt.subplot(3, 1, 3)
    plt.plot(time_steps, true_poses[:, 5], label='True Z', color='black', linestyle='--')
    plt.plot(time_steps, pred_poses[:, 5], label=f'Predicted Z ({model_type})', color='blue')
    plt.ylabel('Z Position (m)')
    plt.xlabel('Time (Overlapping Windows)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    TEST_FOLDER = "hubble-approach-fast-lightboxdiffuser" 
    SNN_WEIGHTS = "NNs_weights/SNN_weights_thr_1.pth"
    if Path(SNN_WEIGHTS).exists():
        evaluate_model("SNN", SNN_WEIGHTS, TEST_FOLDER)
    else:
        print(f"\n[!] SNN weights file not found: {SNN_WEIGHTS}")
        
    # Test the ANN
    ANN_WEIGHTS = "NNs_weights/ANN_weights_.pth"
    if Path(ANN_WEIGHTS).exists():
        evaluate_model("ANN", ANN_WEIGHTS, TEST_FOLDER)
    else:
        print(f"\n[!] ANN weights file not found: {ANN_WEIGHTS}")