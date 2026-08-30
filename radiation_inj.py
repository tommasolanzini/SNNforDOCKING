import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader
from SNN import SEENIC_SNN_Dataset, SpacecraftSNN
from ANN import SEENIC_ANN_Dataset, SpacecraftANN

# 1. The Bit-Flip Injector 
def inject_radiation(state_dict, BER):
    corrupted_dict = {}
    for name, tensor in state_dict.items():
        if 'weight' in name:
            np_arr = tensor.cpu().numpy()
            uint_view = np_arr.view(np.uint32)
            
            flat_size = uint_view.size
            flip_mask = np.zeros(flat_size, dtype=np.uint32)
            
            total_bits = flat_size * 32
            num_flips = int(total_bits * BER)
            
            for _ in range(num_flips):
                idx = np.random.randint(0, flat_size)
                bit_pos = np.random.randint(0, 32)
                flip_mask[idx] ^= (np.uint32(1) << np.uint32(bit_pos))
                
            corrupted_uint = np.bitwise_xor(uint_view.flatten(), flip_mask)
            corrupted_float = corrupted_uint.view(np.float32).reshape(uint_view.shape)
            
            corrupted_float = np.nan_to_num(corrupted_float, nan=0.0, posinf=10.0, neginf=-10.0)
            corrupted_dict[name] = torch.tensor(corrupted_float)
        else:
            corrupted_dict[name] = tensor.clone()
            
    return corrupted_dict

# 2. Evaluation Engine
def evaluate_corrupted_model(model, dataloader, device, model_type="SNN"):
    model.eval()
    errors = []
    
    with torch.no_grad():
        for data, true_pose in dataloader:
            data = data.to(device)
            if model_type == "SNN":
                pred, _ = model(data)
            else:
                pred = model(data)
            
            # CRITICAL FIX: Cast to float64 to prevent intermediate overflow
            p = pred[0].cpu().numpy().astype(np.float64)
            t = true_pose[0].numpy().astype(np.float64)
            error = (p - t)**2
            errors.append(error)
            
    rmse = np.sqrt(np.mean(errors))
    return rmse

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Radiation Simulator active on: {device}")

    TEST_FOLDER = "hubble-approach-fast-lightboxdiffuser" 
    events_file = Path(TEST_FOLDER) / "events.csv"
    poses_file = Path(TEST_FOLDER) / "cam-poses.csv"

    snn_dataset = SEENIC_SNN_Dataset(str(events_file), str(poses_file))
    ann_dataset = SEENIC_ANN_Dataset(str(events_file), str(poses_file))

    snn_loader = DataLoader(snn_dataset, batch_size=1, shuffle=False)
    ann_loader = DataLoader(ann_dataset, batch_size=1, shuffle=False)

    snn_clean_weights = torch.load("SNN_weights_dropout.pth", map_location=device, weights_only=True)
    ann_clean_weights = torch.load("ANN_weights_1.pth", map_location=device, weights_only=True)

    radiation_levels = [0.0, 1e-6, 5e-6, 1e-5, 5e-5, 1e-4]
    num_trials = 20  
    
    snn_death_rates = []
    ann_death_rates = []
    
    # New arrays to track precision loss
    snn_mean_rmse = []
    ann_mean_rmse = []

    print(f"\nStarting Monte Carlo SEU Injection ({num_trials} trials per level)...")
    for ber in radiation_levels:
        print(f"\n[!] Simulating Bit Error Rate: {ber}")
        
        snn_deaths = 0
        ann_deaths = 0
        
        snn_valid_rmse_scores = []
        ann_valid_rmse_scores = []
        
        for trial in range(num_trials):
            # Test SNN
            snn = SpacecraftSNN(beta=0.95, threshold=0.3).to(device)
            snn.load_state_dict(inject_radiation(snn_clean_weights, ber))
            snn_rmse = evaluate_corrupted_model(snn, snn_loader, device, "SNN")
            
            if np.isnan(snn_rmse) or np.isinf(snn_rmse) or snn_rmse > 1.0:
                snn_deaths += 1
            else:
                snn_valid_rmse_scores.append(snn_rmse)

            # Test ANN
            ann = SpacecraftANN().to(device)
            ann.load_state_dict(inject_radiation(ann_clean_weights, ber))
            ann_rmse = evaluate_corrupted_model(ann, ann_loader, device, "ANN")
            
            if np.isnan(ann_rmse) or np.isinf(ann_rmse) or ann_rmse > 1.0:
                ann_deaths += 1
            else:
                ann_valid_rmse_scores.append(ann_rmse)
                
        # 1. Calculate Failure Rates
        snn_fail_pct = (snn_deaths / num_trials) * 100
        ann_fail_pct = (ann_deaths / num_trials) * 100
        snn_death_rates.append(snn_fail_pct)
        ann_death_rates.append(ann_fail_pct)
        
        # 2. Calculate Precision Degradation (Average RMSE of survivors)
        # If all trials died, cap the plot at 1.0 for visualization
        snn_avg_rmse = np.mean(snn_valid_rmse_scores) if len(snn_valid_rmse_scores) > 0 else 1.0
        ann_avg_rmse = np.mean(ann_valid_rmse_scores) if len(ann_valid_rmse_scores) > 0 else 1.0
        
        snn_mean_rmse.append(snn_avg_rmse)
        ann_mean_rmse.append(ann_avg_rmse)
        
        print(f"  -> SNN: {snn_fail_pct}% Dead | Surviving Precision (RMSE): {snn_avg_rmse:.4f}")
        print(f"  -> ANN: {ann_fail_pct}% Dead | Surviving Precision (RMSE): {ann_avg_rmse:.4f}")

    
    # 3. PLOTTING SIDE-BY-SIDE METRICS
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left Plot: Catastrophic Failure Rate
    ax1.plot(radiation_levels, snn_death_rates, label='SNN (Neuromorphic)', marker='o', color='blue', linewidth=2)
    ax1.plot(radiation_levels, ann_death_rates, label='ANN (Standard)', marker='X', color='red', linewidth=2, linestyle='--')
    ax1.set_title("Catastrophic Failure Rate", fontsize=14, fontweight='bold')
    ax1.set_xlabel("Bit Error Rate (BER)", fontsize=12)
    ax1.set_ylabel("Failure Rate (%)", fontsize=12)
    ax1.set_xscale('symlog', linthresh=1e-6)
    ax1.set_ylim(-5, 105)
    ax1.grid(True, which="both", ls="--", alpha=0.5)
    ax1.legend(fontsize=12)
    
    # Right Plot: Precision Degradation (RMSE)
    ax2.plot(radiation_levels, snn_mean_rmse, label='SNN (Neuromorphic)', marker='o', color='blue', linewidth=2)
    ax2.plot(radiation_levels, ann_mean_rmse, label='ANN (Standard)', marker='X', color='red', linewidth=2, linestyle='--')
    ax2.set_title("Precision Degradation (Surviving Networks)", fontsize=14, fontweight='bold')
    ax2.set_xlabel("Bit Error Rate (BER)", fontsize=12)
    ax2.set_ylabel("Mean RMSE (Lower is Better)", fontsize=12)
    ax2.set_xscale('symlog', linthresh=1e-6)
    ax2.set_ylim(0, 1.05)
    ax2.grid(True, which="both", ls="--", alpha=0.5)
    ax2.legend(fontsize=12)

    plt.suptitle("Network Survivability and Precision under Cosmic Radiation (SEU Bit-Flips)", fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    plt.savefig('radiation_full_analysis.png', dpi=300)
    print("\nPlot saved as 'radiation_full_analysis.png'")
    plt.show()