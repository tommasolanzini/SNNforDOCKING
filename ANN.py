import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import matplotlib.pyplot as plt


# DATASET 
class SEENIC_ANN_Dataset(Dataset):
    def __init__(self, events_csv, poses_csv, window_us=50000, stride_us=25000, H=480, W=640):
        self.window_us = window_us
        self.stride_us = stride_us # advance of 25ms each time
        self.H = H
        self.W = W
        self.events = pd.read_csv(events_csv, names=['t', 'x', 'y', 'p'], header=0)
        self.poses = pd.read_csv(poses_csv, names=['t', 'rx', 'ry', 'rz', 'x', 'y', 'z'], header=0)
        self.start_time = self.events['t'].iloc[0]
        self.end_time = self.events['t'].iloc[-1]
        usable_time = max(0, self.end_time - self.start_time - self.window_us)
        self.num_windows = int((usable_time // self.stride_us) + 1)

    def __len__(self):
        return self.num_windows

    def _interpolate_pose(self, query_time):
        poses_t = self.poses['t'].values
        idx = np.searchsorted(poses_t, query_time)
        if idx == 0: return self.poses.iloc[0, 1:].values.astype(np.float32)
        if idx >= len(poses_t): return self.poses.iloc[-1, 1:].values.astype(np.float32)
        
        t0, t1 = poses_t[idx-1], poses_t[idx]
        p0, p1 = self.poses.iloc[idx-1, 1:].values, self.poses.iloc[idx, 1:].values
        ratio = (query_time - t0) / (t1 - t0)
        return (p0 + ratio * (p1 - p0)).astype(np.float32)

    def __getitem__(self, idx):
        t_start = self.start_time + (idx * self.stride_us)
        t_end = t_start + self.window_us
        
        mask = (self.events['t'] >= t_start) & (self.events['t'] < t_end)
        window_events = self.events[mask]

        frame = np.zeros((2, self.H, self.W), dtype=np.float32)
        
        if not window_events.empty:
            x = window_events['x'].values.astype(int)
            y = window_events['y'].values.astype(int)
            p = window_events['p'].values.astype(int)

            np.add.at(frame, (p, y, x), 1)

        t_center = t_start + (self.window_us / 2.0)
        target_pose = self._interpolate_pose(t_center)
        
        return torch.tensor(frame), torch.tensor(target_pose)



# CUSTOM LOSS 

class PoseWeightedMSELossANN(nn.Module):
    def __init__(self, translation_weight=10.0, rotation_weight=1.0):
        super().__init__()
        self.trans_w = translation_weight
        self.rot_w = rotation_weight
        self.mse = nn.MSELoss() 

    def forward(self, predictions, targets):
        pred_rot = predictions[:, :3]
        target_rot = targets[:, :3]       
        pred_trans = predictions[:, 3:]
        target_trans = targets[:, 3:]

        loss_rot = self.mse(pred_rot, target_rot)
        loss_trans = self.mse(pred_trans, target_trans)

        total_weighted_loss = (self.rot_w * loss_rot) + (self.trans_w * loss_trans)
        return total_weighted_loss
    

class SpacecraftANN(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=16, kernel_size=5, stride=2)
        self.relu1 = nn.ReLU()
        
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=2)
        self.relu2 = nn.ReLU()
        
        self.pool = nn.AdaptiveAvgPool2d((5, 5))
        self.flatten = nn.Flatten()
        
        self.fc1 = nn.Linear(32 * 5 * 5, 128)
        self.dropout = nn.Dropout(0.2) 
        self.relu3 = nn.ReLU()
        
        self.fc2 = nn.Linear(128, 6) 

    def forward(self, x):
        # No time loop required for standard ANN
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        
        x = self.pool(x)
        x = self.flatten(x)
        
        x = self.dropout(self.relu3(self.fc1(x)))
        x = self.fc2(x)
        
        return x

# TRAINING LOOP 

if __name__ == "__main__":
    import matplotlib.pyplot as plt # Ensure this is imported

    root_dir = Path("SNN_docking")
    dataset_list = []

    for folder_path in root_dir.iterdir():
        if folder_path.is_dir():
            events_file = folder_path / "events.csv"
            poses_file = folder_path / "cam-poses.csv"
            
            if events_file.exists() and poses_file.exists():
                print(f"Loading sequence: {folder_path.name}")
                
                single_scene_dataset = SEENIC_ANN_Dataset(
                    events_csv=str(events_file), 
                    poses_csv=str(poses_file)
                )
                dataset_list.append(single_scene_dataset)

    full_dataset = ConcatDataset(dataset_list)
    print(f"Total time windows across all folders: {len(full_dataset)}")

    dataloader = DataLoader(full_dataset, batch_size=16, shuffle=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = SpacecraftANN().to(device)
    
    # Custom Loss applicata senza smoothness
    criterion = PoseWeightedMSELossANN(translation_weight=10.0, rotation_weight=1.0)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    epochs = 100 
    print("\n--- STARTING ANN TRAINING ---")

    mean_loss = []
    best_loss = float('inf')
    epochs_without_improvement = 0
    early_stop_patience = 10 

    for epoch in range(epochs):
        model.train() 
        total_loss = 0.0

        for batch_images, batch_target_poses in dataloader:
            batch_images = batch_images.to(device)
            batch_target_poses = batch_target_poses.to(device)

            optimizer.zero_grad()
            predictions = model(batch_images) 
            loss = criterion(predictions, batch_target_poses)
            
            loss.backward()
            optimizer.step() 
            
            total_loss += loss.item()

        current_epoch_loss = total_loss / len(dataloader)
        mean_loss.append(current_epoch_loss)

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{epochs} | Mean Loss: {current_epoch_loss:.4f} | LR: {current_lr:.6f}")

        scheduler.step(current_epoch_loss)

        if current_epoch_loss < best_loss:
            best_loss = current_epoch_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), 'ANN_weights_2.pth') 
        else:
            epochs_without_improvement += 1
            
        if epochs_without_improvement >= early_stop_patience:
            print(f"\n[!] Early stopping triggered! Loss hasn't improved for {early_stop_patience} epochs.")
            print(f"Training stopped at Epoch {epoch+1}. The best weights are safely saved.")
            break

    
    # PLOT TRAINING LOSS
    
    print("\n--- PLOTTING TRAINING LOSS ---")
    plt.figure(figsize=(10, 6))
    
    # Plot the recorded losses
    plt.plot(range(1, len(mean_loss) + 1), mean_loss, marker='o', linestyle='-', color='r', label='Training Mean Loss (ANN)')

    plt.title('ANN Training Loss Over Epochs', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Pose Weighted MSE Loss', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.tight_layout()

    # Save the plot for the LaTeX report
    plt.savefig('ann_training_loss_curve.png', dpi=300)
    print("Loss plot saved as 'ann_training_loss_curve.png'")
    
    plt.show()