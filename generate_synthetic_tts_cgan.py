import os
import torch
import numpy as np
import pandas as pd
import pickle
import argparse
from TransCGAN_model import Generator

def generate_and_restore(appliance, num_samples, seq_len=512, latent_dim=100):
    # 1. Path Setup
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_name = f'tts_cgan_nilm_{appliance}_512'
    log_parent = os.path.join(current_dir, 'logs')
    
    candidates = [d for d in os.listdir(log_parent) if d.startswith(base_name)]
    if not candidates:
        raise FileNotFoundError(f"No log directory found starting with {base_name} in {log_parent}")
    
    candidates.sort(key=lambda x: os.path.getmtime(os.path.join(log_parent, x)), reverse=True)
    log_path = os.path.join(log_parent, candidates[0])
    print(f"Using log directory: {log_path}")
    
    pkl_path = os.path.join(log_path, 'min_max_values.pkl')
    if not os.path.exists(pkl_path):
        pkl_path = os.path.join(log_path, 'Log', 'min_max_values.pkl')
        
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"Could not find min_max_values.pkl in {log_path}")
        
    print(f"Loading Min/Max from: {pkl_path}")
    
    possible_dirs = [
        os.path.join(log_path, 'checkpoints'),
        os.path.join(log_path, 'Model')
    ]
    
    ckpt_dir = None
    for d in possible_dirs:
        if os.path.exists(d):
            ckpt_dir = d
            break
            
    if not ckpt_dir:
        raise FileNotFoundError(f"Could not find checkpoints or Model directory in {log_path}")
        
    print(f"Fetching models from: {ckpt_dir}")
    ckpts = [f for f in os.listdir(ckpt_dir) if f.endswith('.pth')]
    if not ckpts:
        print(f"No checkpoints found for {appliance}")
        return
    last_ckpt = sorted(ckpts, key=lambda x: int(x.split('_')[1].split('.')[0]))[-1]
    checkpoint_path = os.path.join(ckpt_dir, last_ckpt)
    
    print(f"Using checkpoint: {checkpoint_path}")

    # 2. Load Meta Data
    with open(pkl_path, 'rb') as f:
        min_max_dict = pickle.load(f)
    
    # 3. Load Model (Conditional GAN: outputs 1 Power, conditioned on 8 Time)
    checkpoint = torch.load(checkpoint_path)
    gen_net = Generator(seq_len=seq_len, channels=1, num_classes=1, 
                        latent_dim=latent_dim, data_embed_dim=128, label_embed_dim=128, 
                        depth=4, num_heads=8, time_dim=8).cuda()
    
    state_dict = checkpoint['gen_state_dict']
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace('module.', '')
        new_state_dict[name] = v
    gen_net.load_state_dict(new_state_dict)
    gen_net.eval()
    
    # 4. Load Real Time Features for Conditioning
    csv_path = os.path.join(current_dir, 'data', f"{appliance}_training_.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Cannot find {csv_path} for time feature sampling")
        
    print(f"Loading time features from: {csv_path}")
    df = pd.read_csv(csv_path)
    time_data = df.iloc[:, 2:].values  # Columns 2-9 (8 time features)
    num_orig_windows = len(time_data) // seq_len
    time_windows = time_data[:num_orig_windows * seq_len].reshape(-1, seq_len, 8)
    
    # Randomly sample time windows
    idx = np.random.choice(num_orig_windows, num_samples, replace=True)
    sampled_time = time_windows[idx]  # (num_samples, 512, 8)
    sampled_time = sampled_time.transpose(0, 2, 1)  # (num_samples, 8, 512)
    sampled_time = sampled_time[:, :, np.newaxis, :]  # (num_samples, 8, 1, 512)
    sampled_time_tensor = torch.from_numpy(sampled_time).float().cuda()
    
    # 5. Generate Power conditioned on Time
    output_dir = os.path.join(current_dir, 'synthetic_out')
    os.makedirs(output_dir, exist_ok=True)
    
    batch_size = 100
    all_power_samples = []
    all_time_samples = []
    
    print(f"Generating {num_samples} windows for {appliance}...")
    
    for i in range(int(np.ceil(num_samples / batch_size))):
        with torch.no_grad():
            cur_batch = min(batch_size, num_samples - i * batch_size)
            z = torch.randn(cur_batch, latent_dim).cuda()
            labels = torch.zeros(cur_batch, dtype=torch.long).cuda()
            time_batch = sampled_time_tensor[i*batch_size:i*batch_size+cur_batch]
            
            # 🚀 Conditional Generation: Power = G(z, time)
            fake_power = gen_net(z, labels, time_batch).cpu().numpy()
            all_power_samples.append(fake_power)
            all_time_samples.append(sampled_time[i*batch_size:i*batch_size+cur_batch])
    
    power_samples = np.concatenate(all_power_samples, axis=0).squeeze(2)  # (N, 1, 512)
    time_samples = np.concatenate(all_time_samples, axis=0).squeeze(2)    # (N, 8, 512)
    
    # 6. Restore Power from [0, 1] to Z-score
    d_min, d_max = min_max_dict[appliance]
    d_min = d_min.reshape(1, 1, 1)
    d_max = d_max.reshape(1, 1, 1)
    
    power_restored = power_samples * (d_max - d_min + 1e-7) + d_min
    
    # 7. Concatenate Power + Time
    final_data = np.concatenate([power_restored, time_samples], axis=1)  # (N, 9, 512)
    final_data = final_data.transpose(0, 2, 1)  # (N, 512, 9)
    
    save_path = os.path.join(output_dir, f'{appliance}_synthetic_data.npy')
    np.save(save_path, final_data)
    print(f"Successfully saved to {save_path}")
    print(f"Output shape: {final_data.shape}")
    print(f"Power range: [{power_restored.min():.3f}, {power_restored.max():.3f}] (Z-score)")
    print(f"Time range: [{time_samples.min():.3f}, {time_samples.max():.3f}]")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--appliance', type=str, required=True)
    parser.add_argument('--num_samples', type=int, required=True)
    args = parser.parse_args()
    
    generate_and_restore(args.appliance, args.num_samples)
