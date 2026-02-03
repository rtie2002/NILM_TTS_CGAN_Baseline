import os
import torch
import numpy as np
import pickle
import argparse
from TransCGAN_model import Generator

def generate_and_restore(appliance, num_samples, seq_len=512, latent_dim=100):
    # 1. Path Setup
    current_dir = os.path.dirname(os.path.abspath(__file__))
    exp_name = f'tts_cgan_nilm_{appliance}_512'
    log_path = os.path.join(current_dir, 'logs', exp_name)
    pkl_path = os.path.join(log_path, 'min_max_values.pkl')
    
    # Find the last checkpoint
    ckpt_dir = os.path.join(log_path, 'checkpoints')
    ckpts = [f for f in os.listdir(ckpt_dir) if f.endswith('.pth')]
    if not ckpts:
        print(f"No checkpoints found for {appliance}")
        return
    last_ckpt = sorted(ckpts, key=lambda x: int(x.split('_')[1].split('.')[0]))[-1]
    checkpoint_path = os.path.join(ckpt_dir, last_ckpt)
    
    print(f"Using checkpoint: {checkpoint_path}")
    print(f"Loading Min/Max from: {pkl_path}")

    # 2. Load Meta Data
    with open(pkl_path, 'rb') as f:
        min_max_dict = pickle.load(f)
    
    # 3. Load Model
    checkpoint = torch.load(checkpoint_path)
    gen_net = Generator(seq_len=seq_len, channels=9, num_classes=1, 
                        latent_dim=latent_dim, data_embed_dim=128, label_embed_dim=128, 
                        depth=4, num_heads=8).cuda()
    
    state_dict = checkpoint['gen_state_dict']
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace('module.', '')
        new_state_dict[name] = v
    gen_net.load_state_dict(new_state_dict)
    gen_net.eval()
    
    # 4. Generate
    output_dir = os.path.join(current_dir, 'synthetic_out')
    os.makedirs(output_dir, exist_ok=True)
    
    batch_size = 100
    all_samples = []
    print(f"Generating {num_samples} windows for {appliance}...")
    
    for i in range(int(np.ceil(num_samples / batch_size))):
        with torch.no_grad():
            cur_batch = min(batch_size, num_samples - i * batch_size)
            z = torch.randn(cur_batch, latent_dim).cuda()
            labels = torch.zeros(cur_batch, dtype=torch.long).cuda()
            fake_imgs = gen_net(z, labels).cpu().numpy()
            all_samples.append(fake_imgs)
    
    samples = np.concatenate(all_samples, axis=0).squeeze(2) # (N, 9, 512)
    
    # 5. Restore
    d_min, d_max = min_max_dict[appliance]
    d_min = d_min.reshape(1, 9, 1)
    d_max = d_max.reshape(1, 9, 1)
    restored_samples = samples * (d_max - d_min + 1e-7) + d_min
    restored_samples = restored_samples.transpose(0, 2, 1) # (N, 512, 9)
    
    save_path = os.path.join(output_dir, f'{appliance}_synthetic_data.npy')
    np.save(save_path, restored_samples)
    print(f"Successfully saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--appliance', type=str, required=True)
    parser.add_argument('--num_samples', type=int, required=True)
    args = parser.parse_args()
    
    generate_and_restore(args.appliance, args.num_samples)
