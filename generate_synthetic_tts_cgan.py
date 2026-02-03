import os
import torch
import numpy as np
import pickle
from TransCGAN_model import Generator

def generate_and_restore(checkpoint_path, pkl_path, appliance_list, seq_len=512, latent_dim=100, num_samples=20000):
    with open(pkl_path, 'rb') as f:
        min_max_dict = pickle.load(f)
    
    checkpoint = torch.load(checkpoint_path)
    gen_net = Generator(seq_len=seq_len, channels=9, num_classes=len(appliance_list), 
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
    
    os.makedirs('results_restored', exist_ok=True)
    
    for idx, app in enumerate(appliance_list):
        print(f"Generating synthetic data for {app}...")
        batch_size = 100
        all_samples = []
        for _ in range(num_samples // batch_size):
            with torch.no_grad():
                z = torch.randn(batch_size, latent_dim).cuda()
                labels = torch.full((batch_size,), idx, dtype=torch.long).cuda()
                fake_imgs = gen_net(z, labels).cpu().numpy()
                all_samples.append(fake_imgs)
        
        samples = np.concatenate(all_samples, axis=0).squeeze(2)
        d_min, d_max = min_max_dict[app]
        d_min = d_min.reshape(1, 9, 1)
        d_max = d_max.reshape(1, 9, 1)
        restored_samples = samples * (d_max - d_min + 1e-7) + d_min
        restored_samples = restored_samples.transpose(0, 2, 1)
        
        save_path = f'results_restored/{app}_synthetic_data.npy'
        np.save(save_path, restored_samples)
        print(f"Saved to {save_path}")

if __name__ == "__main__":
    APPS = ["dishwasher", "fridge", "kettle", "microwave", "washingmachine"]
    print("Generation script ready.")
