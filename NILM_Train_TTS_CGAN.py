import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import io
import PIL.Image
from torchvision.transforms import ToTensor
from TransCGAN_model import Generator, Discriminator
from adamw import AdamW
from cgan_functions import train, LinearLrDecay, copy_params
from utils import set_log_dir, create_logger
from copy import deepcopy
import pickle

# Configuration
class Args:
    def __init__(self):
        self.random_seed = 123
        self.dataset = 'NILM'
        # Scale to Data/datasets relative to this project
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_path = os.path.abspath(os.path.join(current_dir, '..', '..', 'Data', 'datasets'))
        self.appliances = ["dishwasher", "fridge", "kettle", "microwave", "washingmachine"]
        self.seq_len = 512  
        self.channels = 9   # 1 Power + 8 Time
        self.num_classes = 5
        self.latent_dim = 100
        self.batch_size = 256  # Optimized for 4090
        self.max_epoch = 100
        self.g_lr = 0.0001
        self.d_lr = 0.0001
        self.beta1 = 0.5
        self.beta2 = 0.9
        self.n_critic = 1
        self.val_freq = 5
        self.gpu = 0
        self.exp_name = 'tts_cgan_nilm_512'
        self.init_type = 'normal'
        self.num_workers = 0
        self.optimizer = 'adam'
        self.lr_decay = True
        self.max_iter = None
        self.show = True
        self.dist_url = "env://"
        self.world_size = 1
        self.rank = 0
        self.multiprocessing_distributed = False
        self.distributed = False
        self.gen_model = 'pro_gan'
        self.dis_model = 'pro_gan'
        self.path_helper = None
        self.wd = 1e-3
        self.phi = 1
        self.n_classes = 5
        self.ema = 0.999
        self.ema_kimg = 500
        self.ema_warmup = 0
        self.dis_batch_size = 256
        self.print_freq = 50
        self.grow_steps = [0, 0] 

class NILMDataset(Dataset):
    def __init__(self, data_root, appliances, seq_len=512):
        self.data = []
        self.labels = []
        self.min_max_values = {} 
        
        for idx, app in enumerate(appliances):
            file_path = os.path.join(data_root, f"{app}_training_.csv")
            if not os.path.exists(file_path):
                print(f"Warning: {file_path} not found.")
                continue
            
            print(f"Loading and Scaling {app}...")
            df = pd.read_csv(file_path)
            if len(df.columns) == 10:
                vals = df.iloc[:, 1:].values 
            else:
                vals = df.values 
                
            d_min = vals.min(axis=0)
            d_max = vals.max(axis=0)
            self.min_max_values[app] = (d_min, d_max) 
            
            vals = (vals - d_min) / (d_max - d_min + 1e-7)
            
            num_windows = len(vals) // seq_len
            windows = vals[:num_windows * seq_len].reshape(-1, seq_len, 9)
            
            windows = windows.transpose(0, 2, 1) 
            windows = windows[:, :, np.newaxis, :] 
            
            self.data.append(windows.astype(np.float32))
            self.labels.append(np.full((num_windows,), idx, dtype=np.int64))
            
        self.data = np.concatenate(self.data, axis=0)
        self.labels = np.concatenate(self.labels, axis=0)
        print(f"Total dataset size: {len(self.data)} windows.")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def gen_plot_nilm(gen_net, epoch, args):
    gen_net.eval()
    synthetic_data = [] 
    synthetic_labels = []
    
    with torch.no_grad():
        for idx in range(5): 
            fake_noise = torch.randn(1, args.latent_dim).cuda()
            fake_label = torch.full((1,), idx, dtype=torch.long).cuda()
            fake_sigs = gen_net(fake_noise, fake_label).cpu().numpy()
            synthetic_data.append(fake_sigs)
            synthetic_labels.append(idx)

    fig, axs = plt.subplots(1, 5, figsize=(25, 5))
    fig.suptitle(f'NILM Synthetic Power (Channel 0) at epoch {epoch}', fontsize=20)
    for i in range(5):
        axs[i].plot(synthetic_data[i][0, 0, 0, :], color='blue')
        axs[i].set_title(f"App: {args.appliances[i]}")
        axs[i].set_ylim([-0.1, 1.1]) 
    
    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg')
    buf.seek(0)
    plt.close()
    return buf

def main():
    args = Args()
    torch.cuda.set_device(args.gpu)
    
    gen_net = Generator(seq_len=args.seq_len, channels=args.channels, num_classes=args.num_classes, 
                        latent_dim=args.latent_dim, data_embed_dim=128, label_embed_dim=128, 
                        depth=4, num_heads=8).cuda()
    dis_net = Discriminator(in_channels=args.channels, patch_size=16, data_emb_size=128, label_emb_size=128,
                            seq_length=args.seq_len, depth=4, n_classes=args.num_classes, num_heads=8).cuda()
    
    dataset = NILMDataset(args.data_path, args.appliances, args.seq_len)
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    
    gen_optimizer = torch.optim.Adam(gen_net.parameters(), lr=args.g_lr, betas=(args.beta1, args.beta2))
    dis_optimizer = torch.optim.Adam(dis_net.parameters(), lr=args.d_lr, betas=(args.beta1, args.beta2))
    
    gen_scheduler = LinearLrDecay(gen_optimizer, args.g_lr, 0.0, 0, args.max_epoch * len(train_loader))
    dis_scheduler = LinearLrDecay(dis_optimizer, args.d_lr, 0.0, 0, args.max_epoch * len(train_loader))
    
    args.path_helper = set_log_dir('logs', args.exp_name)
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(args.path_helper['log_path'])
    
    fixed_z = torch.randn(10, args.latent_dim).cuda()
    gen_avg_param = copy_params(gen_net)
    
    os.makedirs(args.path_helper['log_path'], exist_ok=True)
    with open(os.path.join(args.path_helper['log_path'], 'min_max_values.pkl'), 'wb') as f:
        pickle.dump(dataset.min_max_values, f)

    writer_dict = {
        'writer': writer,
        'train_global_steps': 0,
    }
    
    print("Starting Training...")
    for epoch in range(args.max_epoch):
        train(args, gen_net, dis_net, gen_optimizer, dis_optimizer, gen_avg_param, 
              train_loader, epoch, writer_dict, fixed_z, (gen_scheduler, dis_scheduler))
            
        if epoch % 1 == 0:
            plot_buf = gen_plot_nilm(gen_net, epoch, args)
            image = PIL.Image.open(plot_buf)
            image_tensor = ToTensor()(image)
            writer.add_image('Synthetic_Data', image_tensor, epoch)
            
            torch.save({
                'epoch': epoch,
                'gen_state_dict': gen_net.state_dict(),
                'dis_state_dict': dis_net.state_dict(),
                'gen_avg_param': gen_avg_param,
            }, os.path.join(args.path_helper['ckpt_path'], f"checkpoint_{epoch}.pth"))

if __name__ == '__main__':
    main()
