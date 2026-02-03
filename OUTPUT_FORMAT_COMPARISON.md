# TTS-CGAN vs Diffusion Model - Output Comparison

## 📊 Output Format Comparison

| Feature | Diffusion Model | TTS-CGAN |
|---------|----------------|----------|
| **Output Directory** | `OUTPUT/{appliance}_multivariate/` | `OUTPUT/{appliance}_multivariate/` ✅ |
| **Filename** | `ddpm_fake_{appliance}_multivariate.npy` | `cgan_fake_{appliance}_multivariate.npy` ✅ |
| **Shape** | `(N, 512, 9)` | `(N, 512, 9)` ✅ |
| **Column 0 (Power)** | Z-score | Z-score ✅ |
| **Columns 1-8 (Time)** | [-1, 1] | [-1, 1] ✅ |
| **Sample Count** | `(totalPoints / 512 + 1) × 2` | `(totalPoints / 512) × 2` ⚠️ (~1 window diff) |

---

## 📁 Example Output Paths

### Diffusion:
```
OUTPUT/
├── dishwasher_multivariate/
│   └── ddpm_fake_dishwasher_multivariate.npy  # (4917, 512, 9)
├── fridge_multivariate/
│   └── ddpm_fake_fridge_multivariate.npy
└── ...
```

### TTS-CGAN:
```
OUTPUT/
├── dishwasher_multivariate/
│   └── cgan_fake_dishwasher_multivariate.npy  # (4916, 512, 9)
├── fridge_multivariate/
│   └── cgan_fake_fridge_multivariate.npy
└── ...
```

---

## 🔬 Data Range Verification

Both models output the same data format:

```python
import numpy as np

# Load generated data
data = np.load('OUTPUT/dishwasher_multivariate/cgan_fake_dishwasher_multivariate.npy')

print(f"Shape: {data.shape}")            # (N, 512, 9)
print(f"Power (col 0): {data[:,:,0].min():.3f} to {data[:,:,0].max():.3f}")  # Z-score
print(f"Time (col 1-8): {data[:,:,1:].min():.3f} to {data[:,:,1:].max():.3f}")  # [-1, 1]
```

---

## ✅ Compatibility

Both outputs can be directly used by:
- NILMFormer training pipeline
- Data quality evaluation scripts
- Waveform visualization tools

No conversion needed! 🚀
