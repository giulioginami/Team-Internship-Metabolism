"""
gating_network.py  —  Mixture-of-Experts gating network
=========================================================

Architecture:
  Input  : sparse glucose + insulin observations at 5 time points
           (concatenated → 10 features per patient)
  Gating : small MLP → 3 softmax weights  [w_NGT, w_IGT, w_T2DM]
  Output : weighted combination of the three PID expert predictions
             prediction = w_NGT * PID_NGT + w_IGT * PID_IGT + w_T2DM * PID_T2DM

Training objective:
  Minimise the normalised MSE between the weighted PID prediction and the
  actual sparse noisy observations, for both glucose and insulin.

Prerequisites — run these in MATLAB first:
  1. Generate_VirtualPopulation.m   → virtual_population.mat
  2. Label_VirtualPopulation.m      → virtual_population_labelled.mat
  3. Create_SparseDatasets.m        → virtual_population_sparse.mat
  4. PID_optimization_full.m        → optimised k5, k6, k8 per population
  5. Save_PID_Predictions.m         → pid_predictions.mat
     (script that runs the EDES ODE for every patient × 3 PIDs and saves
      predictions at the sparse time points)

     Expected fields in pid_predictions.mat:
       G_pids  [N x 3 x 5]  glucose at sparse times  (axis 1: NGT / IGT / T2DM expert)
       I_pids  [N x 3 x 5]  insulin at sparse times
       labels  [N x 1]      ground-truth class  1=NGT  2=IGT  3=T2DM
"""

import numpy as np
import h5py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt


def load_struct(f, name):
    """Load a MATLAB struct from an HDF5 file.
    MATLAB v7.3 stores arrays in column-major order, so we transpose."""
    g = f[name]
    return {
        'glucose_noisy': np.array(g['glucose_noisy']).T,   # [n x 5]
        'insulin_noisy': np.array(g['insulin_noisy']).T,   # [n x 5]
        'param_matrix':  np.array(g['param_matrix']).T,    # [n x 5]
        'n':             int(np.array(g['n']).flatten()[0]),
    }

# ─────────────────────────────────────────────────────────────────────────────
# Settings
# ─────────────────────────────────────────────────────────────────────────────
SPARSE_FILE  = 'virtual_population_sparse.mat'
PRED_FILE    = 'pid_predictions.mat'

HIDDEN_DIM   = 32       # neurons per hidden layer
EPOCHS       = 300
BATCH_SIZE   = 64
LR           = 1e-3
TRAIN_RATIO  = 0.8
SEED         = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

# ─────────────────────────────────────────────────────────────────────────────
# Load sparse observations  (virtual_population_sparse.mat)
# ─────────────────────────────────────────────────────────────────────────────
with h5py.File(SPARSE_FILE, 'r') as f:
    ngt  = load_struct(f, 'dataset_NGT_sparse')
    igt  = load_struct(f, 'dataset_IGT_sparse')
    t2dm = load_struct(f, 'dataset_T2DM_sparse')

# Stack all three populations row-wise  →  [N x 7]
G_obs = np.vstack([ngt['glucose_noisy'],  igt['glucose_noisy'],  t2dm['glucose_noisy']])
I_obs = np.vstack([ngt['insulin_noisy'],  igt['insulin_noisy'],  t2dm['insulin_noisy']])

n_ngt  = ngt['n']
n_igt  = igt['n']
n_t2dm = t2dm['n']
N      = G_obs.shape[0]

print(f'Loaded sparse data:  NGT={n_ngt}  IGT={n_igt}  T2DM={n_t2dm}  total={N}')

# ─────────────────────────────────────────────────────────────────────────────
# Load precomputed PID predictions  (pid_predictions.mat)
# ─────────────────────────────────────────────────────────────────────────────
with h5py.File(PRED_FILE, 'r') as f:
    # MATLAB stores [N x 3 x 7] as [7 x 3 x N] in HDF5 → transpose back
    G_pids = np.array(f['G_pids']).T.astype(np.float32)   # [N x 3 x 7]
    I_pids = np.array(f['I_pids']).T.astype(np.float32)   # [N x 3 x 7]
    labels = np.array(f['labels']).flatten().astype(int) - 1  # 0-indexed

assert G_pids.shape == (N, 3, 5), f'Expected G_pids shape ({N}, 3, 5), got {G_pids.shape}'
assert I_pids.shape == (N, 3, 5), f'Expected I_pids shape ({N}, 3, 5), got {I_pids.shape}'

# ─────────────────────────────────────────────────────────────────────────────
# Build and normalise input features:  X = [G_sparse | I_sparse]  →  [N x 10]
# ─────────────────────────────────────────────────────────────────────────────
X      = np.concatenate([G_obs, I_obs], axis=1).astype(np.float32)
X_mean = X.mean(axis=0)
X_std  = X.std(axis=0) + 1e-8
X_norm = (X - X_mean) / X_std

# ─────────────────────────────────────────────────────────────────────────────
# Convert to tensors
# ─────────────────────────────────────────────────────────────────────────────
X_t      = torch.tensor(X_norm)
G_obs_t  = torch.tensor(G_obs.astype(np.float32))
I_obs_t  = torch.tensor(I_obs.astype(np.float32))
G_pids_t = torch.tensor(G_pids)
I_pids_t = torch.tensor(I_pids)

# ─────────────────────────────────────────────────────────────────────────────
# Train / test split  (stratified by population)
# ─────────────────────────────────────────────────────────────────────────────
def stratified_split(n, ratio, seed=SEED):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_tr = int(n * ratio)
    return idx[:n_tr], idx[n_tr:]

# Split each population separately so all three appear in both sets
tr_ngt,  te_ngt  = stratified_split(n_ngt,  TRAIN_RATIO)
tr_igt,  te_igt  = stratified_split(n_igt,  TRAIN_RATIO)
tr_t2dm, te_t2dm = stratified_split(n_t2dm, TRAIN_RATIO)

# Offset indices to global positions
tr_igt  += n_ngt;       te_igt  += n_ngt
tr_t2dm += n_ngt+n_igt; te_t2dm += n_ngt+n_igt

tr = np.concatenate([tr_ngt, tr_igt, tr_t2dm])
te = np.concatenate([te_ngt, te_igt, te_t2dm])

def sel(t, idx): return t[idx]

X_tr, X_te           = sel(X_t, tr),      sel(X_t, te)
G_obs_tr, G_obs_te   = sel(G_obs_t, tr),  sel(G_obs_t, te)
I_obs_tr, I_obs_te   = sel(I_obs_t, tr),  sel(I_obs_t, te)
G_pids_tr, G_pids_te = sel(G_pids_t, tr), sel(G_pids_t, te)
I_pids_tr, I_pids_te = sel(I_pids_t, tr), sel(I_pids_t, te)

print(f'Train: {len(tr)}  |  Test: {len(te)}')

train_ds = TensorDataset(X_tr, G_pids_tr, I_pids_tr, G_obs_tr, I_obs_tr)
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

# ─────────────────────────────────────────────────────────────────────────────
# Gating network definition
# ─────────────────────────────────────────────────────────────────────────────
class GatingNetwork(nn.Module):
    """
    Maps 10 input features (sparse G + I) to 3 softmax weights.
    Two hidden layers of HIDDEN_DIM neurons with ReLU activations.
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(10, HIDDEN_DIM), nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.ReLU(),
            nn.Linear(HIDDEN_DIM, 3),
            nn.Softmax(dim=-1),
        )

    def forward(self, x):
        return self.net(x)   # [batch x 3]

model     = GatingNetwork()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

print(f'\nGating network parameters: {sum(p.numel() for p in model.parameters())}')

# ─────────────────────────────────────────────────────────────────────────────
# Loss function
# ─────────────────────────────────────────────────────────────────────────────
# Precompute signal variances for normalisation (computed on the full dataset)
G_var = float(G_obs_t.var()) + 1e-8
I_var = float(I_obs_t.var()) + 1e-8

def moe_loss(w, G_pids, I_pids, G_obs, I_obs):
    """
    w:      [batch x 3]       gating weights (sum to 1)
    G_pids: [batch x 3 x 5]  glucose predictions of each PID expert
    G_obs:  [batch x 5]      observed sparse glucose

    Returns the sum of normalised MSE for glucose and insulin.
    """
    w3     = w.unsqueeze(-1)                      # [batch x 3 x 1]
    G_pred = (w3 * G_pids).sum(dim=1)            # [batch x 5]
    I_pred = (w3 * I_pids).sum(dim=1)            # [batch x 5]
    loss_G = (G_pred - G_obs).pow(2).mean() / G_var
    loss_I = (I_pred - I_obs).pow(2).mean() / I_var
    return loss_G + loss_I

# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────
train_losses, test_losses = [], []

for epoch in range(EPOCHS):
    model.train()
    running = 0.0
    for X_b, G_pids_b, I_pids_b, G_obs_b, I_obs_b in train_dl:
        optimizer.zero_grad()
        loss = moe_loss(model(X_b), G_pids_b, I_pids_b, G_obs_b, I_obs_b)
        loss.backward()
        optimizer.step()
        running += loss.item() * len(X_b)
    train_losses.append(running / len(tr))

    model.eval()
    with torch.no_grad():
        test_loss = moe_loss(
            model(X_te), G_pids_te, I_pids_te, G_obs_te, I_obs_te
        ).item()
    test_losses.append(test_loss)

    if (epoch + 1) % 50 == 0:
        print(f'Epoch {epoch+1:3d}/{EPOCHS}  '
              f'train loss={train_losses[-1]:.4f}  '
              f'test loss={test_loss:.4f}')

# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────
model.eval()
with torch.no_grad():
    w_all = model(X_t).numpy()   # [N x 3]

print('\n--- Average gating weights (full dataset) ---')
for i, name in enumerate(['NGT', 'IGT', 'T2DM']):
    print(f'  w_{name:4s}: {w_all[:, i].mean():.3f} ± {w_all[:, i].std():.3f}')

# Per-population breakdown
pop_slices = {
    'NGT':  slice(0, n_ngt),
    'IGT':  slice(n_ngt, n_ngt + n_igt),
    'T2DM': slice(n_ngt + n_igt, N),
}
print('\n--- Average weights broken down by true population ---')
print(f'{"":6s}  {"w_NGT":>8}  {"w_IGT":>8}  {"w_T2DM":>8}')
for pop, sl in pop_slices.items():
    w = w_all[sl]
    print(f'{pop:6s}  {w[:,0].mean():8.3f}  {w[:,1].mean():8.3f}  {w[:,2].mean():8.3f}')

# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(15, 4))

# Training curve
axes[0].plot(train_losses, label='Train')
axes[0].plot(test_losses,  label='Test')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Normalised MSE')
axes[0].set_title('Training curve')
axes[0].legend()
axes[0].grid(True)

# Weight distribution — boxplot across all patients
# axes[1].boxplot(w_all, labels=['NGT', 'IGT', 'T2DM'])
# axes[1].set_ylabel('Gating weight')
# axes[1].set_title('Weight distribution (full dataset)')
# axes[1].set_ylim([0, 1])
# axes[1].grid(True)

# Per-population mean weights — grouped bar chart
pop_names  = ['NGT', 'IGT', 'T2DM']
expert_names = ['w_NGT', 'w_IGT', 'w_T2DM']
colors = ['#2e9e2e', '#ee9b14', '#cc2222']
x = np.arange(3)
width = 0.25
for j in range(3):
    means = [w_all[sl, j].mean() for sl in pop_slices.values()]
    axes[1].bar(x + j * width, means, width, label=expert_names[j], color=colors[j], alpha=0.8)
axes[1].set_xticks(x + width)
axes[1].set_xticklabels(pop_names)
axes[1].set_ylabel('Mean gating weight')
axes[1].set_title('Mean weights per true population')
axes[1].set_ylim([0, 1])
axes[1].legend()
axes[1].grid(True, axis='y')

plt.tight_layout()
plt.savefig('gating_network_results.png', dpi=150)
plt.show()

# ─────────────────────────────────────────────────────────────────────────────
# Save model and normalisation stats for inference
# ─────────────────────────────────────────────────────────────────────────────
torch.save({
    'model_state': model.state_dict(),
    'X_mean':      X_mean,
    'X_std':       X_std,
}, 'gating_network.pt')
print('\nModel saved to gating_network.pt')

# ─────────────────────────────────────────────────────────────────────────────
# Export weights to .mat so MATLAB can run the forward pass directly
# ─────────────────────────────────────────────────────────────────────────────
from scipy.io import savemat

sd = model.state_dict()
savemat('gating_weights.mat', {
    'W1':     sd['net.0.weight'].numpy(),   # [32 x 14]
    'b1':     sd['net.0.bias'].numpy(),     # [32]
    'W2':     sd['net.2.weight'].numpy(),   # [32 x 32]
    'b2':     sd['net.2.bias'].numpy(),     # [32]
    'W3':     sd['net.4.weight'].numpy(),   # [3  x 32]
    'b3':     sd['net.4.bias'].numpy(),     # [3]
    'X_mean': X_mean,
    'X_std':  X_std,
})
print('Gating weights saved to gating_weights.mat')
