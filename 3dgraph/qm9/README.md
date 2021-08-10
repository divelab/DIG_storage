## SphereNet results on QM9
The evaluation metric is mean absolute error (MAE).

|            | mu     | alpha  | homo | lumo | gap  | r2    | zpve | U0   | U    | H    | G    | Cv     | std. MAE |
| ---------- | ------ | ------ | ---- | ---- | ---- | ----- | ---- | ---- | ---- | ---- | ---- | ------ | -------- |
| reported   | 0.0269 | 0.0465 | 23.6 | 18.9 | 32.3 | 0.292 | 1.12 | 6.26 | 7.33 | 6.4  | 8    | 0.0215 | 0.94     |
| reproduced | **0.0245** | **0.0449** | **22.8** | 19.6 | **31.1** | **0.268** | 1.26 | 6.31 | **6.36** | **6.33** | **7.78** | 0.024  | **0.92**     |

## Usage

The default hyperparameters are listed in parser.

The 'gap' is predicted by taking 'lumo'-'homo'.

For 'r2', we used num_spherical=2. I got MAE=0.296 when using num_spherical=3, but I didn't save correct model.

For 'mu', 'alpha', 'G', 'H', 'Cv', we used lr_decay_step_size=150.

```python
from dig.threedgraph.dataset import QM93D
from dig.threedgraph.method import SphereNet
from dig.threedgraph.evaluation import ThreeDEvaluator
import argparse
import os
import torch
from torch_geometric.data import DataLoader
from tqdm import tqdm


parser = argparse.ArgumentParser(description='QM9 SphereNet')
parser.add_argument('--device', type=int, default=0)

parser.add_argument('--cutoff', type=float, default=5.0)
parser.add_argument('--num_layers', type=int, default=4)
parser.add_argument('--hidden_channels', type=int, default=128)
parser.add_argument('--out_channels', type=int, default=1)
parser.add_argument('--int_emb_size', type=int, default=64)
parser.add_argument('--basis_emb_size_dist', type=int, default=8)
parser.add_argument('--basis_emb_size_angle', type=int, default=8)
parser.add_argument('--basis_emb_size_torsion', type=int, default=8)
parser.add_argument('--out_emb_channels', type=int, default=256)
parser.add_argument('--num_spherical', type=int, default=3)
parser.add_argument('--num_radial', type=int, default=6)

parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--vt_batch_size', type=int, default=32)
parser.add_argument('--lr', type=float, default=0.0005)
parser.add_argument('--lr_decay_factor', type=float, default=0.5)
parser.add_argument('--lr_decay_step_size', type=int, default=100)

args = parser.parse_args()
print(args)

device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
device = torch.device(device)
print('device',device)

save_dir='qm9' 
vt_batch_size = 32

dataset = QM93D(root='dataset/', cutoff=args.cutoff)
split_idx = dataset.get_idx_split(len(dataset.data.y), train_size=110000, valid_size=10000, seed=42)

model = SphereNet(energy_and_force=False, cutoff=args.cutoff, num_layers=args.num_layers, 
            hidden_channels=args.hidden_channels, out_channels=args.out_channels, int_emb_size=args.int_emb_size, 
            basis_emb_size_dist=args.basis_emb_size_dist, basis_emb_size_angle=args.basis_emb_size_angle, basis_emb_size_torsion=args.basis_emb_size_torsion, out_emb_channels=args.out_emb_channels, 
            num_spherical=args.num_spherical, num_radial=args.num_radial, envelope_exponent=5, 
            num_before_skip=1, num_after_skip=2, num_output_layers=3 
            )
model = model.to(device)

evaluation = ThreeDEvaluator()


for target in ['mu','alpha','homo','lumo','zpve','U0','U','H','G','Cv']:
    dataset.data.y = dataset.data[target]
    train_dataset, valid_dataset, test_dataset = dataset[split_idx['train']], dataset[split_idx['valid']], dataset[split_idx['test']]
    print('train, validaion, test:', len(train_dataset), len(valid_dataset), len(test_dataset))
    test_loader = DataLoader(test_dataset, vt_batch_size, shuffle=False)
    
    checkpoint = torch.load(os.path.join(save_dir, target+'_checkpoint.pt'))
    model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()

    preds = torch.Tensor([]).to(device)
    targets = torch.Tensor([]).to(device)

    for step, batch_data in enumerate(tqdm(test_loader)):
        batch_data = batch_data.to(device)
        with torch.no_grad():
            out = model(batch_data)
        preds = torch.cat([preds, out.detach_()], dim=0)
        targets = torch.cat([targets, batch_data.y.unsqueeze(1)], dim=0)

    mae = torch.mean(torch.abs(preds - targets)).cpu().item()
    print('mae:',target, mae)
```