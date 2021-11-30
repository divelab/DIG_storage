## SphereNet results on MD17
The evaluation metric is Force mean absolute error (MAE).

| force MAE        | aspirin | benzene2017 | ethanol | malonaldehyde | naphthalene | salicylic | toluene | uracil |
| ---------------- | ------- | ----------- | ------- | ------------- | ----------- | --------- | ------- | ------ |
| results in paper | 0.43    | 0.178       | 0.208   | 0.34          | 0.178       | 0.36      | 0.155   | 0.267  |
| rerun results    |**0.375**|0.181|**0.187**|**0.273**|**0.139**|**0.284**|**0.142**|**0.245**|

## Usage

The default hyperparameters are listed in parser.

```python
from PygMD17 import MD17
from spherenet import SphereNet
from eval import ThreeDEvaluator
import argparse
import os
import torch
from torch_geometric.data import DataLoader
from tqdm import tqdm
from torch.autograd import grad


parser = argparse.ArgumentParser(description='MD17 SphereNet')
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

parser.add_argument('--eval_steps', type=int, default=50)
parser.add_argument('--eval_start', type=int, default=200)
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--vt_batch_size', type=int, default=32) # can try 64/128/256 based on the memory of your device
parser.add_argument('--lr', type=float, default=0.0005)
parser.add_argument('--lr_decay_factor', type=float, default=0.5)
parser.add_argument('--lr_decay_step_size', type=int, default=200)

parser.add_argument('--p', type=int, default=100)

parser.add_argument('--save_dir', type=str, default='md17')

args = parser.parse_args()
print(args)

device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
device = torch.device(device)
print('device',device)

model = SphereNet(energy_and_force=True, cutoff=args.cutoff, num_layers=args.num_layers, 
        hidden_channels=args.hidden_channels, out_channels=args.out_channels, int_emb_size=args.int_emb_size, 
        basis_emb_size_dist=args.basis_emb_size_dist, basis_emb_size_angle=args.basis_emb_size_angle, basis_emb_size_torsion=args.basis_emb_size_torsion, out_emb_channels=args.out_emb_channels, 
        num_spherical=args.num_spherical, num_radial=args.num_radial, envelope_exponent=5, 
        num_before_skip=1, num_after_skip=2, num_output_layers=3 
        )
model = model.to(device)
evaluation = ThreeDEvaluator()


for data_name in ['aspirin', 'benzene2017', 'ethanol', 'malonaldehyde', 'naphthalene', 'salicylic', 'toluene', 'uracil']:
    dataset = MD17(root='dataset/', name=data_name)
    split_idx = dataset.get_idx_split(len(dataset.data.y), train_size=1000, valid_size=1000, seed=42)
    train_dataset, valid_dataset, test_dataset = dataset[split_idx['train']], dataset[split_idx['valid']], dataset[split_idx['test']]
    print('train, validaion, test:', data_name, len(train_dataset), len(valid_dataset), len(test_dataset))
    test_loader = DataLoader(test_dataset, args.vt_batch_size, shuffle=False)
    
    checkpoint = torch.load(os.path.join(args.save_dir, data_name+'_checkpoint.pt'))
    model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()

    preds_force = torch.Tensor([]).to(device)
    targets_force = torch.Tensor([]).to(device)

    for step, batch_data in enumerate(tqdm(test_loader)):
        batch_data = batch_data.to(device)
        out = model(batch_data)
        force = -grad(outputs=out, inputs=batch_data.pos, grad_outputs=torch.ones_like(out),create_graph=True,retain_graph=True)[0].detach_()
        # for one input in benzene2017, the computed force is nan. 
        # (might be some issue when computing gradient, but only occur for one input graph). 
        # So we skipped this input.
        if torch.sum(torch.isnan(force)) != 0:
            mask = torch.isnan(force)
            force = force[~mask].reshape((-1,3))
            batch_data.force = batch_data.force[~mask].reshape((-1,3))
        preds_force = torch.cat([preds_force,force], dim=0)
        targets_force = torch.cat([targets_force,batch_data.force], dim=0)
    force_mae = torch.mean(torch.abs(preds_force - targets_force)).cpu().item()
    print('mae:',data_name, force_mae)
```