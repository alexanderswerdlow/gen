from math import dist
import autoroot
from pathlib import Path
from typing import Optional

from tensordict import TensorDict
from tqdm import tqdm
import typer
import numpy as np
import torch
from scipy.spatial import KDTree
from gen.utils.decoupled_utils import breakpoint_on_error
import matplotlib.pyplot as plt
import io
from PIL import Image
from image_utils import Im
import torch.nn.functional as F
import einops

typer.main.get_command_name = lambda name: name
app = typer.Typer(pretty_exceptions_show_locals=False)

def distance_matrix(x, y=None, p = 2): #pairwise distance of vectors
    
    y = x if type(y) == type(None) else y

    n = x.size(0)
    m = y.size(0)
    d = x.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)
    
    dist = torch.linalg.vector_norm(x - y, p, 2) if torch.__version__ >= '1.7.0' else torch.pow(x - y, p).sum(2)**(1/p)
    
    return dist


def compute_dist(queries, data, slice_size=100):
    dist = torch.zeros(queries.shape[0], data.shape[0], dtype=data.dtype)
    for i in tqdm(range(int(np.ceil(data.shape[0] / slice_size)))):
        start = slice_size * i
        end = slice_size * (i + 1)
        dist_ = distance_matrix(queries, data[start:end])
        dist[:, start:end] = dist_

    return dist

def get_img():
    buf = io.BytesIO()
    plt.savefig(buf, format="jpeg", bbox_inches="tight")
    buf.seek(0)
    img = Im(Image.open(buf))
    plt.close("all")
    return img

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW

import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate=0.5):
        super(MLP, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate)
        )
        self.layer3 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

def train_mlp(train_data, val_data, num_classes, num_epochs=20, batch_size=4096, learning_rate=2e-3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = MLP(train_data["mask_tokens"].shape[1], train_data["mask_tokens"].shape[1] // 2, num_classes).to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    train_dataset = TensorDataset(train_data["mask_tokens"], train_data["categories"])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

    val_dataset = TensorDataset(val_data["mask_tokens"], val_data["categories"])
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

    for epoch in tqdm(range(num_epochs), total=num_epochs):
        model.train()
        for inputs, labels in tqdm(train_loader, leave=False, total=len(train_loader)):
            inputs, labels = inputs.to(device).to(torch.float32), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        model.eval()
        all_val_preds = []
        all_val_labels = []
        with torch.no_grad():
            for val_inputs, val_labels in val_loader:
                val_inputs, val_labels = val_inputs.to(device).to(torch.float32), val_labels.to(device)
                val_preds = model(val_inputs)
                all_val_preds.append(val_preds.cpu())
                all_val_labels.append(val_labels.cpu())
        
        all_val_preds = torch.cat(all_val_preds)
        all_val_labels = torch.cat(all_val_labels)
        
        plot_data(all_val_preds, all_val_labels, num_classes, name=f'mlp_epoch_{epoch}')

    return model

def plot(pred, tgt, num_classes, top_k) -> tuple:
    from torchmetrics.classification import MulticlassConfusionMatrix
    from torchmetrics.classification import MulticlassAccuracy

    # metric = MulticlassConfusionMatrix(num_classes=num_classes)
    # metric.update(pred, tgt)
    # fig, ax = metric.plot()
    # confusion_matrix = get_img()
    
    metric = MulticlassAccuracy(num_classes=num_classes, top_k=top_k)
    metric.update(pred, tgt)
    fig, ax = metric.plot()
    accuracy = get_img()

    # Im.concat_horizontal(confusion_matrix, accuracy)
    img = Im(accuracy)

    return metric, img


def load_tensordict(path):
    """
    Loads a tensordict from each subdir and concats
    """
    data = []
    for subdir in path.iterdir():
        data.append(TensorDict.load_memmap(subdir).cpu())

    print(f"Loaded {len(data)} tensordicts from {path}")
    
    return torch.cat(data, dim=0)

def knn_probing(train_data, val_data, num_classes, split_token_data, prefix):
    gt_labels = val_data['categories']

    train_tokens = einops.rearrange(train_data['mask_tokens'], "tokens (layers d) -> tokens layers d", layers=split_token_data[0])[:, split_token_data[1]].numpy()

    val_tokens = einops.rearrange(val_data['mask_tokens'], "tokens (layers d) -> tokens layers d", layers=split_token_data[0])[:, split_token_data[1]].numpy()

    import faiss
    co = faiss.GpuMultipleClonerOptions()
    co.useFloat16 = True
    co.useFloat16CoarseQuantizer = True
    co.usePrecomputed = True

    num_train_tokens, train_token_dim = train_tokens.shape
    nlist = int(4 * np.sqrt(num_train_tokens))
    cpu_index = faiss.IndexFlatL2(train_token_dim)
    index = faiss.IndexIVFFlat(cpu_index, train_token_dim, nlist)
    index = faiss.index_cpu_to_all_gpus(cpu_index, co, ngpu=faiss.get_num_gpus())
    
    index_batch_size = 50000
    for i in tqdm(range(0, num_train_tokens, index_batch_size), leave=False):
        index.add(train_tokens[i : i + index_batch_size])

    dists, idx = index.search(val_tokens, k=5)

    dists = torch.from_numpy(dists).to(torch.float32)
    pred_labels = train_data['categories'][idx]

    probs = torch.nn.functional.softmax(dists, dim=-1)
    pred_prob = torch.zeros(dists.shape[0], num_classes)
    pred_prob.scatter_add_(1, pred_labels, probs)

    postfix = f"{split_token_data[1]}_{split_token_data[0]}" if split_token_data[0] > 1 else ""
    plot_data(pred_prob, gt_labels, num_classes, name=f'{prefix} KNN {postfix}')

def plot_data(pred_prob, gt_labels, num_classes, name, should_plot=False):
    values = []
    for k in [1, 3, 5, 10]:
        metric, img = plot(pred_prob, gt_labels, num_classes=num_classes, top_k=k)
        if should_plot:
            img.save(f'output_debug/{name}_top{k}.png')
        values.append(100 * metric.compute())
        # print(f"{name}, Top K={k} Accuracy: {100 * metric.compute():.2f}%")

    formatted_values = ', '.join([f'{v:.2f}' for v in values])
    print(f"('{name}', [{formatted_values}]),")

@app.command()
def main(
    breakpoint_on_start: bool = False,
    knn: bool = True,
    mlp: bool = True,
    split_tokens: Optional[int] = 1,
    exp: str = "",
    path: Optional[Path] = Path("outputs/inference/2024-03-27_knn_v5/inference")
):
    with breakpoint_on_error():
        val_data = load_tensordict(path / "knn_validation")
        train_data = load_tensordict(path / "knn_train")
        num_classes = 133

        train_data = train_data[train_data['categories'] >= 0]
        val_data = val_data[val_data['categories'] >= 0]

        if split_tokens > 1:
            for i in range(split_tokens):
                split_token_data = (split_tokens, i)
                if knn: knn_probing(train_data, val_data, num_classes, split_token_data, exp)
                if mlp: train_mlp(train_data, val_data, num_classes, split_token_data), exp

        if knn: knn_probing(train_data, val_data, num_classes, (1, 0), exp)
        if mlp: train_mlp(train_data, val_data, num_classes, (1, 0), exp)

        if breakpoint_on_start: breakpoint()


if __name__ == "__main__":
    with breakpoint_on_error():
        app()