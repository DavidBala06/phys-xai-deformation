from pathlib import Path
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import trimesh
import matplotlib.pyplot as plt

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


def load_real_colon_mesh(obj_path: str = "data/full_model.obj", target_nodes: int = 1024, k_neighbors: int = 4):
    mesh = trimesh.load(obj_path, process=False)
    all_vertices = np.array(mesh.vertices, dtype=np.float32)
    total_nodes = all_vertices.shape[0]

    step = max(1, total_nodes // target_nodes)
    sampled_indices = np.arange(0, total_nodes, step)[:target_nodes]
    vertices = all_vertices[sampled_indices]
    vertices -= vertices.mean(axis=0)

    num_sampled = vertices.shape[0]
    v_tensor = torch.from_numpy(vertices)
    dist_matrix = torch.cdist(v_tensor, v_tensor)
    dist_matrix.fill_diagonal_(float("inf"))
    _, neighbor_indices = torch.topk(dist_matrix, k=k_neighbors, largest=False)

    edges_set = set()
    for src in range(num_sampled):
        for dst in neighbor_indices[src].tolist():
            edges_set.add(tuple(sorted((src, dst))))

    edges = np.array(list(edges_set), dtype=np.int64)
    return torch.tensor(vertices, dtype=torch.float32), torch.tensor(edges, dtype=torch.long)


class ColonElasticLoss(nn.Module):
    def __init__(self, rest_vertices: torch.Tensor, edges: torch.Tensor):
        super().__init__()
        self.register_buffer("rest_vertices", rest_vertices.float())
        self.register_buffer("edges", edges.long())

        idx1, idx2 = self.edges[:, 0], self.edges[:, 1]
        v1, v2 = self.rest_vertices[idx1], self.rest_vertices[idx2]
        rest_lengths = torch.sqrt(torch.sum((v1 - v2) ** 2, dim=-1) + 1e-8)
        self.register_buffer("rest_lengths", rest_lengths)

    def forward(self, displacements: torch.Tensor) -> torch.Tensor:
        deformed_v = self.rest_vertices.unsqueeze(0) + displacements
        idx1, idx2 = self.edges[:, 0], self.edges[:, 1]
        v1, v2 = deformed_v[:, idx1, :], deformed_v[:, idx2, :]
        current_lengths = torch.sqrt(torch.sum((v1 - v2) ** 2, dim=-1) + 1e-8)
        return F.mse_loss(current_lengths, self.rest_lengths.unsqueeze(0).expand_as(current_lengths))


class ColonDeformationPINN(nn.Module):
    def __init__(self, in_features: int = 32, num_nodes: int = 1024):
        super().__init__()
        self.num_nodes = num_nodes
        self.net = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, num_nodes * 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).view(-1, self.num_nodes, 3)


def save_plots(history: dict, rest_v: np.ndarray, u_true: np.ndarray, u_pred: np.ndarray, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    steps = history["step"]

    # 1. Training curves (Losses & TRE)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(steps, history["data_loss"], label="Data Loss (MSE)")
    axes[0].plot(steps, history["phys_loss"], label="Elastic Loss")
    axes[0].set_yscale("log")
    axes[0].set_title("Training Losses (log scale)")
    axes[0].set_xlabel("Optimization Step")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(steps, history["tre_mm"], color="tab:red", label="Mean TRE")
    axes[1].axhline(2.0, color="tab:green", linestyle="--", linewidth=1.2, label="Clinical reference (<2.0 mm)")
    axes[1].set_title("Target Registration Error (TRE)")
    axes[1].set_xlabel("Optimization Step")
    axes[1].set_ylabel("TRE (mm)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_dir / "training_curves.png", dpi=200)
    plt.close()

    # 2. 3D projection comparison (XY & XZ views)
    deformed_gt = rest_v + u_true
    deformed_pred = rest_v + u_pred

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].scatter(rest_v[:, 0], rest_v[:, 1], s=4, alpha=0.35, label="Rest mesh")
    axes[0].scatter(deformed_gt[:, 0], deformed_gt[:, 1], s=4, alpha=0.5, label="Target deformed")
    axes[0].scatter(deformed_pred[:, 0], deformed_pred[:, 1], s=4, alpha=0.6, marker="x", label="PINN predicted")
    axes[0].set_title("XY Projection")
    axes[0].set_xlabel("X (mm)")
    axes[0].set_ylabel("Y (mm)")
    axes[0].legend(markerscale=3, fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].scatter(rest_v[:, 2], rest_v[:, 1], s=4, alpha=0.35, label="Rest mesh")
    axes[1].scatter(deformed_gt[:, 2], deformed_gt[:, 1], s=4, alpha=0.5, label="Target deformed")
    axes[1].scatter(deformed_pred[:, 2], deformed_pred[:, 1], s=4, alpha=0.6, marker="x", label="PINN predicted")
    axes[1].set_title("ZY Longitudinal Projection")
    axes[1].set_xlabel("Z (mm)")
    axes[1].set_ylabel("Y (mm)")
    axes[1].legend(markerscale=3, fontsize=8)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_dir / "deformation_comparison.png", dpi=200)
    plt.close()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    obj_path = Path("full_model.obj") if Path("full_model.obj").exists() else Path("data/full_model.obj")
    out_dir = Path("results")

    rest_v, edges = load_real_colon_mesh(str(obj_path), target_nodes=1024, k_neighbors=4)
    rest_v, edges = rest_v.to(device), edges.to(device)
    num_nodes = rest_v.shape[0]

    u_true = torch.zeros_like(rest_v)
    u_true[:, 1] = 0.02 * rest_v[:, 2]
    u_true = u_true.unsqueeze(0)

    visual_features = torch.randn(1, 32, device=device)
    model = ColonDeformationPINN(in_features=32, num_nodes=num_nodes).to(device)
    elastic_loss_fn = ColonElasticLoss(rest_v, edges)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    lambda_phys = 0.1

    history = {"step": [], "data_loss": [], "phys_loss": [], "tre_mm": []}

    for step in range(1, 81):
        optimizer.zero_grad()
        u_pred = model(visual_features)
        loss_data = F.mse_loss(u_pred, u_true)
        loss_phys = elastic_loss_fn(u_pred)
        total_loss = loss_data + lambda_phys * loss_phys
        total_loss.backward()
        optimizer.step()

        tre = torch.norm(u_pred - u_true, dim=-1).mean().item()
        history["step"].append(step)
        history["data_loss"].append(loss_data.item())
        history["phys_loss"].append(loss_phys.item())
        history["tre_mm"].append(tre)

        if step % 20 == 0 or step == 1:
            print(
                f"Step {step:2d} | Total: {total_loss.item():.5f} | "
                f"Data: {loss_data.item():.5f} | Phys: {loss_phys.item():.5f} | TRE: {tre:.4f} mm"
            )

    save_plots(
        history,
        rest_v.cpu().numpy(),
        u_true.squeeze(0).detach().cpu().numpy(),
        u_pred.squeeze(0).detach().cpu().numpy(),
        out_dir,
    )
    print(f"Saved plots to {out_dir.resolve()}")


if __name__ == "__main__":
    main()