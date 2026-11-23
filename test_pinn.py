import torch
import torch.nn as nn
import torch.nn.functional as F
import trimesh
import numpy as np

# Set deterministic seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# ========================================================
# 1. LOAD AND SUBSAMPLE THE REAL 3D COLON MESH
# ========================================================
def load_real_colon_mesh(obj_path="full_model.obj", target_nodes=1024, k_neighbors=4):
    """
    Loads the real OBJ file and subsamples the 1.2M points down to target_nodes.
    Constructs an elastic graph using k-nearest neighbors without external dependencies.
    """
    print(f"-> Loading real colon mesh from: {obj_path} ...")
    mesh = trimesh.load(obj_path, process=False)
    
    all_vertices = np.array(mesh.vertices, dtype=np.float32)
    total_nodes = all_vertices.shape[0]
    print(f"   Original mesh has {total_nodes:,} vertices.")

    # Uniform index downsampling to keep processing fast and lightweight
    step = max(1, total_nodes // target_nodes)
    sampled_indices = np.arange(0, total_nodes, step)[:target_nodes]
    vertices = all_vertices[sampled_indices]
    
    # Center vertices around origin
    vertices -= vertices.mean(axis=0)
    num_sampled = vertices.shape[0]

    # Build k-nearest neighbor edges for the elastic regularizer
    print(f"   Building elastic topology for {num_sampled} nodes...")
    v_tensor = torch.from_numpy(vertices)
    dist_matrix = torch.cdist(v_tensor, v_tensor)
    
    # Exclude self-connections
    dist_matrix.fill_diagonal_(float('inf'))
    
    # Find nearest neighbors for each node
    _, neighbor_indices = torch.topk(dist_matrix, k=k_neighbors, largest=False)
    
    edges_set = set()
    for src in range(num_sampled):
        for dst in neighbor_indices[src].tolist():
            edge = tuple(sorted((src, dst)))
            edges_set.add(edge)
            
    edges = np.array(list(edges_set), dtype=np.int64)
    print(f"   Topology built successfully: {num_sampled} nodes and {len(edges)} elastic edges.")

    return torch.tensor(vertices, dtype=torch.float32), torch.tensor(edges, dtype=torch.long)


# ========================================================
# 2. BIOMECHANICAL ELASTIC REGULARIZER (PINN LOSS)
# ========================================================
class ColonElasticLoss(nn.Module):
    """
    Penalizes deviations from the rest edge lengths based on linear elasticity.
    """
    def __init__(self, rest_vertices: torch.Tensor, edges: torch.Tensor):
        super().__init__()
        self.register_buffer("rest_vertices", rest_vertices.float())
        self.register_buffer("edges", edges.long())

        idx1, idx2 = self.edges[:, 0], self.edges[:, 1]
        v1, v2 = self.rest_vertices[idx1], self.rest_vertices[idx2]
        
        # Calculate initial resting lengths (L_0)
        rest_lengths = torch.sqrt(torch.sum((v1 - v2) ** 2, dim=-1) + 1e-8)
        self.register_buffer("rest_lengths", rest_lengths)

    def forward(self, displacements: torch.Tensor) -> torch.Tensor:
        # displacements: [Batch, N, 3]
        deformed_v = self.rest_vertices.unsqueeze(0) + displacements
        idx1, idx2 = self.edges[:, 0], self.edges[:, 1]
        v1, v2 = deformed_v[:, idx1, :], deformed_v[:, idx2, :]

        # Current length under predicted deformation
        current_lengths = torch.sqrt(torch.sum((v1 - v2) ** 2, dim=-1) + 1e-8)
        
        # Mean squared edge deformation energy (strain penalty)
        strain = F.mse_loss(current_lengths, self.rest_lengths.unsqueeze(0).expand_as(current_lengths))
        return strain


# ========================================================
# 3. NEURAL NETWORK ARCHITECTURE
# ========================================================
class ColonDeformationPINN(nn.Module):
    """
    Multi-layer perceptron regressing 3D displacements (dx, dy, dz) per node.
    """
    def __init__(self, in_features=32, num_nodes=1024):
        super().__init__()
        self.num_nodes = num_nodes
        self.net = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, num_nodes * 3)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        return out.view(-1, self.num_nodes, 3)


# ========================================================
# 4. TRAINING LOOP AND VERIFICATION
# ========================================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Active device: {device}")

    try:
        rest_v, edges = load_real_colon_mesh("full_model.obj", target_nodes=1024, k_neighbors=4)
    except Exception as e:
        print(f"Error loading mesh: {e}")
        return

    rest_v = rest_v.to(device)
    edges = edges.to(device)
    num_nodes = rest_v.shape[0]

    # Synthetic Ground Truth deformation (simulating peristalsis/bending)
    u_true = torch.zeros_like(rest_v)
    u_true[:, 1] = 0.02 * rest_v[:, 2] # displacement scaled along the colon's longitudinal axis
    u_true = u_true.unsqueeze(0).to(device)

    # Simulated visual feature embedding from the intraoperative camera
    visual_features = torch.randn(1, 32, device=device)

    # Initialize model, loss function, and optimizer
    model = ColonDeformationPINN(in_features=32, num_nodes=num_nodes).to(device)
    elastic_loss_fn = ColonElasticLoss(rest_v, edges)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)

    lambda_phys = 0.1 # Weight of the physical regularizer

    print("\nStarting PINN optimization on real colon anatomy...")
    for step in range(1, 81):
        optimizer.zero_grad()
        
        # Forward pass
        u_pred = model(visual_features)

        # Supervised data loss + Physics-informed constraint
        loss_data = F.mse_loss(u_pred, u_true)
        loss_phys = elastic_loss_fn(u_pred)
        total_loss = loss_data + lambda_phys * loss_phys

        # Backward pass
        total_loss.backward()
        optimizer.step()

        if step % 20 == 0 or step == 1:
            # Target Registration Error (TRE) in mm
            tre = torch.norm(u_pred - u_true, dim=-1).mean().item()
            print(f"Step {step:2d} | Total Loss: {total_loss.item():.5f} | "
                  f"Data Loss: {loss_data.item():.5f} | Phys Loss: {loss_phys.item():.5f} | TRE: {tre:.4f} mm")

    print("\n" + "="*55)
    print("SUCCESS: PINN optimization completed on real C3VD geometry!")
    print("="*55)

if __name__ == "__main__":
    main()