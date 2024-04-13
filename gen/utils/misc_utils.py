import torch

def compute_centroids(masks, normalize: bool = False):
    masks = masks.float()  # Convert masks to float for calculation
    B, H, W = masks.shape  # Get dimensions of the masks
    
    # Create mesh grids for x and y coordinates
    grid_x, grid_y = torch.meshgrid(torch.arange(W, device=masks.device), torch.arange(H, device=masks.device), indexing='xy')
    
    # Calculate the sum of masks along the height and width dimensions
    total = masks.view(B, -1).sum(dim=1)
    
    # Handle cases where the total is zero to avoid division by zero
    total = total.where(total != 0, torch.ones_like(total))
    
    # Compute the centroids
    centroids_x = (masks * grid_x).view(B, -1).sum(dim=1) / total
    centroids_y = (masks * grid_y).view(B, -1).sum(dim=1) / total
    
    # Stack centroids coordinates for each batch
    centroids = torch.stack((centroids_x, centroids_y), dim=1)
    
    if normalize:
        centroids = (centroids / (masks.size(1) - 1)) * 2 - 1

    return centroids