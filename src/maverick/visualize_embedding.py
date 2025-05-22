import numpy as np
import torch
import matplotlib.pyplot as plt

# -----------------------
# Paths and Settings
# -----------------------
ENTITY2ID_PATH = "/home/azureuser/cloudfiles/code/Users/rvanemmerik1/Msc-AI-Thesis/maverick/data/Wikidata/knowledge_graphs/entity2id.txt"
EMBEDDINGS_PATH = "/home/azureuser/cloudfiles/code/Users/rvanemmerik1/Msc-AI-Thesis/maverick/data/Wikidata/embeddings/dimension_100/transe/entity2vec.bin"
KG_EMBEDDING_DIM = 100  # adjust if needed

# -----------------------
# Helper Functions
# -----------------------
def load_entity_to_index(entity2id_path):
    """
    Load the entity2id file and return a dictionary mapping entity strings to indices.
    Expects each line to be in the format: "entity_str id".
    """
    entity_to_index = {}
    with open(entity2id_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 2:
                continue
            entity, idx = parts[0], int(parts[1])
            entity_to_index[entity] = idx
    return entity_to_index

def load_embeddings(embeddings_file, embedding_shape):
    """
    Loads embeddings as a memory-mapped NumPy array.

    Parameters:
        embeddings_file (str): Path to the binary embeddings file.
        embedding_shape (tuple): Shape of the embeddings array (num_entities, embedding_dim).

    Returns:
        A memory-mapped NumPy array containing the embeddings.
    """
    return np.memmap(embeddings_file, dtype=np.float32, mode='r', shape=embedding_shape)

def get_embedding(entity_str, entity_to_index, embeddings, default_embedding=None):
    """
    Retrieves the embedding for a given entity string and returns it as a torch tensor.
    
    Parameters:
        entity_str (str): The entity's identifier.
        entity_to_index (dict): Mapping from entity strings to integer indices.
        embeddings (np.memmap): The memory-mapped embeddings array.
        default_embedding (torch.Tensor, optional): A default tensor if the entity is not found.
    
    Returns:
        torch.Tensor: Embedding tensor for the entity.
    """
    idx = entity_to_index.get(entity_str)
    if idx is None:
        return default_embedding if default_embedding is not None else torch.zeros(KG_EMBEDDING_DIM)
    emb_np = embeddings[idx]
    return torch.from_numpy(emb_np)

def visualize_full_embedding(embedding_vector, base_radius=0.1):
    """
    Visualizes all dimensions of the embedding vector around a smaller base circle.
    Each dimension is placed at an equal angle increment around the circle.

    Parameters:
        embedding_vector (torch.Tensor or np.ndarray): The full embedding vector (e.g., 100-D).
        base_radius (float): The radius of the 'base circle'. 
                             The final radius for each dimension is base_radius + embedding_value.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import torch

    # Convert to NumPy array if it's a torch tensor
    if isinstance(embedding_vector, torch.Tensor):
        embedding_vector = embedding_vector.numpy()

    # Use all dimensions
    numbers = embedding_vector
    N = len(numbers)

    # Angles for each dimension around the circle
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)

    # Radii = base circle + the embedding value at each dimension
    radii = base_radius + numbers
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)

    # Points on the base circle for reference
    circle_x = base_radius * np.cos(angles)
    circle_y = base_radius * np.sin(angles)

    # Plot
    plt.figure(figsize=(6, 6))
    ax = plt.gca()

    # Draw the base circle
    base_circle = plt.Circle((0, 0), base_radius, edgecolor='gray', facecolor='none')
    ax.add_artist(base_circle)

    # Plot the offset points
    ax.scatter(x, y, color='b')

    # Connect each offset point to the base circle with a dashed line
    for px, py, cx, cy in zip(x, y, circle_x, circle_y):
        ax.plot([px, cx], [py, cy], 'red', linestyle='--')

    # Annotate each point with its value
    for i, (xi, yi) in enumerate(zip(x, y)):
        ax.text(xi, yi, f'{numbers[i]:.4f}', fontsize=8,
                ha='right' if xi < 0 else 'left',
                va='bottom' if yi < 0 else 'top')

    # Dynamically set the plot limits based on data
    all_x = np.concatenate([x, circle_x])
    all_y = np.concatenate([y, circle_y])
    x_margin = (all_x.max() - all_x.min()) * 0.1
    y_margin = (all_y.max() - all_y.min()) * 0.1
    ax.set_xlim(all_x.min() - x_margin, all_x.max() + x_margin)
    ax.set_ylim(all_y.min() - y_margin, all_y.max() + y_margin)

    ax.set_aspect('equal')
    ax.grid(True)

    plt.title('Visualizing All Embedding Dimensions Around a Smaller Base Circle')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.savefig('embedding_visualization.png')

# -----------------------
# Main Routine
# -----------------------
def main():
    # Load the entity-to-index mapping
    entity_to_index = load_entity_to_index(ENTITY2ID_PATH)
    print(f"Loaded entity-to-index mapping with {len(entity_to_index)} entities.")
    
    # Define the expected shape of the embeddings array and load embeddings
    embedding_shape = (len(entity_to_index), KG_EMBEDDING_DIM)
    embeddings = load_embeddings(EMBEDDINGS_PATH, embedding_shape)
    print(f"Loaded embeddings with shape: {embeddings.shape}")
    
    # Define an example entity. Update this with any valid entity ID from your file.
    example_entity = "Q42"
    embedding_tensor = get_embedding(example_entity, entity_to_index, embeddings)
    print(f"Embedding for '{example_entity}': {embedding_tensor}")
    
    # Visualize a slice (first 4 dimensions) of the embedding
    visualize_full_embedding(embedding_tensor, base_radius=0.1)

if __name__ == "__main__":
    main()
