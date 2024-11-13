# Import necessary libraries

from Bio import SeqIO, AlignIO, Phylo
from Bio.Align.Applications import ClustalOmegaCommandline
from Bio.Phylo.TreeConstruction import DistanceCalculator
from Bio.Phylo.BaseTree import Clade, Tree
import numpy as np
import os
import matplotlib.pyplot as plt
import networkx as nx


# Step 1: Insertion of Sequences
file_paths = [
    "X:\\CNAM\\PROJECT\\whole_env_sequences\\hiv1env-a.fasta",
    "X:\\CNAM\\PROJECT\\whole_env_sequences\\hiv1env-b.fasta",
    "X:\\CNAM\\PROJECT\\whole_env_sequences\\hiv1env-c.fasta",
    "X:\\CNAM\\PROJECT\\whole_env_sequences\\hiv1env-d.fasta"

]
combined_fasta = "X:\\CNAM\\PROJECT\\combined_sequences.fasta"

# Initialize an empty list to store all sequences
all_sequences = []

# Parse each file and append the sequences to the list
for file_path in file_paths:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    sequences = list(SeqIO.parse(file_path, "fasta"))
    all_sequences.extend(sequences)

# Write all sequences to the combined FASTA file
with open(combined_fasta, "w") as output_handle:
    SeqIO.write(all_sequences, output_handle, "fasta")

print(f"Total sequences written to combined file: {len(all_sequences)}")

print(combined_fasta)

print('Sequences assembly is finished -----------------------------------------------------------------------------------')

# Step 2: Multiple Sequence Alignment
# Define the Clustal Omega command line
clustalomega_cline = ClustalOmegaCommandline(
    cmd="E:\\clustal-omega-1.2.2-win64\\clustalo.exe",
    infile=combined_fasta,
    outfile="X:\\CNAM\\PROJECT\\aligned_sequences.aln",
    verbose=True,
    auto=True,
    force=True  # Add the force option to overwrite existing files
)

# Run Clustal Omega alignment and capture the output and errors
stdout, stderr = clustalomega_cline()

# Print output and error information
print("Standard Output:", stdout)
print("Standard Error:", stderr)

# Check if alignment file was created successfully
aligned_file = "X:\\CNAM\\PROJECT\\aligned_sequences.aln"
if not os.path.exists(aligned_file):
    raise FileNotFoundError(f"The alignment file {aligned_file} was not created. Please check Clustal Omega command.")

# Read the alignment
alignment = AlignIO.read(aligned_file, "fasta")

# print(alignment)

print('ClustalO is finished -------------------------------------------------------------------------------------------')

# Step 3: Compute Distance Matrix ( a table that shows how different or similar sequences are from one another )

# The identity model calculates distances based on the proportion of identical positions between sequences.
calculator = DistanceCalculator('identity')

# computation of distance matrix
dm = calculator.get_distance(alignment)

# Convert the Bio.Phylo distance matrix to a numpy array (easier to manipulate)
def to_numpy_matrix(dm):
    matrix = np.zeros((len(dm), len(dm)))
    for i, key1 in enumerate(dm.names):
        for j, key2 in enumerate(dm.names):
            matrix[i, j] = dm[key1, key2]
    return matrix, dm.names

D, names = to_numpy_matrix(dm)

print('D-distance', D)
print('NAMES', names)


print('Distance Matrix is calculated -------------------------------------------------------------------------------------------')


# Step 4b: UPGMA Algorithm
class Node:
    def __init__(self, name, left=None, right=None, distance=0.0): # left and right are None - leaf nodes, dist=0 - no parent yet
        self.name = name
        self.left = left
        self.right = right
        self.distance = distance

    def __repr__(self):
        if self.left is None and self.right is None:
            return self.name
        return f"({self.left}, {self.right}):{self.distance:.2f}"

def upgma(dist_matrix, names):

    # -1- Initialization of Clusters
    clusters = [Node(name) for name in names] # a Node for each sequence

    print("+++++++++++++++++++++++")
    n = len(clusters)

    # -2- Loop for Merging Clusters
    while n > 1:
        # -3- Finding the closest pair of clusters
        min_dist = float('inf') # start @ very large to be updated
        x, y = -1, -1 # to be sure there is no valid pair in the beginning
        for i in range(len(dist_matrix)):
            for j in range(i + 1, len(dist_matrix)): # Start at i + 1 to avoid redundant comparisons
                if dist_matrix[i][j] < min_dist:
                    min_dist = dist_matrix[i][j]
                    x, y = i, j # Store the indices of the closest pair

        # -4- Merge clusters x and y
        new_cluster = Node(name=None, left=clusters[x], right=clusters[y], distance=min_dist / 2) # x and y are indices of the closest pair

        # -5- Remove the merged clusters and add the new cluster (updating the clusters list)
        clusters = [clusters[k] for k in range(len(clusters)) if k != x and k != y] + [new_cluster] # x and y are indices of the closest pair, so they are eliminated

        # -6- Update the distance matrix - creation of zeroes matrix
        new_dist_matrix = np.zeros((len(clusters), len(clusters)))

        # -7- Calculating New Distances and filling the matrix
        for i in range(len(clusters) - 1): # Loop over the old clusters
            for j in range(i + 1, len(clusters) - 1): # Only fill the upper triangular matrix (i.e., unique pairs)
                new_dist_matrix[i][j] = new_dist_matrix[j][i] = dist_matrix[min(i, x)][max(i, x)]

        for i in range(len(clusters) - 1):
            new_dist_matrix[i][-1] = new_dist_matrix[-1][i] = (dist_matrix[min(i, x)][max(i, x)] + dist_matrix[min(i, y)][max(i, y)]) / 2

        dist_matrix = new_dist_matrix
        n -= 1

    return clusters[0]

# Build the UPGMA tree
upgma_tree = upgma(D, names)

# Step 5b: Plot the UPGMA tree using matplotlib and networkx
def add_edges(graph, node, pos=None, x=0, y=0, layer=1):
    if pos is None:
        pos = {}
    pos[node] = (x, y)
    if node.left:
        graph.add_edge(node, node.left)
        l = layer - node.distance
        pos = add_edges(graph, node.left, pos=pos, x=x-1/layer, y=y-1, layer=layer+1)
    if node.right:
        graph.add_edge(node, node.right)
        l = layer - node.distance
        pos = add_edges(graph, node.right, pos=pos, x=x+1/layer, y=y-1, layer=layer+1)
    return pos

def plot_upgma_tree(tree, title="UPGMA Tree"):
    graph = nx.DiGraph()
    pos = add_edges(graph, tree)

    labels = {node: node.name if node.name else '' for node in graph.nodes()}

    plt.figure(figsize=(12, 8))
    nx.draw(graph, pos, labels=labels, with_labels=True, node_size=5000, node_color="red", font_size=10, font_weight="bold", arrows=False)
    plt.title(title)
    plt.show()

# Plot the UPGMA tree
plot_upgma_tree(upgma_tree)






# Import additional library for saving plots
import os

# Define the directory where you want to save the images
output_dir = "X:\\CNAM\\PROJECT\\trees"

# Create the directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Function to save the UPGMA tree
def save_upgma_tree(tree, filename="UPGMA_tree.png", title="UPGMA Tree"):
    graph = nx.DiGraph()
    pos = add_edges(graph, tree)

    labels = {node: node.name if node.name else '' for node in graph.nodes()}

    plt.figure(figsize=(12, 8))
    nx.draw(graph, pos, labels=labels, with_labels=True, node_size=5000, node_color="red", font_size=10, font_weight="bold", arrows=False)
    plt.title(title)
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path)
    plt.close()

# Save the UPGMA tree
save_upgma_tree(upgma_tree, filename="UPGMA_tree.png", title="UPGMA Treee")
