from flask import Flask, request, send_file, redirect, jsonify, render_template, url_for
from Bio import SeqIO, AlignIO, Phylo
from Bio.Align.Applications import ClustalOmegaCommandline
from Bio.Phylo.TreeConstruction import DistanceCalculator
from Bio.Phylo.BaseTree import Clade, Tree
import numpy as np
import os
import io
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib
from datetime import datetime
matplotlib.use('Agg')

app = Flask(__name__)

# Define directory for saving output images
OUTPUT_DIR = os.path.join(os.getcwd(), 'static')
os.makedirs(OUTPUT_DIR, exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if the request contains files
        if 'files' not in request.files:
            return "No files found in request", 400

        files = request.files.getlist('files')
        if not files:
            return "No files uploaded", 400

        # Initialize an empty list to store all sequences
        all_sequences = []

        # Process each uploaded file
        for file in files:
            if file and file.filename.endswith('.fasta'):
                # Convert the file stream to a text mode stream
                file_stream = io.StringIO(file.stream.read().decode('utf-8'))
                sequences = list(SeqIO.parse(file_stream, "fasta"))
                all_sequences.extend(sequences)

        # Use secure filename and save the combined FASTA file
        combined_fasta_path = os.path.join(OUTPUT_DIR, "combined_sequences.fasta")
        with open(combined_fasta_path, "w") as output_handle:
            SeqIO.write(all_sequences, output_handle, "fasta")

        # Perform sequence alignment
        aligned_fasta_path = os.path.join(OUTPUT_DIR, "aligned_sequences.aln")
        clustalomega_cline = ClustalOmegaCommandline(
            cmd="E:\\clustal-omega-1.2.2-win64\\clustalo.exe",
            infile=combined_fasta_path,
            outfile=aligned_fasta_path,
            verbose=True,
            auto=True,
            force=True
        )
        stdout, stderr = clustalomega_cline()
        print("Standard Output:", stdout)
        print("Standard Error:", stderr)

        if not os.path.exists(aligned_fasta_path):
            return "Alignment failed. Please check Clustal Omega command.", 500

        # Read the alignment
        alignment = AlignIO.read(aligned_fasta_path, "fasta")

        # Compute distance matrix
        calculator = DistanceCalculator('identity')
        dm = calculator.get_distance(alignment)

        # Convert to numpy matrix
        def to_numpy_matrix(dm):
            matrix = np.zeros((len(dm), len(dm)))
            for i, key1 in enumerate(dm.names):
                for j, key2 in enumerate(dm.names):
                    matrix[i, j] = dm[key1, key2]
            return matrix, dm.names

        D, names = to_numpy_matrix(dm)

        # Neighbor Joining Algorithm
        def neighbor_joining(D, names):
            n = len(D)
            nodes = {i: Clade(name=names[i]) for i in range(n)}
            all_clades = {i: nodes[i] for i in range(n)}
            remaining_indices = list(range(n))  # Track the current indices of the remaining clades

            while n > 1:
                # Debug: Check the number of sequences
                print(f"Current number of sequences (n): {n}")

                # Compute r(i) - net divergence for each sequence
                r = np.sum(D, axis=1)  # This gives the sum of distances for each sequence

                # Debug: Check the net divergence
                print("Net Divergence (r):", r)

                # Compute Mij using the exact formula
                M = np.zeros((n, n))
                for i in range(n):
                    for j in range(n):
                        if i != j:
                            M[i, j] = D[i, j] - (r[i] + r[j]) / (n - 2) if n > 2 else D[i, j]

                # Debug: Log the M matrix
                print("M-Matrix (exact formula):")
                print(M)

                # Find pair (i, j) with the smallest M[i, j]
                i, j = np.unravel_index(np.argmin(M + np.diag([np.inf] * n)), M.shape)

                # Log the pairs being joined
                print(f"Joining Clades: {names[remaining_indices[i]]} and {names[remaining_indices[j]]}")

                # Calculate the branch lengths
                branch_length_i = (D[i, j] + (r[i] - r[j]) / (n - 2)) / 2
                branch_length_j = D[i, j] - branch_length_i

                # Create new clade
                new_clade = Clade()
                new_clade.clades.append(all_clades[remaining_indices[i]])
                new_clade.clades.append(all_clades[remaining_indices[j]])

                # Set the branch lengths appropriately
                all_clades[remaining_indices[i]].branch_length = branch_length_i if branch_length_i > 0 else 0.001
                all_clades[remaining_indices[j]].branch_length = branch_length_j if branch_length_j > 0 else 0.001

                # Log the new clade and its branch lengths
                print(f"New Clade: {new_clade}, Branch Lengths: {branch_length_i}, {branch_length_j}")

                # Update distance matrix before removing i and j
                new_row = (D[i, :] + D[j, :] - D[i, j]) / 2

                # Remove rows and columns for clades i and j from D
                D = np.delete(D, [i, j], axis=0)
                D = np.delete(D, [i, j], axis=1)

                # Create the new row and append it to the distance matrix
                new_row = np.delete(new_row, [i, j])
                D = np.vstack((D, new_row))
                new_col = np.append(new_row, 0).reshape(-1, 1)
                D = np.hstack((D, new_col))

                # Update the list of remaining indices
                remaining_indices.pop(max(i, j))  # Remove the higher index first
                remaining_indices.pop(min(i, j))

                # Add the new clade to the remaining indices
                remaining_indices.append(n - 2)

                # Update nodes
                all_clades[n - 2] = new_clade

                n -= 1
                print(f"N is now {n}")

            # Handle the remaining clades
            remaining_clades = list(all_clades.values())
            root = Tree(root=Clade())
            root.root.clades.extend(remaining_clades)

            # Ensure that the root's branch length is set
            for clade in root.root.clades:
                if clade.branch_length == 0:
                    clade.branch_length = 0.001  # Avoid zero length for branches

            return root

        nj_tree = neighbor_joining(D, names)

        # UPGMA Algorithm
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

        # Function to save and plot trees
        # Function to save and plot trees
        def save_phylo_tree(tree, filename="NJ_tree.png", title="Phylogenetic Tree"):
            fig = plt.figure(figsize=(10, 10))
            axes = fig.add_subplot(1, 1, 1)
            Phylo.draw(tree, do_show=False, axes=axes)
            plt.title(title)

            # Save image in the OUTPUT_DIR
            output_path = os.path.join(OUTPUT_DIR, filename)
            plt.savefig(output_path)
            plt.close(fig)

        def save_upgma_tree(tree, filename="UPGMA_tree.png", title="UPGMA Tree"):
            graph = nx.DiGraph()
            pos = add_edges(graph, tree)

            labels = {node: node.name if node.name else '' for node in graph.nodes()}

            plt.figure(figsize=(12, 8))
            nx.draw(graph, pos, labels=labels, with_labels=True, node_size=5000, node_color="red", font_size=10, font_weight="bold", arrows=False)
            plt.title(title)

            # Save image in the OUTPUT_DIR
            output_path = os.path.join(OUTPUT_DIR, filename)
            plt.savefig(output_path)
            plt.close()

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

        # Render the template with image URLs for display
        nj_tree_path = 'NJ_tree.png'  # Image saved in the OUTPUT_DIR
        upgma_tree_path = 'UPGMA_tree.png'  # Image saved in the OUTPUT_DIR

        # After saving the trees to OUTPUT_DIR, use static folder for serving
        save_phylo_tree(nj_tree, filename=nj_tree_path, title="Neighbor Joining Tree")
        save_upgma_tree(upgma_tree, filename=upgma_tree_path, title="UPGMA Tree")

        timestamp = datetime.now().timestamp()

        # Use url_for to refer to static images
        return render_template('display_trees.html',
                       nj_tree_url=url_for('static', filename='NJ_tree.png'),
                       upgma_tree_url=url_for('static', filename='UPGMA_tree.png'),
                       timestamp=timestamp)

    return '''
    <!doctype html>
    <title>Upload FASTA files</title>
    <center>
        <h1>Upload up to 10 FASTA files</h1>
            <form action="" method="post" enctype="multipart/form-data">
                <input type="file" name="files" accept=".fasta" multiple>
                <input type="submit" value="MAKE TREES">
            </form>
    </center>
    '''

if __name__ == '__main__':
    app.run(debug=True)
