"""
Visualize service topology from GraphML file.

This script loads a GraphML file and visualizes the service dependency graph.
"""

import sys
from pathlib import Path
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib

# Try to use a better backend if available
try:
    matplotlib.use('TkAgg')  # Better for interactive display
except:
    pass  # Fall back to default

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def view_topology(graphml_path: str, output_image: str = None, layout: str = 'spring'):
    """
    Visualize service topology from GraphML file.
    
    Args:
        graphml_path: Path to GraphML file
        output_image: Optional path to save image (if None, will display interactively)
        layout: Layout algorithm ('spring', 'circular', 'hierarchical', 'kamada_kawai')
    """
    # Load graph
    print(f"Loading graph from {graphml_path}...")
    G = nx.read_graphml(graphml_path)
    
    print(f"Graph loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # Print graph info
    print("\nNodes:")
    for node in G.nodes():
        print(f"  - {node}")
    
    print("\nEdges (with weights):")
    for u, v, data in G.edges(data=True):
        weight = data.get('weight', 1)
        print(f"  {u} -> {v} (weight: {weight})")
    
    # Create visualization
    plt.figure(figsize=(14, 10))
    
    # Choose layout
    if layout == 'spring':
        pos = nx.spring_layout(G, k=2, iterations=50)
    elif layout == 'circular':
        pos = nx.circular_layout(G)
    elif layout == 'hierarchical':
        # Try to use hierarchical layout (requires graphviz)
        try:
            pos = nx.nx_agraph.graphviz_layout(G, prog='dot')
        except:
            print("Warning: graphviz not available, using spring layout instead")
            pos = nx.spring_layout(G, k=2, iterations=50)
    elif layout == 'kamada_kawai':
        pos = nx.kamada_kawai_layout(G)
    else:
        pos = nx.spring_layout(G)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                           node_size=2000, alpha=0.9)
    
    # Draw edges with weights
    edges = G.edges()
    weights = [G[u][v].get('weight', 1) for u, v in edges]
    
    # Normalize weights for line width (min 1, max 5)
    if weights:
        min_weight = min(weights)
        max_weight = max(weights)
        if max_weight > min_weight:
            widths = [1 + 4 * (w - min_weight) / (max_weight - min_weight) 
                     for w in weights]
        else:
            widths = [2] * len(weights)
    else:
        widths = [2] * len(edges)
    
    # Draw edges with arrows to show direction
    # Use larger arrow size and make sure arrows are visible
    nx.draw_networkx_edges(
        G, pos, 
        width=widths, 
        alpha=0.7, 
        edge_color='gray',
        arrows=True,           # Enable arrows
        arrowsize=30,          # Larger arrow size
        arrowstyle='->',       # Arrow style
        connectionstyle='arc3,rad=0.1',  # Slight curve for better visibility
        min_source_margin=15,  # Space from source node
        min_target_margin=15   # Space from target node
    )
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
    
    # Draw edge labels (weights) - position them closer to target node to show direction
    edge_labels = {(u, v): str(d.get('weight', 1)) for u, v, d in G.edges(data=True)}
    # Position labels closer to target node to indicate direction
    nx.draw_networkx_edge_labels(
        G, pos, edge_labels, 
        font_size=8,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7),
        label_pos=0.7  # Position label 70% along the edge (closer to target)
    )
    
    plt.title(f"Service Topology ({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)", 
              fontsize=16, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    
    if output_image:
        plt.savefig(output_image, dpi=300, bbox_inches='tight')
        print(f"\nSaved visualization to {output_image}")
    else:
        plt.show()
    
    return G


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize service topology from GraphML file')
    parser.add_argument('graphml_path', type=str, 
                       help='Path to GraphML file')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Output image path (if not specified, will display interactively)')
    parser.add_argument('--layout', '-l', type=str, default='spring',
                       choices=['spring', 'circular', 'hierarchical', 'kamada_kawai'],
                       help='Layout algorithm (default: spring)')
    
    args = parser.parse_args()
    
    view_topology(args.graphml_path, args.output, args.layout)


if __name__ == "__main__":
    # Default: visualize the output topology
    default_path = "output/causal_data/service_topology.graphml"
    
    if len(sys.argv) > 1:
        main()
    else:
        print("Usage examples:")
        print(f"  python {sys.argv[0]} {default_path}")
        print(f"  python {sys.argv[0]} {default_path} --output topology.png")
        print(f"  python {sys.argv[0]} {default_path} --layout circular")
        print()
        print("Trying default path...")
        if Path(default_path).exists():
            view_topology(default_path)
        else:
            print(f"Error: {default_path} not found")
            print("Please specify the path to your GraphML file")

