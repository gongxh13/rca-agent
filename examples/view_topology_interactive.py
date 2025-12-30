"""
Visualize service topology using Pyecharts (Interactive).

This script loads a GraphML file and creates an interactive HTML visualization using Pyecharts.
Supports dragging, zooming, and detailed tooltips.
"""

import sys
import os
import json
from pathlib import Path
import networkx as nx
from pyecharts import options as opts
from pyecharts.charts import Graph
from pyecharts.globals import ThemeType

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def view_topology_interactive(
    graphml_path: str, 
    output_html: str = None, 
    title: str = "Service Topology",
    layout: str = "force",  # 'force', 'circular', 'none'
    repulsion: int = 200,   # Repulsion force for 'force' layout
    edge_length: int = 150  # Edge length for 'force' layout
):
    """
    Visualize service topology using Pyecharts.
    
    Args:
        graphml_path: Path to GraphML file
        output_html: Output HTML file path
        title: Chart title
        layout: Layout type ('force', 'circular', 'none')
        repulsion: Repulsion strength for force layout
        edge_length: Edge length for force layout
    """
    if not os.path.exists(graphml_path):
        print(f"Error: File not found: {graphml_path}")
        return

    print(f"Loading graph from {graphml_path}...")
    try:
        G = nx.read_graphml(graphml_path)
    except Exception as e:
        print(f"Error reading GraphML: {e}")
        return

    print(f"Graph loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Prepare nodes for Pyecharts
    nodes = []
    categories = set()
    
    # Calculate degree for node sizing
    degrees = dict(G.degree())
    max_degree = max(degrees.values()) if degrees else 1
    min_degree = min(degrees.values()) if degrees else 0
    
    # Optional: Community detection for coloring (if networkx has algorithms)
    # Or just use node prefix as category
    node_categories = {}
    
    for node in G.nodes(data=True):
        node_id = node[0]
        data = node[1]
        
        # Determine category based on prefix (e.g., 'Tomcat', 'ServiceTest', 'docker')
        # This assumes typical OpenRCA naming conventions
        if '_' in node_id:
            category = node_id.split('_')[0]
        elif '-' in node_id:
            category = node_id.split('-')[0]
        else:
            # Try to group by common prefixes
            import re
            match = re.match(r"([a-zA-Z]+)", node_id)
            category = match.group(1) if match else "Other"
            
        categories.add(category)
        node_categories[node_id] = category
        
        # Calculate symbol size based on degree
        degree = degrees.get(node_id, 0)
        # Scale size between 10 and 50
        symbol_size = 10 + (degree - min_degree) / (max_degree - min_degree + 1) * 40
        
        # Tooltip content
        tooltip = f"ID: {node_id}<br>Degree: {degree}"
        for k, v in data.items():
            tooltip += f"<br>{k}: {v}"
            
        nodes.append(opts.GraphNode(
            name=node_id,
            symbol_size=symbol_size,
            category=category,
            value=degree, # Value used for tooltip
            label_opts=opts.LabelOpts(is_show=True, position="right"),
            # Ensure attributes are JSON serializable
            tooltip_opts=opts.TooltipOpts(formatter=tooltip)
        ))

    # Prepare categories list
    categories_list = [{"name": c} for c in sorted(list(categories))]
    
    # Prepare links
    links = []
    for u, v, data in G.edges(data=True):
        # Extract edge weight/info
        weight = data.get('weight', 1)
        p_value = data.get('p_value', None)
        lag = data.get('lag', None)
        
        tooltip = f"{u} -> {v}"
        if weight != 1:
            tooltip += f"<br>Weight: {weight}"
        if p_value is not None:
            tooltip += f"<br>p-value: {p_value}"
        if lag is not None:
            tooltip += f"<br>Lag: {lag}"
            
        # Line width based on weight or strength (if available)
        # Normalize roughly
        width = 1
        if 'weight' in data:
            try:
                w = float(data['weight'])
                width = 1 + min(w, 10) / 2  # Cap width
            except:
                pass
                
        links.append(opts.GraphLink(
            source=u,
            target=v,
            value=weight,
            linestyle_opts=opts.LineStyleOpts(
                width=width, 
                curve=0.2, # Curved edges looks better
                opacity=0.7
            ),
            label_opts=opts.LabelOpts(is_show=False) # Hide edge labels by default to reduce clutter
        ))

    # Create Graph chart
    c = (
        Graph(init_opts=opts.InitOpts(
            width="100%", 
            height="900px", 
            page_title=title,
            theme=ThemeType.LIGHT
        ))
        .add(
            "",
            nodes,
            links,
            categories=categories_list,
            layout=layout,
            is_rotate_label=True,
            linestyle_opts=opts.LineStyleOpts(color="source", curve=0.3),
            label_opts=opts.LabelOpts(position="right"),
            gravity=0.1,
            repulsion=repulsion,
            edge_length=edge_length,
            is_draggable=True,  # Enable dragging
            symbol="circle",
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(title=title),
            legend_opts=opts.LegendOpts(
                orient="vertical", pos_left="2%", pos_top="20%"
            ),
            toolbox_opts=opts.ToolboxOpts(is_show=True),
            tooltip_opts=opts.TooltipOpts(trigger="item"), # Use item trigger for richer tooltips
        )
    )

    if output_html is None:
        output_html = graphml_path.replace(".graphml", ".html")
        
    c.render(output_html)
    print(f"Visualization saved to {output_html}")
    
    # Try to open in browser
    import webbrowser
    try:
        webbrowser.open('file://' + os.path.abspath(output_html))
    except:
        print(f"Please open {output_html} in your browser.")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Visualize topology interactively with Pyecharts')
    parser.add_argument('graphml_path', type=str, nargs='?', 
                        default="output/causal_data/service_topology.graphml",
                        help='Path to GraphML file')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output HTML file path')
    parser.add_argument('--layout', '-l', type=str, default='force',
                        choices=['force', 'circular', 'none'],
                        help='Layout type')
    parser.add_argument('--repulsion', '-r', type=int, default=1000,
                        help='Repulsion force (for force layout)')
    
    args = parser.parse_args()
    
    view_topology_interactive(
        args.graphml_path, 
        args.output, 
        layout=args.layout,
        repulsion=args.repulsion
    )

if __name__ == "__main__":
    main()
