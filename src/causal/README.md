# Causal Analysis Module

This module provides functionality for causal graph construction and root cause inference using DoWhy and causal-learn libraries.

## Modules

### 1. `data_preprocessor.py` - CausalDataPreprocessor

Handles data preprocessing for causal analysis:
- Multi-day data integration
- Metric aggregation with configurable time granularity
- Feature engineering (wide table format)
- Service topology construction from traces
- Data alignment and missing data handling

**Key Features:**
- Extracts core metrics (CPU, memory, disk, network, JVM)
- Filters candidate components
- Creates wide table format for causal analysis
- Builds service dependency topology from traces

### 2. `causal_discovery.py` - CausalGraphBuilder

Builds causal graphs using causal discovery algorithms:
- Uses PC algorithm from causal-learn
- Supports background knowledge from service topology
- Graph optimization and validation
- Export to multiple formats (GraphML, GML, JSON, CSV)

**Key Features:**
- PC algorithm for causal discovery
- Optional trace-based prior knowledge
- Graph optimization (removes unreasonable edges)
- Statistics and visualization support

## Usage

### Step 1: Prepare Data

```python
from src.causal.data_preprocessor import CausalDataPreprocessor

# Initialize preprocessor
preprocessor = CausalDataPreprocessor(
    dataset_path="datasets/OpenRCA/Bank",
    time_granularity="5min"  # 5 minutes for causal graph construction
)

# Prepare data
results = preprocessor.prepare_causal_data(
    start_date="2021-03-04",
    end_date="2021-03-25",
    include_app_metrics=True
)

# Save results
preprocessor.save_results(results, "output/causal_data")
```

### Step 2: Build Causal Graph

```python
from src.causal.causal_discovery import CausalGraphBuilder
import pandas as pd
import networkx as nx

# Load preprocessed data
wide_table = pd.read_csv("output/causal_data/all_data.csv", index_col=0, parse_dates=True)
service_topology = nx.read_graphml("output/causal_data/service_topology.graphml")

# Initialize builder
builder = CausalGraphBuilder(
    alpha=0.05,  # Significance level
    use_trace_prior=True,  # Use trace topology as prior knowledge
    verbose=True
)

# Build causal graph
causal_graph = builder.build_causal_graph(
    data=wide_table,
    service_topology=service_topology
)

# Optimize graph
causal_graph = builder.optimize_graph(
    graph=causal_graph,
    service_topology=service_topology
)

# Save graph
builder.save_graph(causal_graph, "output/causal_graph/causal_graph.graphml")
builder.save_edges_csv(causal_graph, "output/causal_graph/causal_edges.csv")
```

## Example Scripts

### `examples/prepare_causal_data.py`

Prepares OpenRCA data for causal analysis:
- Loads multi-day data
- Extracts core metrics
- Creates wide table format
- Builds service topology
- Saves results to output directory

**Run:**
```bash
python examples/prepare_causal_data.py
```

### `examples/build_causal_graph.py`

Builds causal graph from preprocessed data:
- Loads preprocessed data
- Runs PC algorithm
- Optimizes graph
- Saves causal graph and statistics

**Run:**
```bash
python examples/build_causal_graph.py
```

## Output Files

### Data Preprocessing Outputs

- `all_data.csv`: Wide table with all metrics (for causal graph construction)
- `service_topology.graphml`: Service dependency graph from traces
- `data_alignment_report.json`: Data alignment report
- `data_quality_report.json`: Data quality report

### Causal Graph Outputs

- `causal_graph.graphml`: Causal graph in GraphML format
- `causal_edges.csv`: List of causal edges (source -> target)
- `graph_statistics.json`: Graph statistics (nodes, edges, density, etc.)

## Configuration

### Time Granularity

- **Causal graph construction**: 5-10 minutes (recommended: 5min)
- **Root cause inference**: 1 minute (for precise anomaly detection)

### Timezone Handling

**Important**: For causal graph construction, **timezone conversion is not needed**.

- All timestamps are kept in UTC (as in the original dataset)
- Timezone doesn't matter for causal analysis, only relative time matters
- The datetime column is in UTC timezone, which is sufficient for causal graph construction

### Core Metrics

The preprocessor automatically extracts:
- CPU usage
- Memory usage
- Disk I/O
- Disk space
- JVM metrics
- Network metrics

### Candidate Components

Only these components are analyzed:
- `apache01`, `apache02`
- `Tomcat01-04`
- `Mysql01-02`
- `Redis01-02`
- `MG01-02`, `IG01-02`

## Dependencies

Required packages (already in `pyproject.toml`):
- `causallearn >= 0.1.0`
- `dowhy >= 0.11`
- `scikit-learn >= 1.3`
- `scipy >= 1.11`
- `pandas >= 2.0`
- `networkx >= 3.0`

Install with:
```bash
poetry install
# or
pip install causallearn dowhy scikit-learn scipy
```

