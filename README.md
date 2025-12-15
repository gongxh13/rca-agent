# RCA Agent System

A DeepAgent-based root cause analysis system for distributed systems using specialized sub-agents for log, metric, and trace analysis.

## Architecture

The system consists of:

- **DeepAgent (Coordinator)**: Guides the RCA investigation without directly accessing data
- **Log Analysis Agent**: Analyzes application and system logs
- **Metric Analysis Agent**: Analyzes application performance and infrastructure metrics  
- **Trace Analysis Agent**: Analyzes distributed traces

## Installation

1. Install dependencies:

```bash
pip install -e .
```

Or install specific dependencies:

```bash
pip install langchain langchain-core langchain-anthropic deepagents pandas networkx
```

2. Set up your API key for the LLM provider (e.g., Anthropic):

```bash
export ANTHROPIC_API_KEY="your-api-key-here"
```

## Usage

### Basic Usage

```python
from langchain_anthropic import ChatAnthropic
from src.agents.rca_agents import create_rca_deep_agent

# Initialize the model
model = ChatAnthropic(
    model="claude-sonnet-4-20250514",
    temperature=0
)

# Create the RCA agent
rca_agent = create_rca_deep_agent(
    model=model,
    config={"dataset_path": "datasets/OpenRCA/Bank"}
)

# Ask a question (in UTC+8 timezone)
question = """
在2021年3月4日下午14:00到15:00之间（东8区时间），系统性能下降。
请帮我分析根本原因。

Translation: Between 14:00 and 15:00 on March 4, 2021 (UTC+8), 
system performance degraded. Please help analyze the root cause.
"""

# Run the investigation
result = rca_agent.invoke({"input": question})
print(result["output"])
```

### Running Examples

1. **Basic Test**:
```bash
python examples/test_rca_agent.py
```

2. **Multiple Scenarios**:
```bash
python examples/example_rca_scenario.py
```

3. **Flamegraph Analysis**:
```bash
python examples/example_flamegraph_analysis.py
```

## How It Works

### Investigation Workflow

1. **Problem Understanding**: Extract time range, affected services, and symptoms
2. **Initial Assessment**: Get high-level summaries from all sub-agents
3. **Hypothesis Formation**: Form hypotheses about potential root causes
4. **Targeted Investigation**: Direct sub-agents to investigate specific hypotheses
5. **Correlation Analysis**: Look for temporal correlations across data sources
6. **Root Cause Determination**: Synthesize findings and identify root cause
7. **Recommendations**: Provide actionable recommendations

### Timezone Handling

- User questions contain times in **UTC+8** (China Standard Time)
- The system automatically converts to ISO format for tool calls
- CSV data contains Unix timestamps (handled by tools)
- Results are presented with UTC+8 context for clarity

### Sub-Agent Capabilities

**Log Analysis Agent**:
- Find error patterns
- Detect log volume anomalies
- Analyze error frequency
- Find correlated events
- Query raw logs

**Metric Analysis Agent**:
- Analyze service performance
- Find slow services
- Detect resource usage anomalies
- Monitor component health
- Compare performance across time periods

**Trace Analysis Agent**:
- Find slow spans
- Analyze call chains
- Map service dependencies
- Detect latency anomalies
- Identify bottlenecks

**Flamegraph Analysis Agent**:
- Real-time CPU profiling with py-spy or perf
- Interactive flamegraph collection and analysis
- Automatic bottleneck identification
- Hierarchical function call analysis
- Performance optimization recommendations

## Example Scenarios

The system can handle various RCA scenarios:

1. **Latency Spikes**: Identify services with sudden response time increases
2. **Error Spikes**: Analyze sudden increases in error logs
3. **Resource Exhaustion**: Detect CPU, memory, or disk issues
4. **Cascading Failures**: Trace failure propagation through service dependencies
5. **CPU Performance Bottlenecks**: Analyze CPU flamegraphs to identify hot functions and optimization opportunities

## Flamegraph Analysis

The system provides comprehensive CPU flamegraph analysis capabilities for identifying performance bottlenecks in Python applications.

### Features

- **Real-time Profiling**: Collect CPU flamegraphs from running Python processes
- **Interactive Collection**: User-friendly interface for selecting processes and configuring collection parameters
- **AI-Powered Analysis**: Automatic identification of performance bottlenecks using LLM agents
- **Hierarchical View**: Analyze function call hierarchies and CPU usage distribution
- **Drill-down Analysis**: Deep dive into specific functions to understand their call chains

### Prerequisites

1. **Install py-spy** (for Python profiling):
```bash
pip install py-spy
```

2. **macOS Users**: py-spy requires root permissions to profile other processes. See [Troubleshooting](#troubleshooting) for details.

### Quick Start

#### 1. Start a Test Application

Use the provided simulation script to create a CPU-intensive workload:

```bash
# Run for 5 minutes (default)
python examples/simulate_cpu_bottleneck.py

# Or specify custom duration (e.g., 60 seconds)
python examples/simulate_cpu_bottleneck.py --duration 60
```

#### 2. Collect Flamegraph

Run the interactive flamegraph analysis tool:

```bash
python examples/example_flamegraph_analysis.py
```

The tool will guide you through:
- **Collection Mode**: Choose between analyzing existing files or collecting new flamegraphs
- **Process Selection**: Select the target Python process from a list
- **Collection Parameters**:
  - Output directory (default: `/tmp`)
  - Sampling rate (50-500 Hz, default: 100 Hz)
  - Collection mode (timed or manual stop)
  - Duration (if using timed mode)

#### 3. Analyze Flamegraph

After collection, the tool will:
- Automatically parse the SVG flamegraph
- Extract function-level performance data
- Provide AI-powered analysis of bottlenecks
- Answer questions about CPU usage patterns

### Collection Modes

#### Python Profiling (py-spy)

- **Use Case**: Profile Python applications
- **Requirements**: 
  - py-spy installed (`pip install py-spy`)
  - Root permissions on macOS (for profiling other processes)
- **Features**:
  - Low overhead sampling
  - Python-specific optimizations
  - Automatic SVG generation

#### Perf Profiling (Linux)

- **Use Case**: System-level profiling of any process
- **Requirements**: 
  - `perf` tool installed (usually pre-installed on Linux)
  - Root permissions
- **Features**:
  - System-wide profiling
  - Kernel and user-space analysis
  - Requires `flamegraph` tool for SVG generation

### Analysis Capabilities

The flamegraph analysis agent can:

1. **Overview Analysis**: Get a hierarchical view of CPU usage by function level
   - Identify top CPU-consuming functions
   - Understand call stack distribution
   - Find performance hotspots

2. **Drill-down Analysis**: Deep dive into specific functions
   - Analyze function call chains
   - Identify sub-functions contributing to CPU usage
   - Support exact and fuzzy matching

3. **Question Answering**: Ask natural language questions such as:
   - "What are the top CPU-consuming functions?"
   - "Why is function X slow?"
   - "What functions call Y and consume the most CPU?"
   - "How can I optimize this code?"

### Example Workflow

```bash
# Terminal 1: Start the test application
python examples/simulate_cpu_bottleneck.py --duration 300

# Terminal 2: Collect and analyze flamegraph
python examples/example_flamegraph_analysis.py

# Follow the interactive prompts:
# 1. Select "采集新文件" (Collect new file)
# 2. Choose "Python" profiling type
# 3. Select the simulate_cpu_bottleneck.py process
# 4. Configure collection parameters
# 5. Wait for collection to complete
# 6. Ask questions about the flamegraph
```

### Troubleshooting

#### macOS Permission Issues

On macOS, py-spy requires root permissions to profile other processes. The tool automatically detects this and provides solutions:

**Option 1: Use sudo with environment preservation (Recommended for conda)**
```bash
sudo -E python examples/example_flamegraph_analysis.py
```

**Option 2: Preserve PATH environment variable**
```bash
sudo env PATH=$PATH python examples/example_flamegraph_analysis.py
```

**Option 3: Profile current process**
- Don't specify a PID when prompted
- The tool will profile itself (useful for testing)

#### Process Not Found Errors

If you see "Failed to find python version from target process":
- Ensure the target process is a Python process
- Verify the process is still running
- Check that you have sufficient permissions
- Try profiling the current process instead

#### File Not Generated

If the flamegraph file is not generated:
- Check error messages in the console
- Verify output directory permissions
- Ensure sufficient disk space
- Try a shorter collection duration

## Configuration

You can customize the agent behavior by modifying:

- `src/agents/rca_config.py`: System prompts and workflow definitions
- Tool configurations in `create_rca_deep_agent()` function

## Dataset

The system uses the OpenRCA dataset structure:
```
datasets/OpenRCA/{scenario}/telemetry/{date}/{type}/{file}.csv
```

Supported data types:
- **Logs**: `log_service.csv`
- **Metrics**: `metric_app.csv`, `metric_container.csv`
- **Traces**: `trace_span.csv`

## Development

### Project Structure

```
RCAAgent/
├── src/
│   ├── agents/
│   │   ├── rca_agents.py         # RCA agent implementations
│   │   ├── rca_config.py         # RCA configuration and prompts
│   │   ├── flamegraph_agents.py  # Flamegraph analysis agent
│   │   └── flamegraph_config.py  # Flamegraph agent configuration
│   ├── tools/
│   │   ├── local_log_tool.py        # Log analysis tools
│   │   ├── local_metric_tool.py     # Metric analysis tools
│   │   ├── local_trace_tool.py      # Trace analysis tools
│   │   └── flamegraph_cpu_analyzer.py  # Flamegraph collection and analysis
│   └── utils/
│       └── streaming_output.py   # Streaming output handler
├── examples/
│   ├── test_rca_agent.py            # Basic RCA test
│   ├── example_rca_scenario.py      # Multiple RCA scenarios
│   ├── example_flamegraph_analysis.py  # Interactive flamegraph analysis
│   └── simulate_cpu_bottleneck.py  # CPU bottleneck simulation for testing
└── datasets/
    └── OpenRCA/                     # Dataset files
```

### Adding New Capabilities

To add new analysis capabilities:

1. Add methods to the appropriate tool class (`local_log_tool.py`, etc.)
2. Add corresponding LangChain Tool definitions in `rca_agents.py`
3. Update the agent prompt in `rca_config.py` to describe the new capability

## Troubleshooting

**Import Errors**: Make sure all dependencies are installed:
```bash
pip install -e .
```

**API Key Issues**: Ensure your LLM provider API key is set:
```bash
export ANTHROPIC_API_KEY="your-key"
```

**Data Not Found**: Verify the dataset path in your configuration matches your actual dataset location.

**Flamegraph Collection Issues**:
- **Permission denied**: On macOS, use `sudo -E` to preserve environment variables
- **Process not found**: Ensure target process is running and is a Python process
- **File not generated**: Check error messages, verify permissions, and ensure sufficient disk space
- **py-spy not found**: Install with `pip install py-spy`

## License

MIT
