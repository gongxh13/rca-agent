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

## Example Scenarios

The system can handle various RCA scenarios:

1. **Latency Spikes**: Identify services with sudden response time increases
2. **Error Spikes**: Analyze sudden increases in error logs
3. **Resource Exhaustion**: Detect CPU, memory, or disk issues
4. **Cascading Failures**: Trace failure propagation through service dependencies

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
│   │   ├── rca_agents.py      # Agent implementations
│   │   └── rca_config.py      # Configuration and prompts
│   └── tools/
│       ├── local_log_tool.py     # Log analysis tools
│       ├── local_metric_tool.py  # Metric analysis tools
│       └── local_trace_tool.py   # Trace analysis tools
├── examples/
│   ├── test_rca_agent.py         # Basic test
│   └── example_rca_scenario.py   # Multiple scenarios
└── datasets/
    └── OpenRCA/                  # Dataset files
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

## License

MIT
