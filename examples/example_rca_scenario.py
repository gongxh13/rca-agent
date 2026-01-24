"""
Interactive RCA Scenario Analysis

This script provides an interactive interface for root cause analysis.
Users can input the time range and number of faults, and the system will
generate questions using fixed templates and perform analysis with streaming.

This version uses async streaming execution with multiple stream modes:
- messages: Stream LLM tokens as they are generated
- custom: Stream custom updates from tools
- updates: Stream agent progress updates
"""

import asyncio
import dotenv
import os
import sys
from datetime import datetime
import uuid
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

dotenv.load_dotenv()

from InquirerPy import inquirer
from InquirerPy.base.control import Choice
from langchain.messages import HumanMessage
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.prompt import Prompt, IntPrompt
from src.agents.rca_agents import create_rca_deep_agent
from src.utils.streaming_output import StreamingOutputHandler, stream_agent_execution
from src.tools.data_loader import create_data_loader
from scripts.fetch_logs import run_fetch_logs, DEFAULT_FILE_MAPPING
import pytz

async def fetch_logs_interactive(console: Console):
    """
    交互式拉取日志
    """
    console.print()
    console.print(Rule("[bold yellow]日志拉取配置 / Log Fetching[/bold yellow]", style="yellow"))
    
    # Check config
    from src.config.loader import load_config
    try:
        cfg = load_config()
        log_cfg = cfg.log_fetch
    except Exception:
        log_cfg = None

    # Default parameters
    mode = "local"
    local_dir = "."
    host = None
    port = 22
    user = None
    password = None
    remote_dir = "/tmp/fault_logs"
    files_map = DEFAULT_FILE_MAPPING
    date_str = datetime.now().strftime("%Y-%m-%d")

    if log_cfg:
        console.print(f"[dim]Using configuration from config.yaml (mode={log_cfg.mode})...[/dim]")
        mode = log_cfg.mode
        local_dir = log_cfg.local_source_dir
        host = log_cfg.remote_ip
        port = log_cfg.remote_port
        user = log_cfg.remote_username
        password = log_cfg.remote_password
        remote_dir = log_cfg.remote_log_dir
        if log_cfg.files:
            files_map = log_cfg.files
    else:
        # If no config, fallback to interactive selection
        mode = await inquirer.select(
            message="选择拉取模式 / Select Mode:",
            choices=[
                Choice(value="local", name="Local (Current Directory)"),
                Choice(value="remote", name="Remote (SSH)")
            ],
            default="local"
        ).execute_async()
        
        if mode == "local":
            local_dir = Prompt.ask("请输入本地日志目录 / Local Log Dir", default=".")
        else:
            host = Prompt.ask("Host IP")
            user = Prompt.ask("Username")
            password = Prompt.ask("Password", password=True)
            remote_dir = Prompt.ask("Remote Log Dir", default="/tmp/fault_logs")
    
    # Execute fetch function directly
    console.print("[dim]Executing fetch script directly...[/dim]")
    try:
        success = run_fetch_logs(
            mode=mode,
            local_dir=local_dir,
            host=host,
            port=port,
            user=user,
            password=password,
            remote_dir=remote_dir,
            date_str=date_str,
            files_map=files_map
        )
        if success:
            console.print("[bold green]✓ 日志拉取完成 / Logs fetched successfully[/bold green]")
        else:
            console.print("[bold red]日志拉取失败 / Fetch failed[/bold red]")
    except Exception as e:
        console.print(f"[bold red]日志拉取错误 / Fetch error:[/bold red]\n{e}")


def setup_agent(dataset_path: str = "datasets/OpenRCA/Bank", domain: str = "openrca"):
    """Initialize the RCA agent with Langfuse tracing."""
    try:
        from langfuse import get_client
        from langfuse.langchain import CallbackHandler
        
        from src.config.loader import load_config
        from src.utils.llm_factory import init_langchain_models_from_llm_config

        # Initialize Langfuse client
        langfuse = get_client()
        
        # Initialize Langfuse CallbackHandler for Langchain (tracing)
        langfuse_handler = CallbackHandler()
        
        # Load config and init model
        config_data = load_config()
        models, default_model = init_langchain_models_from_llm_config(config_data.llm)
        
        if not default_model:
            print("Error: No valid LLM model found in config")
            return None, None, None
            
        model = default_model
        
        # Config for the agent
        # We inject domain type and dataset path
        # Tools will use this to pick the right loader and prompt adapter
        config = {
            "dataset_path": dataset_path,
            "metric_analyzer": {"dataset_path": dataset_path, "domain_adapter": domain, "dataloader": domain},
            "log_analyzer": {"dataset_path": dataset_path, "domain_adapter": domain, "dataloader": domain},
            "trace_analyzer": {"dataset_path": dataset_path, "domain_adapter": domain, "dataloader": domain},
        }

        # Merge evaluation config
        if config_data.evaluation:
            config["evaluation"] = config_data.evaluation.model_dump()
        
        # Create a dataloader instance to get configuration (like timezone)
        dataloader = create_data_loader({"dataloader": domain, "dataset_path": dataset_path})
        
        rca_agent = create_rca_deep_agent(
            model=model,
            config=config
        )
        
        return rca_agent, langfuse_handler, dataloader
        
    except ImportError as e:
        if "langfuse" in str(e):
            print("Error: langfuse not installed")
            print("Install with: pip install langfuse")
        else:
            print(f"Error: {e}")
        return None, None, None
    except Exception as e:
        print(f"Error initializing agent: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


def generate_question(start_time: str, end_time: str, num_faults: int) -> str:
    """
    根据时间段和故障数量生成问题模板
    
    Args:
        start_time: 开始时间 (格式: YYYY-MM-DD HH:MM 或 YYYY-MM-DDTHH:MM)
        end_time: 结束时间 (格式: YYYY-MM-DD HH:MM 或 YYYY-MM-DDTHH:MM)
        num_faults: 故障数量 (1 或多个)
        
    Returns:
        生成的问题文本
    """
    # 格式化时间显示（移除 T 分隔符，统一格式）
    start_display = start_time.replace("T", " ")
    end_display = end_time.replace("T", " ")
    
    if num_faults == 1:
        # 单故障模板
        question = f"""
在{start_display}至{end_display}的时间范围内，系统中检测到一次故障。然而，目前尚不清楚故障的根本原因组件、根本原因发生的确切时间以及故障的根本原因。你的任务是确定根本原因组件、根本原因发生的具体时间以及根本原因。
        """
    else:
        # 多故障模板
        question = f"""
在{start_display}至{end_display}的指定时间段内，系统中检测到{num_faults}次故障。导致这些故障的具体组件、发生的确切时间以及根本原因尚不清楚。你的任务是确定故障的根本原因组件、根本原因发生的具体时间以及根本原因。
        """
    
    return question.strip()


def get_user_input(console: Console) -> tuple[str, str, int]:
    """
    获取用户输入：时间段和故障数量
    
    Args:
        console: Rich Console 实例
        
    Returns:
        (start_time, end_time, num_faults) 元组
    """
    console.print()
    console.print(Rule("[bold cyan]请输入故障分析参数[/bold cyan]", style="cyan"))
    console.print()
    
    # 提示时间格式
    console.print("[dim]时间格式示例: 2021-03-04 13:30 或 2021-03-04T13:30[/dim]")
    console.print()
    
    # 获取开始时间
    start_time = Prompt.ask(
        "[bold yellow]请输入开始时间[/bold yellow]",
        default="2021-03-04 13:30"
    )
    
    # 获取结束时间
    end_time = Prompt.ask(
        "[bold yellow]请输入结束时间[/bold yellow]",
        default="2021-03-04 14:00"
    )
    
    # 获取故障数量
    num_faults = IntPrompt.ask(
        "[bold yellow]请输入故障数量[/bold yellow]",
        default=1
    )
    
    # 验证故障数量
    if num_faults < 1:
        console.print("[bold red]故障数量必须大于0，已设置为1[/bold red]")
        num_faults = 1
    
    return start_time, end_time, num_faults


async def run_analysis(
    agent,
    langfuse_handler,
    console: Console,
    question: str,
    dataloader=None
):
    """
    执行根因分析
    
    Args:
        agent: RCA agent 实例
        langfuse_handler: Langfuse callback handler
        console: Rich Console 实例
        question: 用户输入的分析问题
        dataloader: 数据加载器实例 (用于获取时区配置)
    """
    # 显示问题
    console.print()
    console.print(Rule("[bold cyan]分析问题[/bold cyan]", style="cyan"))
    console.print()
    console.print(Panel(question, title="[bold yellow]问题[/bold yellow]", border_style="yellow"))
    console.print()
    
    try:
        # 获取当前时间
        current_time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        if dataloader:
            try:
                tz_name = dataloader.get_timezone()
                if tz_name:
                    tz = pytz.timezone(tz_name)
                    # Get current time in that timezone
                    current_time_str = datetime.now(tz).strftime('%Y-%m-%d %H:%M:%S')
            except Exception as e:
                console.print(f"[dim yellow]Warning: Failed to get timezone from dataloader: {e}[/dim yellow]")

        # 创建流式输出处理器
        handler = StreamingOutputHandler(console=console)
        
        # 异步流式执行
        result = await stream_agent_execution(
            agent=agent,
            input=dict(
                messages=[
                    HumanMessage(content=f"Current system time: {current_time_str}"),
                    HumanMessage(content=question)
                ]
            ),
            config={"callbacks": [langfuse_handler]},
            stream_modes=["messages", "custom", "updates"],
            handler=handler
        )
        
        # 显示最终结果
        console.print()
        console.print(Rule("[bold green]分析结果[/bold green]", style="green"))
        console.print()
        
        output = result or "No output"
        console.print(Panel(output, title="[bold green]最终结果[/bold green]", border_style="green"))
        
        save_dir = os.path.join(os.getcwd(), "outputs", "rca", "md")
        os.makedirs(save_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique = uuid.uuid4().hex[:6]
        filename = f"rca_result_{ts}_{unique}.md"
        save_path = os.path.join(save_dir, filename)
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(output if isinstance(output, str) else str(output))
        console.print(f"[bold green]✓ 结果已保存至[/bold green] {save_path}")
        
    except Exception as e:
        console.print(f"[bold red]Error: {e}[/bold red]")
        import traceback
        console.print(traceback.format_exc())


async def main():
    """Interactive RCA analysis with user input."""
    console = Console()
    
    console.print()
    console.print(Rule("[bold cyan]RCA Agent - 交互式故障分析[/bold cyan]", style="cyan"))
    console.print()

    # Domain Selection
    console.print("[bold yellow]请选择分析场景 / Select Analysis Domain:[/bold yellow]")
    
    domain_choice = await inquirer.select(
        message="请选择分析场景 / Select Analysis Domain:",
        choices=[
            Choice(value="1", name="1. OpenRCA (Microservices/Bank)"),
            Choice(value="2", name="2. Disk Fault (System Logs)")
        ],
        default="1"
    ).execute_async()
    
    if domain_choice == "1":
        domain = "openrca"
        dataset_path = "datasets/OpenRCA/Bank"
    else:
        domain = "disk_fault"
        dataset_path = "datasets/disk_fault_logs"

    # Initialize agent
    console.print()
    console.print(Rule("[bold yellow]初始化 RCA Agent...[/bold yellow]", style="yellow"))
    console.print(f"Domain: {domain}")
    console.print(f"Dataset: {dataset_path}")
    console.print()
    
    agent, langfuse_handler, dataloader = setup_agent(dataset_path=dataset_path, domain=domain)
    if agent is None or langfuse_handler is None:
        console.print("[bold red]初始化失败，退出。[/bold red]")
        return
    
    console.print("[bold green]✓ Agent 初始化成功[/bold green]")
    console.print("[bold green]✓ Langfuse tracing 已启用[/bold green]")
    console.print("[bold green]✓ 流式输出模式已启用[/bold green]")
    
    # 主循环：允许用户多次分析
    while True:
        try:
            # 获取用户输入
            if domain == "disk_fault":
                # 磁盘故障模式：直接输入问题
                console.print()
                console.print(Rule("[bold cyan]请输入分析问题[/bold cyan]", style="cyan"))
                console.print()
                console.print("[dim]例如：请分析2026-01-23 10:00到10:30期间app应用无法使用[/dim]")
                
                question = Prompt.ask(
                    "[bold yellow]请输入您的问题[/bold yellow]",
                )
                
                if not question.strip():
                    console.print("[bold red]问题不能为空，请重新输入[/bold red]")
                    continue
                await fetch_logs_interactive(console)
                await run_analysis(agent, langfuse_handler, console, question, dataloader=dataloader)
            else:
                # OpenRCA 模式：保持原有的时间段输入
                start_time, end_time, num_faults = get_user_input(console)
                question = generate_question(start_time, end_time, num_faults)
                await run_analysis(agent, langfuse_handler, console, question, dataloader=dataloader)
                
            # 是否继续
            continue_analysis = await inquirer.confirm(
                message="是否继续分析其他故障？/ Continue analysis?",
                default=False
            ).execute_async()
            
            if not continue_analysis:
                console.print("[bold green]分析结束，再见！[/bold green]")
                break
                
        except KeyboardInterrupt:
            console.print("\n[bold yellow]用户取消操作[/bold yellow]")
            break
        except Exception as e:
            console.print(f"[bold red]发生错误: {e}[/bold red]")
            import traceback
            console.print(traceback.format_exc())
            break
    
    console.print()
    console.print(Rule("[bold green]分析完成[/bold green]", style="green"))
    console.print()
    console.print("[bold yellow]提示: 请访问 Langfuse 控制台查看详细的执行追踪信息[/bold yellow]")


if __name__ == "__main__":
    asyncio.run(main())
