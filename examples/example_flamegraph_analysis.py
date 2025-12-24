"""
Interactive Flamegraph Analysis

This script provides an interactive interface for analyzing CPU flamegraph files.
Users can input the flamegraph file path and ask questions about performance bottlenecks.

This version uses async streaming execution with multiple stream modes:
- messages: Stream LLM tokens as they are generated
- custom: Stream custom updates from tools
- updates: Stream agent progress updates
"""

import asyncio
import dotenv
import os
import time
import subprocess
import platform
import logging
from pathlib import Path
from typing import Optional, List, Dict
import threading
from concurrent.futures import ThreadPoolExecutor

from langchain_core.tools import StructuredTool

dotenv.load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 创建一个线程池执行器用于运行同步的 InquirerPy
_executor = ThreadPoolExecutor(max_workers=1)


def run_inquirer_sync(prompt_func):
    """
    在单独的线程中运行 InquirerPy prompt，避免事件循环冲突
    
    Args:
        prompt_func: 返回 InquirerPy prompt 的函数
        
    Returns:
        prompt 的执行结果
    """
    def _run_in_thread():
        # 在新线程中创建新的事件循环
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return prompt_func().execute()
        finally:
            loop.close()
    
    # 在线程池中执行
    future = _executor.submit(_run_in_thread)
    return future.result()

from langchain.messages import HumanMessage
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from InquirerPy import inquirer
from InquirerPy.base.control import Choice
from src.agents.flamegraph_agents import (
    create_flamegraph_analysis_agent,
    create_flamegraph_auto_profiling_agent,
)
from src.utils.streaming_output import StreamingOutputHandler, stream_agent_execution
from src.tools.flamegraph_cpu_analyzer import FlamegraphProfilingTool

# 创建火焰图采集工具实例
_profiling_tool = FlamegraphProfilingTool()


def setup_agent():
    """Initialize the flamegraph analysis agent with Langfuse tracing."""
    try:
        from langchain_deepseek import ChatDeepSeek
        from langfuse import get_client
        from langfuse.langchain import CallbackHandler
        
        # Initialize Langfuse client
        langfuse = get_client()
        
        # Initialize Langfuse CallbackHandler for Langchain (tracing)
        langfuse_handler = CallbackHandler()
        
        model = ChatDeepSeek(
            model="deepseek-chat"
        )
        
        # Default: analysis-only agent. Some modes will create auto-profiling agent dynamically.
        flamegraph_agent = create_flamegraph_analysis_agent(model=model, config={})
        
        return flamegraph_agent, langfuse_handler
        
    except ImportError as e:
        if "langfuse" in str(e):
            print("Error: langfuse not installed")
            print("Install with: pip install langfuse")
        else:
            print("Error: langchain-deepseek not installed")
            print("Install with: pip install langchain-deepseek")
        return None, None
    except Exception as e:
        print(f"Error initializing agent: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def get_running_processes(profiling_type: str) -> List[Dict[str, str]]:
    """
    获取当前运行的进程列表
    
    Args:
        profiling_type: 采集类型，'python' 或 'perf'
        
    Returns:
        进程列表，每个进程包含 pid, name, cmdline 等信息
    """
    processes = []
    
    try:
        if platform.system() == "Linux" or platform.system() == "Darwin":  # Linux or macOS
            # 使用 ps 命令获取进程列表
            if profiling_type == 'python':
                # 只获取Python进程
                # 使用 ps aux 获取所有进程，然后过滤Python进程
                cmd = ['ps', 'aux']
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
                
                if result.returncode == 0:
                    lines = result.stdout.strip().split('\n')
                    # 跳过标题行
                    for line in lines[1:]:
                        parts = line.split(None, 10)  # 最多分割10次
                        if len(parts) >= 11:
                            pid = parts[1]
                            if not pid.isdigit():
                                continue
                            
                            cmdline = parts[10] if len(parts) > 10 else ' '.join(parts[2:])
                            
                            # Python进程检测逻辑：
                            # 检查命令行中是否包含 'python'（不区分大小写）
                            # 这样可以检测到：
                            # - python app.py
                            # - python3 app.py  
                            # - /usr/bin/python3 app.py
                            # - python -m flask run
                            # - python -m gunicorn app:app
                            # - /usr/bin/python3 -m flask run
                            # 注意：即使使用 flask run 或 gunicorn 直接启动，
                            # 底层仍然是Python进程，命令行中通常会显示Python解释器路径
                            if 'python' in cmdline.lower():
                                processes.append({
                                    'pid': int(pid),
                                    'name': parts[10].split()[0] if parts[10] else 'python',
                                    'cmdline': cmdline[:80]  # 限制长度
                                })
            else:
                # perf 可以监控任何进程
                cmd = ['ps', 'aux']
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
                
                if result.returncode == 0:
                    lines = result.stdout.strip().split('\n')
                    # 跳过标题行
                    for line in lines[1:]:
                        parts = line.split(None, 10)
                        if len(parts) >= 11:
                            pid = parts[1]
                            if pid.isdigit():
                                processes.append({
                                    'pid': int(pid),
                                    'name': parts[10].split()[0] if parts[10] else parts[0],
                                    'cmdline': parts[10][:80] if len(parts) > 10 else ' '.join(parts[2:])[:80]
                                })
        
        # 按PID排序
        processes.sort(key=lambda x: x['pid'])
        
    except Exception as e:
        logger.warning(f"获取进程列表失败: {e}")
    
    return processes


def select_process(console: Console, profiling_type: str) -> Optional[int]:
    """
    让用户从进程列表中选择一个进程
    
    Args:
        console: Rich Console 实例
        profiling_type: 采集类型
        
    Returns:
        选中的进程ID，如果用户取消则返回None
    """
    console.print()
    console.print("[bold cyan]正在获取进程列表...[/bold cyan]")
    
    processes = get_running_processes(profiling_type)
    
    if not processes:
        console.print("[bold red]未找到可用的进程，无法继续[/bold red]")
        return None
    
    # 显示进程列表
    console.print()
    table = Table(title=f"可用的{'Python' if profiling_type == 'python' else ''}进程列表")
    table.add_column("序号", style="cyan", width=6)
    table.add_column("PID", style="green", width=8)
    table.add_column("进程名", style="yellow", width=20)
    table.add_column("命令行", style="white", width=50)
    
    for idx, proc in enumerate(processes, 1):
        table.add_row(
            str(idx),
            str(proc['pid']),
            proc['name'],
            proc['cmdline']
        )
    
    console.print(table)
    console.print()
    
    # 让用户选择（只允许从列表中选择，不允许手动输入）
    process_choices = [
        Choice(
            value=idx,
            name=f"PID {proc['pid']} - {proc['name']} ({proc['cmdline'][:50]}...)"
        )
        for idx, proc in enumerate(processes)
    ]
    
    selected_idx = run_inquirer_sync(
        lambda: inquirer.select(
            message=f"请选择进程:",
            choices=process_choices,
            default=0
        )
    )
    
    return processes[selected_idx]['pid']


async def collect_flamegraph(console: Console) -> Optional[str]:
    """
    采集火焰图
    
    Args:
        console: Rich Console 实例
        
    Returns:
        生成的火焰图文件路径，如果采集失败则返回None
    """
    console.print()
    console.print(Rule("[bold cyan]火焰图采集[/bold cyan]", style="cyan"))
    console.print()
    
    # 选择采集类型
    profiling_type = run_inquirer_sync(
        lambda: inquirer.select(
            message="请选择采集类型:",
            choices=[
                Choice(value="python", name="Python - 采集Python进程"),
                Choice(value="perf", name="Perf - 采集任意进程（需要perf工具）")
            ],
            default="python"
        )
    )
    
    # 选择进程
    pid = select_process(console, profiling_type)
    if pid is None:
        console.print("[bold red]未选择进程，取消采集[/bold red]")
        return None
    
    console.print(f"[bold green]✓ 已选择进程: PID {pid}[/bold green]")
    console.print()
    
    # 获取默认的火焰图存储目录（当前路径下的 data/flamegraphs/）
    default_flamegraph_dir = os.path.join(os.getcwd(), "data", "flamegraphs")
    
    # 选择输出路径（提供默认选项或手动输入）
    output_dir_choice = run_inquirer_sync(
        lambda: inquirer.select(
            message="选择输出目录:",
            choices=[
                Choice(value="data", name=f"data/flamegraphs/ - 项目数据目录（推荐）"),
                Choice(value="tmp", name=f"/tmp - 系统临时目录"),
                Choice(value="current", name=f"{os.getcwd()} - 当前目录"),
                Choice(value="custom", name="自定义路径")
            ],
            default="data"
        )
    )
    
    if output_dir_choice == "data":
        output_dir = default_flamegraph_dir
    elif output_dir_choice == "tmp":
        output_dir = "/tmp"
    elif output_dir_choice == "current":
        output_dir = os.getcwd()
    else:
        output_dir = run_inquirer_sync(
            lambda: inquirer.text(
                message="请输入输出目录路径:",
                default=default_flamegraph_dir
            )
        )
    
    # 确保输出目录是绝对路径
    output_dir = os.path.abspath(output_dir)
    
    # 确保目录存在
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir, exist_ok=True)
            console.print(f"[bold green]✓ 已创建目录: {output_dir}[/bold green]")
        except Exception as e:
            console.print(f"[bold red]无法创建目录 {output_dir}: {e}[/bold red]")
            return None
    
    output_filename = f"flamegraph_{int(time.time())}.svg"
    output_path = os.path.join(output_dir, output_filename)
    # 确保输出路径是绝对路径
    output_path = os.path.abspath(output_path)
    
    console.print(f"[bold cyan]输出路径: {output_path}[/bold cyan]")
    console.print()
    
    # 选择采样频率
    rate_choice = run_inquirer_sync(
        lambda: inquirer.select(
            message="请选择采样频率:",
            choices=[
                Choice(value="50", name="50 Hz - 低频率采样"),
                Choice(value="100", name="100 Hz - 推荐频率"),
                Choice(value="200", name="200 Hz - 高频率采样"),
                Choice(value="500", name="500 Hz - 超高频率采样"),
                Choice(value="custom", name="自定义频率")
            ],
            default="100"
        )
    )
    
    if rate_choice == "custom":
        rate = int(run_inquirer_sync(
            lambda: inquirer.number(
                message="请输入采样频率 (Hz):",
                default=100,
                min_allowed=1,
                max_allowed=1000
            )
        ))
    else:
        rate = int(rate_choice)
    
    # 选择采集模式
    duration_mode = run_inquirer_sync(
        lambda: inquirer.select(
            message="请选择采集模式:",
            choices=[
                Choice(value="timed", name="定时采集 - 自动停止"),
                Choice(value="manual", name="手动停止 - 按Enter键停止")
            ],
            default="timed"
        )
    )
    
    duration = None
    if duration_mode == "timed":
        duration_choice = run_inquirer_sync(
            lambda: inquirer.select(
                message="请选择采集时长:",
                choices=[
                    Choice(value="10", name="10秒"),
                    Choice(value="30", name="30秒"),
                    Choice(value="60", name="60秒 - 推荐"),
                    Choice(value="120", name="120秒"),
                    Choice(value="300", name="300秒"),
                    Choice(value="custom", name="自定义时长")
                ],
                default="60"
            )
        )
        
        if duration_choice == "custom":
            duration = int(run_inquirer_sync(
                lambda: inquirer.number(
                    message="请输入采集时长 (秒):",
                    default=60,
                    min_allowed=1,
                    max_allowed=3600
                )
            ))
        else:
            duration = int(duration_choice)
    
    # 启动采集
    console.print()
    console.print(f"[bold cyan]正在启动{profiling_type}采集...[/bold cyan]")
    
    try:
        result = await _profiling_tool.start_flamegraph_profiling(
            profiling_type=profiling_type,
            output_path=output_path,
            pid=pid,
            duration=duration,
            rate=rate
        )
        
        if not result.get('success'):
            console.print(f"[bold red]启动采集失败: {result.get('error')}[/bold red]")
            return None
        
        task_id = result.get('task_id')
        console.print(f"[bold green]✓ 采集已启动，任务ID: {task_id}[/bold green]")
        
        if duration:
            console.print(f"[bold cyan]采集将在 {duration} 秒后自动停止...[/bold cyan]")
            # 等待采集完成
            await asyncio.sleep(duration)
            console.print("[bold cyan]采集时间到，正在停止并生成火焰图...[/bold cyan]")
        else:
            console.print("[bold cyan]采集正在进行中，按 Enter 键停止采集...[/bold cyan]")
            input()  # 等待用户按Enter
            console.print("[bold cyan]正在停止采集并生成火焰图...[/bold cyan]")
        
        # 停止采集
        stop_result = await _profiling_tool.stop_flamegraph_profiling(task_id)
        
        if stop_result.get('success'):
            output_file = stop_result.get('output_path')
            if output_file and os.path.exists(output_file):
                file_size = os.path.getsize(output_file)
                console.print(f"[bold green]✓ 采集完成，火焰图已生成: {output_file} ({file_size} 字节)[/bold green]")
                return output_file
            else:
                console.print(f"[bold yellow]警告: 采集已停止，但文件未找到: {output_file}[/bold yellow]")
                if stop_result.get('error'):
                    console.print(f"[bold red]错误信息: {stop_result.get('error')}[/bold red]")
                return None
        else:
            error_msg = stop_result.get('error', '未知错误')
            console.print(f"[bold red]停止采集失败: {error_msg}[/bold red]")
            if stop_result.get('output_path'):
                console.print(f"[dim]预期输出路径: {stop_result.get('output_path')}[/dim]")
            return None
            
    except Exception as e:
        console.print(f"[bold red]采集过程出错: {e}[/bold red]")
        import traceback
        console.print(traceback.format_exc())
        return None


def find_flamegraph_files(search_dirs: List[str] = None) -> List[Dict[str, str]]:
    """
    查找常见的火焰图文件位置中的SVG文件
    
    Args:
        search_dirs: 要搜索的目录列表，如果为None则使用默认目录
        
    Returns:
        找到的火焰图文件列表，每个文件包含 path 和 name
    """
    if search_dirs is None:
        # 默认搜索目录：优先搜索项目数据目录，然后是其他常见位置
        default_flamegraph_dir = os.path.join(os.getcwd(), "data", "flamegraphs")
        search_dirs = [
            default_flamegraph_dir,  # 项目数据目录（优先）
            "/tmp",
            os.getcwd(),
            os.path.expanduser("~/")
        ]
    
    flamegraph_files = []
    seen_files = set()
    
    for search_dir in search_dirs:
        if not os.path.exists(search_dir) or not os.path.isdir(search_dir):
            continue
        
        try:
            for file in os.listdir(search_dir):
                if file.endswith('.svg') and ('flamegraph' in file.lower() or 'flame' in file.lower()):
                    file_path = os.path.join(search_dir, file)
                    if file_path not in seen_files:
                        flamegraph_files.append({
                            'path': file_path,
                            'name': file,
                            'dir': search_dir
                        })
                        seen_files.add(file_path)
        except (PermissionError, OSError):
            continue
    
    # 按修改时间排序（最新的在前）
    flamegraph_files.sort(key=lambda x: os.path.getmtime(x['path']) if os.path.exists(x['path']) else 0, reverse=True)
    
    return flamegraph_files


def select_flamegraph_file(console: Console) -> Optional[str]:
    """
    让用户从文件列表中选择火焰图文件，或手动输入路径
    
    Args:
        console: Rich Console 实例
        
    Returns:
        选中的火焰图文件路径，如果用户取消则返回None
    """
    console.print()
    console.print("[bold cyan]正在查找火焰图文件...[/bold cyan]")
    
    files = find_flamegraph_files()
    
    if not files:
        console.print("[bold yellow]未找到火焰图文件，请手动输入路径[/bold yellow]")
        flamegraph_path = run_inquirer_sync(
            lambda: inquirer.text(
                message="请输入火焰图SVG文件路径:",
                default=""
            )
        )
        if flamegraph_path and os.path.exists(flamegraph_path):
            return flamegraph_path
        elif flamegraph_path:
            console.print(f"[bold red]警告: 文件 {flamegraph_path} 不存在[/bold red]")
            return None
        else:
            return None
    
    # 构建文件选择列表
    file_choices = []
    for idx, file_info in enumerate(files):
        try:
            mtime = os.path.getmtime(file_info['path'])
            mtime_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(mtime))
        except:
            mtime_str = "未知"
        
        file_choices.append(
            Choice(
                value=idx,
                name=f"{file_info['name']} ({file_info['dir']}) - {mtime_str}"
            )
        )
    
    # 添加手动输入选项
    file_choices.append(
        Choice(value="manual", name="手动输入路径")
    )
    
    # 让用户选择
    choice = run_inquirer_sync(
        lambda: inquirer.select(
            message="请选择火焰图文件:",
            choices=file_choices,
            default=0
        )
    )
    
    if choice == 'manual':
        flamegraph_path = run_inquirer_sync(
            lambda: inquirer.text(
                message="请输入火焰图SVG文件路径:",
                default=""
            )
        )
        if flamegraph_path and os.path.exists(flamegraph_path):
            return flamegraph_path
        elif flamegraph_path:
            console.print(f"[bold red]警告: 文件 {flamegraph_path} 不存在[/bold red]")
            return None
        else:
            return None
    else:
        return files[choice]['path']


def get_user_input(console: Console) -> tuple[str, str]:
    """
    获取用户输入：火焰图文件路径和问题
    
    Args:
        console: Rich Console 实例
        
    Returns:
        (flamegraph_path, question) 元组
    """
    console.print()
    console.print(Rule("[bold cyan]请输入火焰图分析参数[/bold cyan]", style="cyan"))
    console.print()
    
    # 从文件列表中选择火焰图文件
    flamegraph_path = select_flamegraph_file(console)
    
    if not flamegraph_path:
        console.print("[bold red]未选择有效的火焰图文件，无法继续分析[/bold red]")
        return "", ""
    
    console.print(f"[bold green]✓ 已选择文件: {flamegraph_path}[/bold green]")
    
    # 获取用户问题
    console.print()
    console.print("[dim]提示: 可以询问性能瓶颈、CPU占用最高的函数、优化建议等问题[/dim]")
    console.print()
    
    while True:
        question = run_inquirer_sync(
            lambda: inquirer.text(
                message="请输入您的问题:",
                default="请分析这个火焰图，找出CPU占用最高的函数和性能瓶颈"
            )
        )
        
        if question and question.strip():
            break
        else:
            console.print("[bold yellow]问题不能为空，请重新输入[/bold yellow]")
            console.print()
    
    return flamegraph_path, question


def build_question(flamegraph_path: str, user_question: str) -> str:
    """
    构建完整的问题，包含火焰图文件路径
    
    Args:
        flamegraph_path: 火焰图文件路径
        user_question: 用户的问题
        
    Returns:
        完整的问题文本
    """
    if flamegraph_path:
        question = f"""
火焰图文件路径: {flamegraph_path}

{user_question}

请使用火焰图分析工具来分析这个文件并回答我的问题。
        """
    else:
        question = user_question
    
    return question.strip()


async def run_analysis(
    agent,
    langfuse_handler,
    console: Console,
    flamegraph_path: str,
    user_question: str
):
    """
    执行火焰图分析
    
    Args:
        agent: Flamegraph analysis agent 实例
        langfuse_handler: Langfuse callback handler
        console: Rich Console 实例
        flamegraph_path: 火焰图文件路径
        user_question: 用户的问题
    """
    # 构建问题
    question = build_question(flamegraph_path, user_question)
    
    # 显示问题
    console.print()
    console.print(Rule("[bold cyan]分析问题[/bold cyan]", style="cyan"))
    console.print()
    console.print(Panel(question, title="[bold yellow]问题[/bold yellow]", border_style="yellow"))
    console.print()
    
    try:
        # 创建流式输出处理器
        handler = StreamingOutputHandler(console=console)
        
        # 异步流式执行
        result = await stream_agent_execution(
            agent=agent,
            input=dict(
                messages=[
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
        
        output = result.get("output", "No output")
        if isinstance(output, str):
            console.print(Panel(output, title="[bold green]最终结果[/bold green]", border_style="green"))
        else:
            console.print(Panel(str(output), title="[bold green]最终结果[/bold green]", border_style="green"))
        
    except Exception as e:
        console.print(f"[bold red]Error: {e}[/bold red]")
        import traceback
        console.print(traceback.format_exc())


async def main():
    """Interactive flamegraph analysis with user input."""
    console = Console()
    
    console.print()
    console.print(Rule("[bold cyan]Flamegraph Analysis Agent - CPU性能分析[/bold cyan]", style="cyan"))
    console.print()
    
    # Initialize agent (analysis-only by default)
    console.print()
    console.print(Rule("[bold yellow]初始化 Flamegraph Analysis Agent...[/bold yellow]", style="yellow"))
    console.print()
    
    agent, langfuse_handler = setup_agent()
    if agent is None or langfuse_handler is None:
        console.print("[bold red]初始化失败，退出。[/bold red]")
        return
    
    console.print("[bold green]✓ Agent 初始化成功[/bold green]")
    console.print("[bold green]✓ Langfuse tracing 已启用[/bold green]")
    console.print("[bold green]✓ 流式输出模式已启用[/bold green]")
    
    # 主循环：允许用户多次分析
    while True:
        try:
            # 选择模式：使用已有文件或采集新文件
            console.print()
            console.print(Rule("[bold cyan]选择操作模式[/bold cyan]", style="cyan"))
            console.print()
            
            mode = run_inquirer_sync(
                lambda: inquirer.select(
                    message="请选择操作模式:",
                    choices=[
                        Choice(value="auto", name="自动采集并分析（推荐）- 适合“CPU占用高，帮我分析”"),
                        Choice(value="file", name="分析已有文件 - 分析已存在的火焰图SVG文件"),
                        Choice(value="collect", name="采集新文件 - 实时采集进程的CPU火焰图"),
                    ],
                    default="auto"
                )
            )
            
            # 显示选中的模式
            mode_names = {
                "file": "分析已有文件",
                "collect": "采集新文件",
                "auto": "自动采集并分析（推荐）",
            }
            console.print(f"[bold green]✓ 已选择: {mode_names[mode]}[/bold green]")
            console.print()
            
            flamegraph_path = None
            
            user_question = ""
            
            if mode == "auto":
                # 自动采集并分析：使用具备采集工具的 agent
                from langchain_deepseek import ChatDeepSeek
                model = ChatDeepSeek(model="deepseek-chat")
                agent = create_flamegraph_auto_profiling_agent(model=model, config={})

                console.print()
                console.print(Rule("[bold cyan]自动采集并分析（推荐）[/bold cyan]", style="cyan"))
                console.print()
                console.print("[dim]提示: 该模式会自动采集一段时间的CPU火焰图，然后基于采集结果给出分析报告。[/dim]")

                # 自动模式：不要求用户选择进程，交给模型通过 shell_exec 自行发现可疑 PID
                profiling_type = run_inquirer_sync(
                    lambda: inquirer.select(
                        message="请选择采集类型（如果不确定，选 Python；模型会根据进程情况调整）:",
                        choices=[
                            Choice(value="python", name="Python - 优先用于 Python 进程（py-spy）"),
                            Choice(value="perf", name="Perf - 用于任意进程（需要 perf 工具）")
                        ],
                        default="python"
                    )
                )

                # 自动模式默认采集 60 秒
                duration = int(run_inquirer_sync(
                    lambda: inquirer.select(
                        message="请选择采集时长:",
                        choices=[
                            Choice(value="30", name="30秒"),
                            Choice(value="60", name="60秒 - 推荐"),
                            Choice(value="120", name="120秒"),
                            Choice(value="custom", name="自定义时长")
                        ],
                        default="60"
                    )
                ))
                if duration == "custom":  # type: ignore[comparison-overlap]
                    duration = int(run_inquirer_sync(
                        lambda: inquirer.number(
                            message="请输入采集时长 (秒):",
                            default=60,
                            min_allowed=1,
                            max_allowed=3600
                        )
                    ))

                rate = int(run_inquirer_sync(
                    lambda: inquirer.select(
                        message="请选择采样频率:",
                        choices=[
                            Choice(value="50", name="50 Hz"),
                            Choice(value="100", name="100 Hz - 推荐"),
                            Choice(value="200", name="200 Hz"),
                            Choice(value="custom", name="自定义频率")
                        ],
                        default="100"
                    )
                ))
                if rate == "custom":  # type: ignore[comparison-overlap]
                    rate = int(run_inquirer_sync(
                        lambda: inquirer.number(
                            message="请输入采样频率 (Hz):",
                            default=100,
                            min_allowed=1,
                            max_allowed=1000
                        )
                    ))

                # 输出目录：默认 data/flamegraphs
                default_flamegraph_dir = os.path.join(os.getcwd(), "data", "flamegraphs")
                os.makedirs(default_flamegraph_dir, exist_ok=True)
                output_path = os.path.abspath(
                    os.path.join(default_flamegraph_dir, f"flamegraph_auto_{int(time.time())}.svg")
                )

                # 构造一个让 agent 自动采集并分析的自然语言请求
                user_question = run_inquirer_sync(
                    lambda: inquirer.text(
                        message="请输入你的排障诉求（可选）:",
                        default="我当前机器CPU占用比较高，帮我分析主要问题"
                    )
                ).strip()

                auto_request = f"""
我这台机器 CPU 占用比较高，请你自动采集并分析主要问题。

请先使用 ShellToolMiddleware 提供的 Shell 工具查询当前 CPU 占用最高的进程，并自动选择最可疑的目标 pid（必要时再向我确认一次）。

采集参数：
- profiling_type: {profiling_type}
- duration: {duration}
- rate: {rate}
- output_path: {output_path}

诉求：
{user_question}

请使用 flamegraph_collect_profiling 做阻塞式采集（一次调用闭环），采集结束后对生成的 SVG 做 overview 并钻取热点函数，给出分析报告。
                """.strip()

                # 通过 agent 来执行自动采集 + 分析
                await run_analysis(agent, langfuse_handler, console, flamegraph_path="", user_question=auto_request)

            elif mode == "collect":
                # 采集模式
                flamegraph_path = await collect_flamegraph(console)
                
                if not flamegraph_path:
                    console.print("[bold red]采集失败，无法继续分析[/bold red]")
                    continue
                
                # 询问是否分析
                console.print()
                analyze_after_collect = run_inquirer_sync(
                    lambda: inquirer.confirm(
                        message="采集完成，是否立即分析这个火焰图？",
                        default=True
                    )
                )
                
                if not analyze_after_collect:
                    continue
                
                # 获取用户问题
                console.print()
                console.print(Rule("[bold cyan]请输入分析问题[/bold cyan]", style="cyan"))
                console.print()
                console.print("[dim]提示: 可以询问性能瓶颈、CPU占用最高的函数、优化建议等问题[/dim]")
                console.print()
                
                while True:
                    user_question = run_inquirer_sync(
                        lambda: inquirer.text(
                            message="请输入您的问题:",
                            default="请分析这个火焰图，找出CPU占用最高的函数和性能瓶颈"
                        )
                    )
                    
                    if user_question and user_question.strip():
                        break
                    else:
                        console.print("[bold yellow]问题不能为空，请重新输入[/bold yellow]")
                        console.print()
            
            else:
                # 文件模式：获取用户输入（文件路径和问题）
                flamegraph_path, user_question = get_user_input(console)
            
            # 执行分析（确保有文件路径）——auto 模式已在上面执行
            if flamegraph_path and mode != "auto":
                await run_analysis(agent, langfuse_handler, console, flamegraph_path, user_question)
            elif mode != "auto":
                console.print("[bold yellow]未提供火焰图文件路径，跳过分析[/bold yellow]")
            
            # 询问是否继续
            console.print()
            continue_analysis = run_inquirer_sync(
                lambda: inquirer.confirm(
                    message="是否继续分析其他火焰图？",
                    default=False
                )
            )
            
            if not continue_analysis:
                break
                
        except KeyboardInterrupt:
            console.print()
            console.print("[bold yellow]用户中断，退出程序。[/bold yellow]")
            break
        except Exception as e:
            console.print(f"[bold red]发生错误: {e}[/bold red]")
            import traceback
            console.print(traceback.format_exc())
            
            # 询问是否继续
            continue_analysis = run_inquirer_sync(
                lambda: inquirer.confirm(
                    message="是否继续？",
                    default=False
                )
            )
            
            if not continue_analysis:
                break
    
    console.print()
    console.print(Rule("[bold green]分析完成[/bold green]", style="green"))
    console.print()
    console.print("[bold yellow]提示: 请访问 Langfuse 控制台查看详细的执行追踪信息[/bold yellow]")


if __name__ == "__main__":
    asyncio.run(main())

