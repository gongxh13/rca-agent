"""
通用流式输出工具

提供美观的流式输出功能，支持 LangChain 的多种流式模式：
- messages: LLM token 流式输出
- custom: 自定义更新流式输出
- updates: Agent 进度更新流式输出

使用 rich 库进行美观的命令行输出。
"""

import asyncio
import uuid
import json
import time
import codecs
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.text import Text
from rich.live import Live
from rich.console import Group
from rich.markdown import Markdown
from datetime import datetime
from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.validation import Validator, ValidationError
from langgraph.types import Command, Interrupt


class NonEmptyValidator(Validator):
    """验证输入不能为空"""
    def validate(self, document):
        if not document.text.strip():
            raise ValidationError(message="输入不能为空，请重新输入")


def build_prompt_message(header: str) -> HTML:
    """构建提示消息"""
    return HTML(
        f"\n\n➡️ <b><ansiyellow>{header}</ansiyellow></b> > \n\n"
        "<ansiblue>👉 编辑完成后，请按 </ansiblue>"
        "<ansigreen><b>Esc</b></ansigreen>"
        "<ansiblue> 然后 </ansiblue>"
        "<ansigreen><b>Enter</b></ansigreen>"
        "<ansiblue> 提交。</ansiblue>\n\n"
    )


def fix_utf8_encoding(text: str) -> str:
    """
    修复 UTF-8 编码错误，特别是处理不完整的中文字符序列
    
    当用户在命令行删除部分汉字时，可能会产生代理对（surrogates）或无效的 UTF-8 序列。
    这个函数会尝试修复这些问题。
    
    Args:
        text: 可能包含编码错误的文本
        
    Returns:
        修复后的文本
    """
    if not text:
        return text
    
    # 如果输入不是字符串，先转换为字符串
    if not isinstance(text, str):
        try:
            text = str(text)
        except Exception:
            return ""
    
    try:
        # 方法1：尝试使用 surrogatepass 处理代理对
        # 先将字符串编码为 UTF-8（允许代理对），然后解码（替换无效字符）
        text_bytes = text.encode('utf-8', errors='surrogatepass')
        fixed_text = text_bytes.decode('utf-8', errors='replace')
        return fixed_text
    except (UnicodeEncodeError, UnicodeDecodeError, UnicodeError):
        try:
            # 方法2：直接使用 replace 错误处理策略
            # 这会替换所有无效字符为替换字符（通常是小方块）
            text_bytes = text.encode('utf-8', errors='replace')
            fixed_text = text_bytes.decode('utf-8', errors='replace')
            return fixed_text
        except Exception:
            try:
                # 方法3：使用 ignore 策略，完全忽略无效字符
                # 这可能会丢失一些字符，但至少不会崩溃
                fixed_text = text.encode('utf-8', errors='ignore').decode('utf-8', errors='ignore')
                return fixed_text
            except Exception:
                # 最后的备选方案：返回空字符串或原始文本的 ASCII 表示
                try:
                    return text.encode('ascii', errors='ignore').decode('ascii')
                except Exception:
                    return ""


async def handle_interrupt(
    interrupt_data: Any,
    console: Console,
    live: Optional[Live]
) -> Tuple[str, Any]:
    """
    处理中断事件，获取用户决策或输入
    
    Args:
        interrupt_data: 中断数据，可能是 Interrupt 对象的 tuple 或单个 Interrupt，或自定义格式
        console: Rich Console 实例
        live: Live 实例（如果有的话，需要先停止）
        
    Returns:
        tuple[str, Any]: 
        - 如果是内置 HITL 格式，返回 ("decisions", decisions_list)
        - 如果是自定义格式，返回 ("text", user_input_string)
    """
    # 停止 Live 显示（如果有）
    if live is not None:
        live.stop()
    
    # 提取实际的 HITL 请求数据
    hitl_request = None
    raw_interrupt_value = None
    
    # 处理 tuple 格式的中断数据
    if isinstance(interrupt_data, tuple):
        for each in interrupt_data:
            if isinstance(each, Interrupt):
                raw_interrupt_value = each.value
                hitl_request = each.value if isinstance(each.value, dict) else None
                break
    # 处理单个 Interrupt 对象
    elif isinstance(interrupt_data, Interrupt):
        raw_interrupt_value = interrupt_data.value
        hitl_request = interrupt_data.value if isinstance(interrupt_data.value, dict) else None
    # 处理字典格式
    elif isinstance(interrupt_data, dict):
        raw_interrupt_value = interrupt_data
        # 如果字典中已经有 action_requests，说明已经是 HITL 请求格式
        if "action_requests" in interrupt_data:
            hitl_request = interrupt_data
        else:
            # 可能是包装在其他键中，尝试查找
            for key, value in interrupt_data.items():
                if isinstance(value, dict) and "action_requests" in value:
                    hitl_request = value
                    break
    
    # 检查是否是内置 HITL 格式（有 action_requests 和 review_configs）
    is_builtin_hitl = (
        hitl_request is not None 
        and isinstance(hitl_request, dict) 
        and "action_requests" in hitl_request
    )
    
    # 如果不是内置格式，按自定义格式处理
    if not is_builtin_hitl:
        console.print()
        console.print(Rule("[bold yellow]检测到自定义中断事件[/bold yellow]", style="yellow"))
        console.print()
        
        # 显示中断数据
        if raw_interrupt_value is not None:
            if isinstance(raw_interrupt_value, dict):
                # 检查是否是 {"message": "..."} 格式，如果是 markdown 则美观显示
                if len(raw_interrupt_value) == 1 and "message" in raw_interrupt_value:
                    message_content = raw_interrupt_value["message"]
                    if isinstance(message_content, str) and message_content.strip():
                        # 检查是否包含 markdown 标记（如 #, **, *, `, ``` 等）
                        has_markdown = any(
                            marker in message_content
                            for marker in ["# ", "**", "* ", "`", "```", "> ", "- ", "1. ", "[", "]("]
                        )
                        if has_markdown:
                            # 使用 Markdown 渲染
                            console.print("[bold cyan]中断消息:[/bold cyan]")
                            console.print()
                            console.print(Markdown(message_content), style="cyan")
                        else:
                            # 普通文本，使用 Panel 包装
                            console.print("[bold cyan]中断消息:[/bold cyan]")
                            console.print()
                            console.print(Panel(message_content, border_style="cyan", expand=False))
                    else:
                        # 空的或非字符串 message，显示 JSON
                        console.print("[bold cyan]中断数据:[/bold cyan]")
                        console.print(json.dumps(raw_interrupt_value, indent=2, ensure_ascii=False))
                else:
                    # 其他字典格式，使用 JSON 格式显示
                    console.print("[bold cyan]中断数据:[/bold cyan]")
                    console.print(json.dumps(raw_interrupt_value, indent=2, ensure_ascii=False))
            elif isinstance(raw_interrupt_value, str):
                # 如果是字符串，检查是否是 markdown
                if any(marker in raw_interrupt_value for marker in ["# ", "**", "* ", "`", "```", "> ", "- ", "1. ", "[", "]("]):
                    console.print("[bold cyan]中断消息:[/bold cyan]")
                    console.print()
                    console.print(Markdown(raw_interrupt_value), style="cyan")
                else:
                    console.print("[bold cyan]中断数据:[/bold cyan]")
                    console.print(Panel(raw_interrupt_value, border_style="cyan", expand=False))
            else:
                # 其他类型，转换为字符串显示
                console.print("[bold cyan]中断数据:[/bold cyan]")
                console.print(str(raw_interrupt_value))
        else:
            console.print("[bold cyan]中断数据:[/bold cyan]")
            console.print(str(interrupt_data))
        
        console.print()
        console.print("[bold yellow]💡 请输入您的响应（将在 resume 中作为输入）[/bold yellow]")
        
        # 获取用户输入
        session = PromptSession(
            multiline=True,
            validator=NonEmptyValidator(),
            validate_while_typing=False,
        )
        try:
            user_input = await session.prompt_async(
                build_prompt_message("请输入响应")
            )
            # 修复可能的 UTF-8 编码错误
            user_input = fix_utf8_encoding(user_input)
        except UnicodeError as e:
            console.print(f"[bold red]编码错误: {e}[/bold red]")
            console.print("[bold yellow]尝试修复编码问题...[/bold yellow]")
            user_input = ""
        
        console.print()
        console.print(Rule(style="dim"))
        console.print()
        
        return ("text", user_input.strip())
    
    # 以下是内置 HITL 格式的处理
    
    action_requests = hitl_request.get("action_requests", [])
    review_configs = hitl_request.get("review_configs", [])
    
    if not action_requests:
        console.print("[bold yellow]警告: 中断请求中没有需要审核的操作[/bold yellow]")
        return ("decisions", [])
    
    console.print()
    console.print(Rule("[bold yellow]需要人工审核的操作[/bold yellow]", style="yellow"))
    console.print()
    
    # 显示每个需要审核的操作
    decisions = []
    session = PromptSession(
        multiline=True,
        validator=NonEmptyValidator(),
        validate_while_typing=False,
    )
    
    for idx, action_request in enumerate(action_requests):
        action_name = action_request.get("name", "unknown")
        arguments = action_request.get("arguments", {})
        description = action_request.get("description", "")
        
        # 获取该操作允许的决策类型
        allowed_decisions = ["approve", "edit", "reject"]  # 默认允许所有
        for review_config in review_configs:
            if review_config.get("action_name") == action_name:
                allowed_decisions = review_config.get("allowed_decisions", allowed_decisions)
                break
        
        # 显示操作信息
        console.print(f"\n[bold cyan]操作 {idx + 1}/{len(action_requests)}: {action_name}[/bold cyan]")
        console.print(f"[dim]参数:[/dim] {json.dumps(arguments, indent=2, ensure_ascii=False)}")
        if description:
            console.print(Markdown(description), style="cyan")
        
        # 显示可用的决策选项
        options_text = "可用选项: "
        if "approve" in allowed_decisions:
            options_text += "[green]✅ approve[/green]"
        if "edit" in allowed_decisions:
            options_text += " [yellow]✏️ edit[/yellow]"
        if "reject" in allowed_decisions:
            options_text += " [red]❌ reject[/red]"
        console.print(options_text)
        console.print()
        
        # 获取用户决策
        console.print("[bold yellow]💡 请选择您的决策[/bold yellow]")
        try:
            user_input = await session.prompt_async(
                build_prompt_message("请输入决策 (approve/edit/reject)")
            )
            # 修复可能的 UTF-8 编码错误
            user_input = fix_utf8_encoding(user_input)
        except UnicodeError as e:
            console.print(f"[bold red]编码错误: {e}[/bold red]")
            console.print("[bold yellow]尝试修复编码问题...[/bold yellow]")
            user_input = ""
        
        user_input = user_input.strip().lower()
        
        # 解析用户输入
        decision = None
        
        # 检查用户输入是否匹配 approve
        if user_input in ["approve", "a", "y", "yes", "同意", "批准"]:
            if "approve" in allowed_decisions:
                decision = {"type": "approve"}
            else:
                console.print(f"[bold red]错误: 此操作不允许 approve 决策，可用选项: {', '.join(allowed_decisions)}[/bold red]")
                # 继续处理，使用默认决策
        
        # 检查用户输入是否匹配 edit
        elif user_input in ["edit", "e", "修改", "编辑"]:
            if "edit" in allowed_decisions:
                # 获取编辑后的操作
                console.print("[bold yellow]请输入编辑后的工具名称 (留空表示不变):[/bold yellow]")
                try:
                    new_tool_name = await session.prompt_async(
                        build_prompt_message("工具名称")
                    )
                    # 修复可能的 UTF-8 编码错误
                    new_tool_name = fix_utf8_encoding(new_tool_name)
                except UnicodeError as e:
                    console.print(f"[bold red]编码错误: {e}[/bold red]")
                    console.print("[bold yellow]使用原始工具名称...[/bold yellow]")
                    new_tool_name = action_name
                
                new_tool_name = new_tool_name.strip() or action_name
                
                console.print("[bold yellow]请输入编辑后的参数 (JSON格式，留空表示不变):[/bold yellow]")
                try:
                    new_args_input = await session.prompt_async(
                        build_prompt_message("参数 (JSON)")
                    )
                    # 修复可能的 UTF-8 编码错误
                    new_args_input = fix_utf8_encoding(new_args_input)
                except UnicodeError as e:
                    console.print(f"[bold red]编码错误: {e}[/bold red]")
                    console.print("[bold yellow]使用原始参数...[/bold yellow]")
                    new_args_input = ""
                
                new_args_input = new_args_input.strip()
                
                if new_args_input:
                    try:
                        new_args = json.loads(new_args_input)
                    except json.JSONDecodeError:
                        console.print("[bold red]错误: JSON格式无效，使用原始参数[/bold red]")
                        new_args = arguments
                else:
                    new_args = arguments
                
                decision = {
                    "type": "edit",
                    "edited_action": {
                        "name": new_tool_name,
                        "args": new_args
                    }
                }
            else:
                console.print(f"[bold red]错误: 此操作不允许 edit 决策，可用选项: {', '.join(allowed_decisions)}[/bold red]")
                # 继续处理，使用默认决策
        
        # 检查用户输入是否匹配 reject
        elif user_input in ["reject", "r", "n", "no", "拒绝", "驳回"]:
            if "reject" in allowed_decisions:
                console.print("[bold yellow]请输入拒绝原因:[/bold yellow]")
                try:
                    reject_message = await session.prompt_async(
                        build_prompt_message("拒绝原因")
                    )
                    # 修复可能的 UTF-8 编码错误
                    reject_message = fix_utf8_encoding(reject_message)
                except UnicodeError as e:
                    console.print(f"[bold red]编码错误: {e}[/bold red]")
                    console.print("[bold yellow]使用默认拒绝原因...[/bold yellow]")
                    reject_message = "用户拒绝"
                
                decision = {
                    "type": "reject",
                    "message": reject_message.strip()
                }
            else:
                console.print(f"[bold red]错误: 此操作不允许 reject 决策，可用选项: {', '.join(allowed_decisions)}[/bold red]")
                # 继续处理，使用默认决策
        
        # 如果用户输入无效或不被允许，使用默认决策
        if decision is None:
            if allowed_decisions:
                default_decision_type = allowed_decisions[0]
                if user_input not in ["approve", "a", "y", "yes", "同意", "批准", 
                                       "edit", "e", "修改", "编辑",
                                       "reject", "r", "n", "no", "拒绝", "驳回"]:
                    # 安全地打印用户输入，避免编码错误
                    safe_user_input = fix_utf8_encoding(user_input) if user_input else ""
                    try:
                        console.print(f"[bold yellow]警告: 无法识别输入 '{safe_user_input}'，使用默认决策: {default_decision_type}[/bold yellow]")
                    except UnicodeError:
                        console.print(f"[bold yellow]警告: 无法识别输入，使用默认决策: {default_decision_type}[/bold yellow]")
                
                if default_decision_type == "approve":
                    decision = {"type": "approve"}
                elif default_decision_type == "edit":
                    # 对于 edit，使用原始参数（不修改）
                    decision = {
                        "type": "edit",
                        "edited_action": {
                            "name": action_name,
                            "args": arguments
                        }
                    }
                elif default_decision_type == "reject":
                    decision = {"type": "reject", "message": "默认拒绝"}
        
        if decision:
            decisions.append(decision)
        
        console.print()
    
    console.print(Rule(style="dim"))
    console.print()
    
    return ("decisions", decisions)


class StreamingOutputHandler:
    """
    流式输出处理器
    
    处理 LangChain agent 的流式输出，支持多种流式模式，并提供美观的实时显示。
    """
    
    def __init__(
        self,
        console: Optional[Console] = None,
        show_progress: bool = True,
        show_tokens: bool = True,
        show_custom: bool = True,
    ):
        """
        初始化流式输出处理器
        
        Args:
            console: Rich Console 实例，如果为 None 则创建新的
            show_progress: 是否显示 agent 进度更新
            show_tokens: 是否显示 LLM token 流式输出
            show_custom: 是否显示自定义更新
        """
        self.console = console or Console()
        self.show_progress = show_progress
        self.show_tokens = show_tokens
        self.show_custom = show_custom
        
        # 存储当前流式内容，使用消息id作为key
        self.accumulated_texts: Dict[str, str] = {}  # message_id -> content
        self.message_metadata: Dict[str, Dict[str, Any]] = {}  # message_id -> metadata (node_name, etc.)
        self.current_custom: List[str] = []
        self.current_updates: List[str] = []
        
        # 用于跟踪当前正在流式输出的消息id
        self.current_streaming_msg_id: Optional[str] = None
        
        # 使用 Live 组件来管理实时更新（全局单一实例）
        self.live: Optional[Live] = None
        
        # 记录开始时间，用于计算执行时间
        self.start_time: Optional[float] = None
        
    def _format_timestamp(self) -> str:
        """格式化时间戳"""
        return datetime.now().strftime("%H:%M:%S")
    
    def _format_elapsed_time(self, elapsed_seconds: float) -> str:
        """
        格式化已执行时间为易读的格式
        
        Args:
            elapsed_seconds: 已执行的秒数
            
        Returns:
            格式化后的时间字符串，如 "1m 23s" 或 "45s"
        """
        if elapsed_seconds < 60:
            return f"{int(elapsed_seconds)}s"
        elif elapsed_seconds < 3600:
            minutes = int(elapsed_seconds // 60)
            seconds = int(elapsed_seconds % 60)
            return f"{minutes}m {seconds}s"
        else:
            hours = int(elapsed_seconds // 3600)
            minutes = int((elapsed_seconds % 3600) // 60)
            seconds = int(elapsed_seconds % 60)
            return f"{hours}h {minutes}m {seconds}s"
    
    def _get_elapsed_time_str(self) -> str:
        """
        获取当前已执行时间的字符串表示
        
        Returns:
            已执行时间的字符串，如果未开始则返回空字符串
        """
        if self.start_time is None:
            self.start_time = time.time()
            return "0s"
        
        elapsed = time.time() - self.start_time
        return self._format_elapsed_time(elapsed)
    
    def _get_message_id(self, message_chunk: Any, metadata: Dict[str, Any]) -> str:
        """
        从消息块或元数据中提取或生成唯一的消息id
        
        Args:
            message_chunk: 消息块
            metadata: 元数据
            
        Returns:
            消息id字符串
        """
        # 尝试从消息块中获取id
        if hasattr(message_chunk, "id") and message_chunk.id:
            return str(message_chunk.id)
        
        # 尝试从metadata中获取run_id和node_name组合
        run_id = metadata.get("run_id", "") if metadata else ""
        node_name = metadata.get("langgraph_node", "unknown") if metadata else "unknown"
        
        # 使用run_id + node_name作为唯一标识符
        # 如果run_id存在，使用它；否则只使用node_name
        if run_id:
            return f"{run_id}_{node_name}"
        else:
            return node_name
    
    def _get_display_panel_for_message(self, msg_id: str) -> Panel:
        """
        为指定消息id构建显示Panel
        
        Args:
            msg_id: 消息id
            
        Returns:
            Panel 对象
        """
        if msg_id not in self.accumulated_texts:
            elapsed_time = self._get_elapsed_time_str()
            return Panel("", title=f"Message (已执行: {elapsed_time})", border_style="blue", expand=True)
        
        content = self.accumulated_texts[msg_id]
        metadata = self.message_metadata.get(msg_id, {})
        node_name = metadata.get("node_name", "unknown")
        
        # 获取已执行时间
        elapsed_time = self._get_elapsed_time_str()
        
        # 只显示消息内容，不显示时间戳和节点名称
        display_text = Text(content, style="")
        
        return Panel(display_text, title=f"Message ({node_name}) | 已执行: {elapsed_time}", border_style="blue", expand=True)
    
    def handle_messages_stream(
        self,
        message_chunk: Any,
        metadata: Dict[str, Any]
    ) -> None:
        """
        处理 messages 流式输出（LLM tokens）
        
        Args:
            message_chunk: LLM 生成的消息块（AIMessageChunk, AIMessage, ToolMessage 等）
            metadata: 元数据，包含节点信息等
        """
        if not self.show_tokens:
            return
        
        # 获取消息id
        msg_id = self._get_message_id(message_chunk, metadata)
        node_name = metadata.get("langgraph_node", "unknown") if metadata else "unknown"
        
        # 如果是新消息，初始化
        if msg_id not in self.accumulated_texts:
            self.accumulated_texts[msg_id] = ""
            self.message_metadata[msg_id] = {
                "node_name": node_name,
                "run_id": metadata.get("run_id", "") if metadata else "",
            }
            # 如果是新消息且当前有正在流式输出的消息，直接切换到新消息（通过刷新覆盖）
            # 不打印旧消息，让它被新消息覆盖
            if self.current_streaming_msg_id is not None and self.current_streaming_msg_id != msg_id:
                # 旧消息会被新消息覆盖，不需要特殊处理
                pass
        
        # 提取消息内容
        content = ""
        
        # 处理不同类型的消息块
        # 1. AIMessageChunk 或 AIMessage
        if hasattr(message_chunk, "content"):
            if isinstance(message_chunk.content, str):
                content = message_chunk.content
            elif isinstance(message_chunk.content, list):
                # 处理 content_blocks 格式
                for item in message_chunk.content:
                    if isinstance(item, dict):
                        if item.get("type") == "text":
                            content += item.get("text", "")
                        elif item.get("type") == "tool_call_chunk":
                            # 工具调用流式输出
                            args = item.get("args", "")
                            if args:
                                content += args
                    elif hasattr(item, "text"):
                        content += item.text
        
        # 2. 处理 content_blocks 属性（某些消息格式）
        elif hasattr(message_chunk, "content_blocks"):
            for block in message_chunk.content_blocks:
                if isinstance(block, dict):
                    if block.get("type") == "text":
                        content += block.get("text", "")
                    elif block.get("type") == "tool_call_chunk":
                        args = block.get("args", "")
                        if args:
                            content += args
                elif hasattr(block, "text"):
                    content += block.text
        
        # 3. 如果 message_chunk 是字符串
        elif isinstance(message_chunk, str):
            content = message_chunk
        
        # 4. 如果 message_chunk 是字典
        elif isinstance(message_chunk, dict):
            content = message_chunk.get("content", "")
            if isinstance(content, list):
                content = "".join(str(item) for item in content)
        
        if content:
            # 更新当前消息的内容
            self.accumulated_texts[msg_id] += content
            
            # 如果还没有 Live 实例，创建一个（全局单一实例）
            if self.live is None:
                panel = self._get_display_panel_for_message(msg_id)
                self.live = Live(panel, console=self.console, refresh_per_second=10, transient=False)
                self.live.start()
            
            # 更新 Live 显示内容（刷新当前消息）
            panel = self._get_display_panel_for_message(msg_id)
            self.live.update(panel)
            
            # 更新当前流式输出的消息id
            self.current_streaming_msg_id = msg_id
    
    def handle_custom_stream(self, chunk: Any) -> None:
        """
        处理 custom 流式输出（自定义更新）
        
        Args:
            chunk: 自定义更新内容
        """
        if not self.show_custom:
            return
        
        # 将自定义更新添加到列表
        custom_text = str(chunk)
        if custom_text not in self.current_custom:
            self.current_custom.append(custom_text)
        
        # 显示自定义更新（换行显示，不刷新）
        display_text = f"[yellow][{self._format_timestamp()}][/yellow] [bold yellow]Custom[/bold yellow]: {custom_text}"
        self.console.print(display_text)
    
    def handle_updates_stream(self, chunk: Dict[str, Any]) -> None:
        """
        处理 updates 流式输出（当前不处理）
        
        Args:
            chunk: 更新块，包含节点名称和状态信息或中断信息
        """
        # 暂时不处理 updates 类型的消息
        return
    
    def finalize_message_stream(self, msg_id: str) -> None:
        """
        完成某个消息的流式输出
        
        Args:
            msg_id: 消息id
        """
        # 如果这是当前正在流式输出的消息，停止 Live 并打印最终结果
        if self.current_streaming_msg_id == msg_id and self.live is not None:
            self.live.stop()
            self.live = None
            self.current_streaming_msg_id = None
        
        # 如果有内容，打印最终结果（正常换行，不会被覆盖）
        if msg_id in self.accumulated_texts and self.accumulated_texts[msg_id]:
            final_panel = self._get_display_panel_for_message(msg_id)
            self.console.print(final_panel)
            # 清空该消息的内容
            del self.accumulated_texts[msg_id]
            if msg_id in self.message_metadata:
                del self.message_metadata[msg_id]
    
    def get_last_message_content(self) -> Optional[str]:
        """
        获取最后一条消息的内容
        
        Returns:
            最后一条消息的内容，如果没有则返回 None
        """
        # 优先返回当前正在流式输出的消息
        if self.current_streaming_msg_id and self.current_streaming_msg_id in self.accumulated_texts:
            content = self.accumulated_texts[self.current_streaming_msg_id]
            if content:
                return content
        
        # 如果没有当前流式消息，返回最后一条有内容的消息
        if self.accumulated_texts:
            # 获取最后一条消息（按消息id顺序，或直接取任意一条）
            for msg_id in reversed(list(self.accumulated_texts.keys())):
                content = self.accumulated_texts[msg_id]
                if content:
                    return content
        
        return None
    
    def finalize_all(self) -> None:
        """完成所有流式输出，只显示最后一条消息的最终结果"""
        # 停止 Live 实例
        if self.live is not None:
            self.live.stop()
            self.live = None
        
        # 只打印最后一条消息（当前正在流式输出的消息）
        if self.current_streaming_msg_id:
            if self.current_streaming_msg_id in self.accumulated_texts and self.accumulated_texts[self.current_streaming_msg_id]:
                final_panel = self._get_display_panel_for_message(self.current_streaming_msg_id)
                self.console.print(final_panel)
                # 不清空，保留内容以便后续获取（在 cleanup 中清理）
        
        # 打印分隔线
        self.console.print()
        self.console.print(Rule(style="dim"))
        self.console.print()
    
    def cleanup(self) -> None:
        """清理所有资源，释放内存"""
        # 清空所有累积的消息
        self.accumulated_texts.clear()
        self.message_metadata.clear()
        self.current_custom.clear()
        self.current_updates.clear()
        self.current_streaming_msg_id = None


async def stream_agent_execution(
    agent: Any,
    input: Dict[str, Any],
    config: Optional[Dict[str, Any]] = None,
    stream_modes: List[str] = ["messages", "custom", "updates"],
    handler: Optional[StreamingOutputHandler] = None,
    _is_resume: bool = False  # 内部参数，表示是否是恢复执行
) -> Dict[str, Any]:
    """
    异步流式执行 agent，支持人机交互（Human-in-the-Loop）
    
    Args:
        agent: LangChain agent 实例
        input: 输入数据，如果是 Command 对象则表示恢复执行
        config: 配置信息（必须包含 thread_id 以支持中断恢复）
        stream_modes: 流式模式列表，支持 ["messages", "custom", "updates"]
        handler: 流式输出处理器，如果为 None 则创建新的
        _is_resume: 内部参数，表示是否是恢复执行（递归调用时使用）
        
    Returns:
        最终的执行结果
    """
    if handler is None:
        handler = StreamingOutputHandler()
    
    # 确保 stream_modes 是列表
    if isinstance(stream_modes, str):
        stream_modes = [stream_modes]
    
    # 准备配置
    agent_config = config or {}
    
    # 确保 config 中有 thread_id（用于中断恢复）
    if "configurable" not in agent_config:
        agent_config["configurable"] = {}
    if "thread_id" not in agent_config["configurable"]:
        agent_config["configurable"]["thread_id"] = str(uuid.uuid4())
    
    # 显示开始信息（仅在首次调用时，不是恢复执行时）
    if not _is_resume and not isinstance(input, Command):
        handler.console.print()
        handler.console.print(Rule("[bold cyan]Agent Execution Started[/bold cyan]", style="cyan"))
        handler.console.print()
    
    final_result = None
    last_update_state = None
    
    try:
        # 使用 astream 进行异步流式执行
        # 根据 stream_adapter.py，astream 可能返回：
        # 1. 三元组：(namespace, mode, data) - 多模式流式输出
        # 2. 二元组：(mode, data) - 某些情况下的多模式输出
        # 3. 直接数据 - 单模式输出
        async for chunk in agent.astream(
            input,
            config=agent_config,
            stream_mode=stream_modes,
            subgraphs=True,
        ):
            # 处理多模式流式输出
            if isinstance(chunk, tuple):
                if len(chunk) == 3:
                    # 三元组格式：(namespace, mode, data)
                    namespace, mode, chunk_data = chunk
                elif len(chunk) == 2:
                    # 二元组格式：(mode, chunk_data)
                    mode, chunk_data = chunk
                else:
                    # 无法识别的格式，跳过
                    continue
                
                if mode == "messages":
                    # messages 模式：data 是 (message_chunk, metadata) 元组
                    if isinstance(chunk_data, tuple) and len(chunk_data) == 2:
                        message_chunk, metadata = chunk_data
                        handler.handle_messages_stream(message_chunk, metadata)
                    else:
                        # 如果格式不对，尝试直接处理
                        handler.handle_messages_stream(chunk_data, {})
                
                elif mode == "custom":
                    # custom 模式：data 直接是自定义对象（字符串、FinalResult、ToolUnifiedResponse 等）
                    handler.handle_custom_stream(chunk_data)
                
                elif mode == "updates":
                    # updates 模式：检查是否有中断
                    # 根据用户提供的代码片段，__interrupt__ 可能在 chunk_data 中
                    if isinstance(chunk_data, dict) and "__interrupt__" in chunk_data:
                        interrupt_data = chunk_data["__interrupt__"]
                        # 处理中断，获取用户决策或输入
                        interrupt_result = await handle_interrupt(
                            interrupt_data,
                            handler.console,
                            handler.live
                        )
                        
                        interrupt_type, interrupt_value = interrupt_result
                        
                        if interrupt_type == "decisions":
                            # 内置 HITL 格式，使用 decisions
                            decisions = interrupt_value
                            if decisions:
                                resume_command = Command(resume={"decisions": decisions})
                                return await stream_agent_execution(
                                    agent,
                                    resume_command,
                                    config=agent_config,
                                    stream_modes=stream_modes,
                                    handler=handler,
                                    _is_resume=True
                                )
                        elif interrupt_type == "text":
                            # 自定义格式，使用文本输入
                            user_input = interrupt_value
                            if user_input:
                                resume_command = Command(resume=user_input)
                                return await stream_agent_execution(
                                    agent,
                                    resume_command,
                                    config=agent_config,
                                    stream_modes=stream_modes,
                                    handler=handler,
                                    _is_resume=True
                                )
                    # 也检查 chunk_data 本身是否包含中断信息（某些情况下可能直接在 updates 中）
                    elif isinstance(chunk_data, dict):
                        # 检查是否有 Interrupt 相关的数据
                        for key, value in chunk_data.items():
                            if key == "__interrupt__" or (isinstance(value, (tuple, list)) and 
                                any(isinstance(item, Interrupt) for item in value if isinstance(item, Interrupt))):
                                interrupt_data = value if key == "__interrupt__" else chunk_data
                                interrupt_result = await handle_interrupt(
                                    interrupt_data,
                                    handler.console,
                                    handler.live
                                )
                                interrupt_type, interrupt_value = interrupt_result
                                
                                if interrupt_type == "decisions":
                                    decisions = interrupt_value
                                    if decisions:
                                        resume_command = Command(resume={"decisions": decisions})
                                        return await stream_agent_execution(
                                            agent,
                                            resume_command,
                                            config=agent_config,
                                            stream_modes=stream_modes,
                                            handler=handler,
                                            _is_resume=True
                                        )
                                elif interrupt_type == "text":
                                    user_input = interrupt_value
                                    if user_input:
                                        resume_command = Command(resume=user_input)
                                        return await stream_agent_execution(
                                            agent,
                                            resume_command,
                                            config=agent_config,
                                            stream_modes=stream_modes,
                                            handler=handler,
                                            _is_resume=True
                                        )
                                break
            
            else:
                # 单模式输出或直接是更新块
                # 检查是否是 messages 模式的输出（message_chunk, metadata）
                if "messages" in stream_modes and isinstance(chunk, tuple) and len(chunk) == 2:
                    message_chunk, metadata = chunk
                    handler.handle_messages_stream(message_chunk, metadata)
                elif "custom" in stream_modes:
                    handler.handle_custom_stream(chunk)
                elif "updates" in stream_modes and isinstance(chunk, dict):
                    # updates 模式：检查是否有中断
                    if "__interrupt__" in chunk:
                        interrupt_data = chunk["__interrupt__"]
                        # 处理中断，获取用户决策或输入
                        interrupt_result = await handle_interrupt(
                            interrupt_data,
                            handler.console,
                            handler.live
                        )
                        
                        interrupt_type, interrupt_value = interrupt_result
                        
                        if interrupt_type == "decisions":
                            # 内置 HITL 格式，使用 decisions
                            decisions = interrupt_value
                            if decisions:
                                resume_command = Command(resume={"decisions": decisions})
                                return await stream_agent_execution(
                                    agent,
                                    resume_command,
                                    config=agent_config,
                                    stream_modes=stream_modes,
                                    handler=handler,
                                    _is_resume=True
                                )
                        elif interrupt_type == "text":
                            # 自定义格式，使用文本输入
                            user_input = interrupt_value
                            if user_input:
                                resume_command = Command(resume=user_input)
                                return await stream_agent_execution(
                                    agent,
                                    resume_command,
                                    config=agent_config,
                                    stream_modes=stream_modes,
                                    handler=handler,
                                    _is_resume=True
                                )
    
    except Exception as e:
        handler.console.print(f"[bold red]Error during streaming: {e}[/bold red]")
        import traceback
        handler.console.print(traceback.format_exc())
        raise
    
    finally:
        # 只有在正常结束（不是通过递归恢复执行）时才 finalize 和 cleanup
        if not _is_resume:
            # 完成所有流式输出
            handler.finalize_all()
            # 清理资源
            handler.cleanup()
    
    # 从流式输出中获取最后一条消息的内容作为最终结果
    last_message_content = handler.get_last_message_content()
    if last_message_content:
        final_result = {"output": last_message_content, "messages": []}
    
    return final_result

