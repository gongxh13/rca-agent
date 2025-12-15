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
from typing import Any, Dict, List, Optional
from collections import defaultdict
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.text import Text
from rich.live import Live
from rich.console import Group
from datetime import datetime


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
        
    def _format_timestamp(self) -> str:
        """格式化时间戳"""
        return datetime.now().strftime("%H:%M:%S")
    
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
            return Panel("", title="Message", border_style="blue", expand=True)
        
        content = self.accumulated_texts[msg_id]
        metadata = self.message_metadata.get(msg_id, {})
        node_name = metadata.get("node_name", "unknown")
        
        # 只显示消息内容，不显示时间戳和节点名称
        display_text = Text(content, style="")
        
        return Panel(display_text, title=f"Message ({node_name})", border_style="blue", expand=True)
    
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
    input_data: Dict[str, Any],
    config: Optional[Dict[str, Any]] = None,
    stream_modes: List[str] = ["messages", "custom", "updates"],
    handler: Optional[StreamingOutputHandler] = None
) -> Dict[str, Any]:
    """
    异步流式执行 agent
    
    Args:
        agent: LangChain agent 实例
        input_data: 输入数据
        config: 配置信息（包含 callbacks 等）
        stream_modes: 流式模式列表，支持 ["messages", "custom", "updates"]
        handler: 流式输出处理器，如果为 None 则创建新的
        
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
    
    # 显示开始信息
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
            input_data,
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
                    # updates 模式：暂时不处理
                    # 注意：不要在这里 finalize，因为 updates 可能不是消息完成的标志
                    # 只在流式输出真正结束时才 finalize
                    pass
            
            else:
                # 单模式输出或直接是更新块
                # 检查是否是 messages 模式的输出（message_chunk, metadata）
                if "messages" in stream_modes and isinstance(chunk, tuple) and len(chunk) == 2:
                    message_chunk, metadata = chunk
                    handler.handle_messages_stream(message_chunk, metadata)
                elif "custom" in stream_modes:
                    handler.handle_custom_stream(chunk)
                elif "updates" in stream_modes and isinstance(chunk, dict):
                    # updates 模式：暂时不处理
                    # 注意：不要在这里 finalize，因为 updates 可能不是消息完成的标志
                    pass
    
    except Exception as e:
        handler.console.print(f"[bold red]Error during streaming: {e}[/bold red]")
        import traceback
        handler.console.print(traceback.format_exc())
        raise
    
    finally:
        # 完成所有流式输出
        handler.finalize_all()
    
    # 从流式输出中获取最后一条消息的内容作为最终结果
    last_message_content = handler.get_last_message_content()
    if last_message_content:
        final_result = {"output": last_message_content, "messages": []}
    # 清理资源
    handler.cleanup()
    
    return final_result

