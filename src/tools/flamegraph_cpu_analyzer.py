"""
Flamegraph CPU Analyzer Tool

Provides tools for analyzing CPU flamegraph SVG files to identify performance bottlenecks
and understand function call hierarchies. This tool parses flamegraph SVG files and
extracts function-level performance data including CPU usage, call stacks, and hierarchical
relationships.

Uses LangChain's @tool decorator for agent integration.
"""

import json
import re
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional
from collections import defaultdict
import logging
import subprocess
import os
import signal
import time
import asyncio
import platform
import sys
from pathlib import Path
from langchain_core.tools import StructuredTool
from typing_extensions import Annotated, Doc

from langchain.tools import BaseTool, tool

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _get_current_script_path() -> Optional[str]:
    """
    获取当前运行的脚本路径
    
    Returns:
        脚本路径（相对路径或绝对路径），如果无法确定则返回 None
    """
    try:
        # 方法1: 检查 __main__ 模块（最可靠的方法）
        import __main__
        if hasattr(__main__, '__file__'):
            main_file = __main__.__file__
            if main_file and os.path.exists(main_file):
                # 尝试使用相对路径（相对于当前工作目录）
                try:
                    cwd = os.getcwd()
                    rel_path = os.path.relpath(main_file, cwd)
                    # 如果相对路径不包含 '..' 且更短，优先使用相对路径
                    if '..' not in rel_path and len(rel_path) < len(main_file):
                        return rel_path
                except (ValueError, OSError):
                    pass
                # 如果相对路径不合适，返回绝对路径
                return main_file
        
        # 方法2: 检查调用栈（备用方法）
        import inspect
        current_file = os.path.abspath(__file__)
        frame = inspect.currentframe()
        # 跳过当前函数和调用它的函数
        if frame:
            frame = frame.f_back  # 跳过 _get_current_script_path
            if frame:
                frame = frame.f_back  # 跳过 _get_permission_error_message
        
        # 向上查找调用栈，找到主脚本
        while frame:
            frame_file = os.path.abspath(frame.f_code.co_filename)
            # 跳过当前文件和其他库文件
            if frame_file != current_file and not frame_file.startswith(os.path.dirname(os.__file__)):
                script_path = frame.f_code.co_filename
                if os.path.exists(script_path):
                    try:
                        cwd = os.getcwd()
                        rel_path = os.path.relpath(script_path, cwd)
                        if '..' not in rel_path and len(rel_path) < len(script_path):
                            return rel_path
                    except (ValueError, OSError):
                        pass
                    return script_path
            frame = frame.f_back
    except Exception:
        pass
    
    return None


def _get_permission_error_message(stderr_text: str = '') -> str:
    """
    生成权限错误消息，包含针对 conda 环境和自定义环境变量的解决方案
    
    Args:
        stderr_text: 错误输出文本
        
    Returns:
        格式化的错误消息字符串
    """
    python_path = sys.executable
    
    # 检测相关环境变量
    conda_env = os.environ.get('CONDA_DEFAULT_ENV', '')
    conda_prefix = os.environ.get('CONDA_PREFIX', '')
    pythonpath = os.environ.get('PYTHONPATH', '')
    
    # 收集所有需要保留的环境变量
    env_vars_to_preserve = []
    if conda_prefix:
        env_vars_to_preserve.append(f'CONDA_PREFIX={conda_prefix}')
    if conda_env:
        env_vars_to_preserve.append(f'CONDA_DEFAULT_ENV={conda_env}')
    if pythonpath:
        env_vars_to_preserve.append(f'PYTHONPATH={pythonpath}')
    
    # 获取当前脚本路径
    script_path = _get_current_script_path()
    if not script_path:
        # 如果无法检测，使用默认值
        script_path = 'examples/example_flamegraph_analysis.py'
    
    # 构建解决方案
    solutions = []
    
    # 方案1: 使用 sudo -E 保留所有环境变量（最简单，推荐）
    solutions.append(
        f'1. 使用 sudo -E 保留所有环境变量（推荐）:\n'
        f'   sudo -E {python_path} {script_path}'
    )
    
    # 方案2: 如果有关键环境变量，提供显式保留的方案
    if env_vars_to_preserve:
        env_vars_str = ' '.join(env_vars_to_preserve)
        solutions.append(
            f'2. 或者显式保留关键环境变量:\n'
            f'   sudo env PATH=$PATH {env_vars_str} {python_path} {script_path}'
        )
    else:
        solutions.append(
            f'2. 或者使用 sudo env PATH=$PATH 保留 PATH:\n'
            f'   sudo env PATH=$PATH {python_path} {script_path}'
        )
    
    solutions.append('3. 或者采集当前进程（不指定 pid）')
    
    error_msg = 'py-spy 在 macOS 上需要 root 权限才能采集其他进程。\n'
    if stderr_text:
        error_msg += '错误信息: ' + stderr_text.strip() + '\n\n'
    
    # 如果有环境变量，显示提示信息
    if env_vars_to_preserve:
        error_msg += '检测到以下环境变量需要保留：\n'
        for var in env_vars_to_preserve:
            error_msg += f'  - {var}\n'
        error_msg += '\n'
    
    error_msg += '解决方案：\n' + '\n'.join(solutions)
    
    return error_msg


def _get_python_version_error_message(stderr_text: str, pid: Optional[int] = None) -> str:
    """
    生成 Python 版本检测失败的错误消息
    
    Args:
        stderr_text: 错误输出文本
        pid: 目标进程ID
        
    Returns:
        格式化的错误消息字符串
    """
    error_msg = 'py-spy 无法从目标进程中检测到 Python 版本。\n'
    error_msg += f'错误信息: {stderr_text.strip()}\n\n'
    
    # 检查进程是否存在
    process_exists = False
    is_python_process = False
    
    if pid:
        try:
            # 检查进程是否存在
            if platform.system() == 'Darwin':
                result = subprocess.run(['ps', '-p', str(pid)], 
                                      capture_output=True, timeout=2)
                process_exists = result.returncode == 0
                
                if process_exists:
                    # 检查是否是 Python 进程
                    result = subprocess.run(['ps', '-p', str(pid), '-o', 'comm='], 
                                          capture_output=True, text=True, timeout=2)
                    if result.returncode == 0:
                        comm = result.stdout.strip().lower()
                        is_python_process = 'python' in comm or comm.endswith('python') or comm.endswith('python3')
        except Exception:
            pass
    
    error_msg += '可能的原因：\n'
    
    if not process_exists:
        error_msg += '1. ❌ 目标进程已不存在（进程可能已结束）\n'
        error_msg += '   - 请确认进程ID是否正确\n'
        error_msg += '   - 或者重新启动目标进程后再试\n'
    elif not is_python_process:
        error_msg += '1. ❌ 目标进程不是 Python 进程\n'
        error_msg += f'   - 进程ID {pid} 对应的进程可能不是 Python 程序\n'
        error_msg += '   - py-spy 只能采集 Python 进程的火焰图\n'
        error_msg += '   - 请确认选择了正确的 Python 进程\n'
    else:
        error_msg += '1. ⚠️  py-spy 无法识别该 Python 版本\n'
        error_msg += '   - 可能是 Python 版本过旧或过新\n'
        error_msg += '   - 或者 Python 解释器被修改过\n'
    
    error_msg += '2. ⚠️  权限不足，无法读取进程内存\n'
    error_msg += '   - 即使有 root 权限，某些系统保护机制可能阻止访问\n'
    
    error_msg += '\n解决方案：\n'
    error_msg += '1. 确认目标进程是 Python 进程且仍在运行\n'
    error_msg += '2. 尝试采集当前进程（不指定 pid）\n'
    error_msg += '3. 如果必须采集其他进程，确保：\n'
    error_msg += '   - 使用 sudo 运行（见权限错误提示）\n'
    error_msg += '   - 目标进程是标准的 Python 解释器\n'
    error_msg += '   - Python 版本在 py-spy 支持范围内\n'
    
    return error_msg


class FlamegraphProfilingTool:
    """
    火焰图采集工具类
    
    管理火焰图采集任务的生命周期，包括启动、停止和查询任务。
    """
    
    def __init__(self):
        """初始化火焰图采集工具"""
        self.active_profiling_tasks: Dict[str, Dict] = {}
    
    async def start_flamegraph_profiling(
        self,
        profiling_type: Annotated[str, Doc("采集类型：'python' 使用 py-spy，'perf' 使用 perf")],
        output_path: Annotated[str, Doc("输出火焰图SVG文件的路径")],
        pid: Annotated[Optional[int], Doc("目标进程ID（可选，如果提供则监控该进程）")] = None,
        duration: Annotated[Optional[int], Doc("采集时长（秒），如果为None则持续采集直到手动停止")] = None,
        rate: Annotated[int, Doc("采样频率（Hz），默认100")] = 100,
        task_id: Annotated[Optional[str], Doc("任务ID，用于后续停止采集（可选，自动生成）")] = None
    ) -> Dict:
        """
        启动火焰图采集任务。
        
        支持两种采集方式：
        1. Python (py-spy): 用于分析Python程序的CPU性能
        2. perf: Linux系统级性能分析工具
        
        Args:
            profiling_type: 采集类型，'python' 或 'perf'
            output_path: 输出火焰图SVG文件的路径
            pid: 目标进程ID（可选）
            duration: 采集时长（秒），如果为None则持续采集直到手动停止
            rate: 采样频率（Hz），默认100
            task_id: 任务ID，用于后续停止采集（可选，自动生成）
            
        Returns:
            字典，包含任务ID和状态信息
        """
        try:
            # 在 macOS 上检查权限（需要 root 权限才能采集其他进程）
            # 必须在函数开始时就检查，避免后续操作浪费资源
            if profiling_type == 'python' and platform.system() == 'Darwin' and pid:
                # 检查当前是否有 root 权限
                if os.geteuid() != 0:
                    # 尝试测试采集权限（不实际采集）
                    try:
                        test_cmd = ['py-spy', 'record', '--pid', str(pid), '-d', '0.1', '-o', '/dev/null']
                        test_process = subprocess.Popen(
                            test_cmd,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE
                        )
                        test_process.wait(timeout=3)
                        if test_process.returncode != 0:
                            stderr = test_process.stderr.read().decode('utf-8', errors='ignore')
                            if 'root' in stderr.lower() or 'permission' in stderr.lower() or 'requires root' in stderr.lower():
                                return {
                                    'success': False,
                                    'error': _get_permission_error_message(stderr)
                                }
                    except (subprocess.TimeoutExpired, Exception) as e:
                        # 如果测试失败，继续尝试，让实际采集时再报错
                        logger.warning(f"权限检查失败: {e}")
                        pass
            
            # 生成任务ID
            if not task_id:
                task_id = f"{profiling_type}_{int(time.time())}"
            
            # 检查输出目录是否存在
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            
            # 检查工具是否可用
            if profiling_type == 'python':
                try:
                    subprocess.run(['py-spy', '--version'], 
                                 capture_output=True, check=True, timeout=5)
                except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
                    return {
                        'success': False,
                        'error': 'py-spy 未安装或不可用。请安装: pip install py-spy'
                    }
            elif profiling_type == 'perf':
                try:
                    subprocess.run(['perf', '--version'], 
                                 capture_output=True, check=True, timeout=5)
                except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
                    return {
                        'success': False,
                        'error': 'perf 未安装或不可用。请确保系统已安装 perf 工具'
                    }
            else:
                return {
                    'success': False,
                    'error': f'不支持的采集类型: {profiling_type}。支持的类型: python, perf'
                }
            
            # 启动采集进程
            process = None
            if profiling_type == 'python':
                # py-spy 采集
                cmd = ['py-spy', 'record', '-o', output_path, '-r', str(rate)]
                if pid:
                    cmd.extend(['--pid', str(pid)])
                else:
                    return {
                        'success': False,
                        'error': 'Python采集需要提供进程ID (pid)'
                    }
                
                if duration:
                    cmd.extend(['-d', str(duration)])
                
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                
                # 立即检查进程是否因为权限问题失败（等待一小段时间，作为备用检查）
                await asyncio.sleep(0.5)
                if process.poll() is not None:
                    # 进程已经退出，可能是权限错误
                    stderr_output = None
                    if process.stderr:
                        try:
                            stderr_output = process.stderr.read().decode('utf-8', errors='ignore')
                        except:
                            pass
                    
                    if stderr_output and ('root' in stderr_output.lower() or 'permission' in stderr_output.lower() or 'requires root' in stderr_output.lower()):
                        return {
                            'success': False,
                            'error': _get_permission_error_message(stderr_output)
                        }
                
            elif profiling_type == 'perf':
                # perf 采集
                if not pid:
                    return {
                        'success': False,
                        'error': 'perf采集需要提供进程ID (pid)'
                    }
                
                # perf record 命令
                perf_data_file = output_path.replace('.svg', '.data')
                cmd = ['perf', 'record', '-F', str(rate), '-g', '--pid', str(pid), '-o', perf_data_file]
                
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
            
            # 记录任务信息
            task_info = {
                'task_id': task_id,
                'profiling_type': profiling_type,
                'pid': pid,
                'output_path': output_path,
                'process': process,
                'start_time': time.time(),
                'duration': duration,
                'rate': rate
            }
            self.active_profiling_tasks[task_id] = task_info
            
            # 如果有持续时间限制，启动异步定时任务
            if duration:
                async def stop_after_duration():
                    await asyncio.sleep(duration)
                    if task_id in self.active_profiling_tasks:
                        await self.stop_flamegraph_profiling(task_id)
                
                # 创建后台任务
                asyncio.create_task(stop_after_duration())
            
            return {
                'success': True,
                'task_id': task_id,
                'profiling_type': profiling_type,
                'pid': pid,
                'output_path': output_path,
                'status': 'running',
                'message': f'采集任务已启动，任务ID: {task_id}'
            }
            
        except Exception as e:
            logger.error(f"启动火焰图采集失败: {e}")
            return {
                'success': False,
                'error': f'启动采集失败: {str(e)}'
            }
    
    async def stop_flamegraph_profiling(
        self,
        task_id: Annotated[str, Doc("要停止的任务ID")]
    ) -> Dict:
        """
        停止正在进行的火焰图采集任务并生成火焰图。
        
        Args:
            task_id: 要停止的任务ID
            
        Returns:
            字典，包含生成的火焰图路径和状态信息
        """
        try:
            if task_id not in self.active_profiling_tasks:
                return {
                    'success': False,
                    'error': f'任务ID {task_id} 不存在或已完成'
                }
            
            task_info = self.active_profiling_tasks[task_id]
            process = task_info['process']
            profiling_type = task_info['profiling_type']
            output_path = task_info['output_path']
            
            # 停止采集进程
            try:
                if process.poll() is None:  # 进程仍在运行
                    if profiling_type == 'python':
                        # py-spy 会自动在收到信号时停止并生成SVG
                        # 注意：如果使用了 -d 参数，py-spy 会自动在时间到后停止
                        # 如果没有 -d 参数，需要发送信号停止
                        if task_info.get('duration') is None:
                            # 手动停止模式：发送 SIGINT 信号
                            process.send_signal(signal.SIGINT)
                        
                        # 等待进程完成
                        try:
                            process.wait(timeout=30)  # 增加超时时间，给py-spy足够时间生成文件
                        except subprocess.TimeoutExpired:
                            logger.warning("py-spy进程超时，强制终止")
                            process.kill()
                            process.wait()
                        
                        # 检查进程退出码和错误输出
                        stderr_output = None
                        if process.stderr:
                            try:
                                stderr_output = process.stderr.read().decode('utf-8', errors='ignore')
                                if stderr_output:
                                    logger.warning(f"py-spy错误输出: {stderr_output[:500]}")
                            except:
                                pass
                        
                        if process.returncode != 0 and process.returncode is not None:
                            logger.warning(f"py-spy进程退出码: {process.returncode}")
                        
                        # 检查是否是权限错误
                        if stderr_output and ('root' in stderr_output.lower() or 'permission' in stderr_output.lower() or 'requires root' in stderr_output.lower()):
                            # 权限错误，直接返回错误信息
                            return {
                                'success': False,
                                'task_id': task_id,
                                'status': 'failed',
                                'output_path': output_path,
                                'error': _get_permission_error_message(stderr_output)
                            }
                        
                        # 检查是否是 Python 版本检测失败
                        if stderr_output and ('failed to find python version' in stderr_output.lower() or 'python version' in stderr_output.lower()):
                            pid = task_info.get('pid')
                            return {
                                'success': False,
                                'task_id': task_id,
                                'status': 'failed',
                                'output_path': output_path,
                                'error': _get_python_version_error_message(stderr_output, pid)
                            }
                        
                        # 等待文件生成（最多等待5秒）
                        for i in range(50):  # 50次 * 0.1秒 = 5秒
                            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                                break
                            await asyncio.sleep(0.1)
                        
                        # 如果文件仍然不存在，记录警告
                        if not os.path.exists(output_path):
                            logger.error(f"py-spy未生成输出文件: {output_path}")
                            if stderr_output:
                                logger.error(f"py-spy完整错误输出: {stderr_output}")
                    elif profiling_type == 'perf':
                        # perf 需要先停止记录，然后生成火焰图
                        process.send_signal(signal.SIGINT)
                        process.wait(timeout=10)
                        
                        # 生成火焰图
                        perf_data_file = output_path.replace('.svg', '.data')
                        if os.path.exists(perf_data_file):
                            # 使用 perf script 和 flamegraph 生成SVG
                            # 注意：需要安装 flamegraph 工具
                            try:
                                # 检查 flamegraph 是否可用
                                subprocess.run(['flamegraph', '--version'], 
                                             capture_output=True, check=True, timeout=5)
                                
                                # 生成火焰图
                                perf_script_cmd = ['perf', 'script', '-i', perf_data_file]
                                flamegraph_cmd = ['flamegraph']
                                
                                perf_script = subprocess.Popen(
                                    perf_script_cmd,
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE
                                )
                                
                                flamegraph = subprocess.Popen(
                                    flamegraph_cmd,
                                    stdin=perf_script.stdout,
                                    stdout=open(output_path, 'w'),
                                    stderr=subprocess.PIPE
                                )
                                
                                perf_script.stdout.close()
                                flamegraph.wait(timeout=30)
                                perf_script.wait(timeout=10)
                                
                                # 清理临时文件
                                if os.path.exists(perf_data_file):
                                    os.remove(perf_data_file)
                                    
                            except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
                                # 如果 flamegraph 不可用，返回 perf data 文件路径
                                return {
                                    'success': True,
                                    'task_id': task_id,
                                    'status': 'stopped',
                                    'perf_data_file': perf_data_file,
                                    'output_path': output_path,
                                    'message': 'perf采集已停止，但需要手动生成火焰图。请使用: perf script -i <data_file> | flamegraph > <output.svg>',
                                    'warning': 'flamegraph 工具未安装，无法自动生成SVG'
                                }
            except subprocess.TimeoutExpired:
                # 强制终止
                process.kill()
                process.wait()
            
            # 检查输出文件是否生成
            if os.path.exists(output_path):
                file_size = os.path.getsize(output_path)
                if file_size > 0:
                    result = {
                        'success': True,
                        'task_id': task_id,
                        'status': 'completed',
                        'output_path': output_path,
                        'file_size': file_size,
                        'message': f'采集已完成，火焰图已生成: {output_path}'
                    }
                else:
                    # 文件存在但为空，可能是生成失败
                    result = {
                        'success': False,
                        'task_id': task_id,
                        'status': 'failed',
                        'output_path': output_path,
                        'error': f'文件已生成但为空（{file_size} 字节），可能是采集过程中出错'
                    }
            else:
                # 文件不存在，检查进程错误输出
                error_msg = '采集已停止，但输出文件未找到'
                stderr_output = None
                if process.stderr:
                    try:
                        # 尝试读取 stderr（可能已经被读取过）
                        stderr_output = process.stderr.read().decode('utf-8', errors='ignore')
                        if not stderr_output:
                            # 如果读取失败，尝试从进程对象获取
                            try:
                                stderr_output = process.stderr.read().decode('utf-8', errors='ignore')
                            except:
                                pass
                    except:
                        pass
                
                if stderr_output:
                    # 检查是否是权限错误
                    if 'root' in stderr_output.lower() or 'permission' in stderr_output.lower() or 'requires root' in stderr_output.lower():
                        error_msg = _get_permission_error_message(stderr_output)
                    # 检查是否是 Python 版本检测失败
                    elif 'failed to find python version' in stderr_output.lower() or 'python version' in stderr_output.lower():
                        # 获取任务信息以检查进程状态
                        pid = task_info.get('pid')
                        error_msg = _get_python_version_error_message(stderr_output, pid)
                    else:
                        error_msg += f'\n错误信息: {stderr_output[:500]}'  # 限制长度
                
                result = {
                    'success': False,
                    'task_id': task_id,
                    'status': 'failed',
                    'output_path': output_path,
                    'error': error_msg
                }
            
            # 清理任务记录
            del self.active_profiling_tasks[task_id]
            
            return result
            
        except Exception as e:
            logger.error(f"停止火焰图采集失败: {e}")
            # 清理任务记录
            if task_id in self.active_profiling_tasks:
                del self.active_profiling_tasks[task_id]
            
            return {
                'success': False,
                'error': f'停止采集失败: {str(e)}'
            }
    
    async def list_profiling_tasks(self) -> Dict:
        """
        查看当前正在进行的火焰图采集任务列表。
        
        Returns:
            字典，包含所有活跃任务的列表
        """
        try:
            tasks = []
            for task_id, task_info in self.active_profiling_tasks.items():
                process = task_info['process']
                is_running = process.poll() is None if process else False
                
                tasks.append({
                    'task_id': task_id,
                    'profiling_type': task_info['profiling_type'],
                    'pid': task_info['pid'],
                    'output_path': task_info['output_path'],
                    'status': 'running' if is_running else 'stopped',
                    'start_time': task_info['start_time'],
                    'duration': task_info.get('duration'),
                    'rate': task_info.get('rate')
                })
            
            return {
                'success': True,
                'tasks': tasks,
                'count': len(tasks)
            }
            
        except Exception as e:
            logger.error(f"获取任务列表失败: {e}")
            return {
                'success': False,
                'error': f'获取任务列表失败: {str(e)}'
            }

async def _fetch_flamegraph_svg(profile_path: str) -> Optional[str]:
    """
    读取火焰图SVG文件内容。
    
    Args:
        profile_path: 火焰图SVG文件的路径
        
    Returns:
        SVG文件内容字符串，如果文件不存在或读取失败则返回None
    """
    try:
        with open(profile_path, 'r', encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError:
        logger.info(f"错误：火焰图文件 {profile_path} 不存在")
        return None
    except UnicodeDecodeError:
        logger.info(f"错误：火焰图文件 {profile_path} 编码解析失败")
        return None

def _parse_flamegraph_svg(svg_content: str) -> Dict:
    """
    解析火焰图SVG，提取性能数据并构建层级关系。
    
    解析SVG文件中的rect元素和title元素，提取函数名称、采样数、百分比等信息，
    并构建函数调用的层级关系。支持正向和反向火焰图。
    
    Args:
        svg_content: SVG文件内容字符串
        
    Returns:
        包含以下键的字典：
        - total_functions: 总函数数量
        - total_samples: 总采样数
        - functions: 函数列表，每个函数包含id、name、x、y、width、height、samples、percentage、level等信息
        - functions_by_level: 按层级分组的函数字典
        - max_level: 最大层级
        - min_level: 最小层级
        - svg_width: SVG宽度
        - svg_height: SVG高度
        - root_function: 根节点函数（通常是'all'函数）
        - is_inverted: 是否为反向火焰图
        如果解析失败，返回包含'error'键的字典
    """
    try:
        # 解析SVG
        root = ET.fromstring(svg_content)

        # 提取所有rect元素，支持不同的命名空间
        rects = []
        # 尝试不同的命名空间查找方式
        for ns in ['', '{http://www.w3.org/2000/svg}']:
            found_rects = root.findall(f'.//{ns}rect')
            if found_rects:
                rects = found_rects
                break

        # 如果还是没找到，尝试直接查找所有rect元素
        if not rects:
            for elem in root.iter():
                if elem.tag.endswith('rect'):
                    rects.append(elem)

        logger.info(f"解析到 {len(rects)} 个rect元素")

        # 构建函数调用数据
        functions = []

        # 获取SVG尺寸
        svg_width = float(root.get('width', 1440))
        svg_height = float(root.get('height', 1686))

        # 收集所有有效的Y坐标来计算层级
        valid_y_coords = []
        for rect in rects:
            try:
                # 处理百分比格式的坐标值
                y_str = rect.get('y', '0')
                height_str = rect.get('height', '0')
                width_str = rect.get('width', '0')

                # 移除百分比符号并转换为浮点数
                y = float(y_str.rstrip('%'))
                height = float(height_str.rstrip('%'))
                width = float(width_str.rstrip('%'))

                # 跳过背景rect和无效rect
                if width >= 99.0 and height >= 99.0:
                    continue
                if height <= 0 or width <= 0:
                    continue

                valid_y_coords.append(y)
            except (ValueError, TypeError):
                continue

        # 对Y坐标排序，用于计算层级
        valid_y_coords = sorted(set(valid_y_coords))

        # 检测火焰图方向：通过查找"all"函数的位置来判断
        # 如果all函数在Y坐标较小的位置，说明是正向火焰图（all在顶部）
        # 如果all函数在Y坐标较大的位置，说明是反向火焰图（all在底部）
        all_function_y = None
        for rect in rects:
            try:
                # 查找title元素
                title_element = None
                for ns in ['', '{http://www.w3.org/2000/svg}']:
                    title_element = rect.find(f'.//{ns}title')
                    if title_element is not None:
                        break

                if title_element is not None and title_element.text:
                    title_text = title_element.text.strip()
                    if title_text.startswith('all ') or title_text == 'all':
                        y_str = rect.get('y', '0')
                        all_function_y = float(y_str.rstrip('%'))
                        break
            except (ValueError, TypeError):
                continue

        # 判断火焰图方向
        is_inverted = False  # 默认正向（all在顶部）
        if all_function_y is not None and valid_y_coords:
            # 如果all函数的Y坐标在中位数以上，说明是反向火焰图
            median_y = sorted(valid_y_coords)[len(valid_y_coords) // 2]
            is_inverted = all_function_y > median_y
            logger.info(f"检测到火焰图方向: {'反向(all在底部)' if is_inverted else '正向(all在顶部)'}, all_y={all_function_y}, median_y={median_y}")

        # 创建Y坐标到层级的映射
        y_to_level = {}
        if is_inverted:
            # 反向火焰图：Y值越大层级越低（L1在底部）
            valid_y_coords_sorted = sorted(set(valid_y_coords), reverse=True)
            for i, y in enumerate(valid_y_coords_sorted):
                y_to_level[y] = i + 1
        else:
            # 正向火焰图：Y值越小层级越低（L1在顶部）
            valid_y_coords_sorted = sorted(set(valid_y_coords))
            for i, y in enumerate(valid_y_coords_sorted):
                y_to_level[y] = i + 1

        # 构建元素到父元素的映射
        parent_map = {}
        for parent in root.iter():
            for child in parent:
                parent_map[child] = parent

        # 解析每个rect及其对应的title
        for i, rect in enumerate(rects):
            try:
                # 处理百分比格式的坐标值
                x_str = rect.get('x', '0')
                y_str = rect.get('y', '0')
                width_str = rect.get('width', '0')
                height_str = rect.get('height', '0')

                # 移除百分比符号并转换为浮点数
                x = float(x_str.rstrip('%'))
                y = float(y_str.rstrip('%'))
                width = float(width_str.rstrip('%'))
                height = float(height_str.rstrip('%'))

                # 跳过背景rect（通常是第一个，覆盖整个SVG）
                if width >= 99.0 and height >= 99.0:  # 调整阈值以适应百分比格式
                    continue

                # 跳过无效的rect
                if height <= 0 or width <= 0:
                    continue

                # 初始化变量
                function_name = ""
                samples = 0
                percentage = 0.0
                title_text = ""

                # 获取原始的x和width值（如果存在fg:x和fg:w属性）
                original_x = rect.get('fg:x')
                original_width = rect.get('fg:w')
                if original_x is not None:
                    try:
                        original_x = int(original_x)
                    except (ValueError, TypeError):
                        original_x = None
                if original_width is not None:
                    try:
                        original_width = int(original_width)
                        samples = original_width  # fg:w通常表示采样数
                    except (ValueError, TypeError):
                        original_width = None

                # 查找与当前rect关联的title元素
                # 方法1：查找rect的直接子元素title
                title_element = None
                for ns in ['', '{http://www.w3.org/2000/svg}']:
                    title_element = rect.find(f'.//{ns}title')
                    if title_element is not None:
                        break

                # 方法2：如果没找到子title，查找父元素中的title
                if title_element is None:
                    parent = parent_map.get(rect)
                    if parent is not None:
                        # 查找父元素下的所有title
                        for ns in ['', '{http://www.w3.org/2000/svg}']:
                            parent_titles = parent.findall(f'.//{ns}title')
                            if parent_titles:
                                # 尝试找到位置最接近的title
                                for title in parent_titles:
                                    if title.text and title.text.strip():
                                        title_element = title
                                        break
                                if title_element is not None:
                                    break

                # 方法3：如果还是没找到，查找同级的g元素中的title
                if title_element is None:
                    parent = parent_map.get(rect)
                    if parent is not None:
                        # 查找同级g元素
                        for ns in ['', '{http://www.w3.org/2000/svg}']:
                            siblings = parent.findall(f'.//{ns}g')
                            if siblings:
                                for sibling in siblings:
                                    title = sibling.find(f'.//{ns}title')
                                    if title is not None and title.text and title.text.strip():
                                        # 检查这个g元素是否包含当前rect的坐标范围
                                        g_rects = sibling.findall(f'.//{ns}rect')
                                        for g_rect in g_rects:
                                            try:
                                                g_x_str = g_rect.get('x', '0')
                                                g_y_str = g_rect.get('y', '0')
                                                g_width_str = g_rect.get('width', '0')
                                                g_height_str = g_rect.get('height', '0')

                                                g_x = float(g_x_str.rstrip('%'))
                                                g_y = float(g_y_str.rstrip('%'))
                                                g_width = float(g_width_str.rstrip('%'))
                                                g_height = float(g_height_str.rstrip('%'))

                                                # 检查坐标是否匹配（允许小的误差）
                                                if (abs(g_x - x) < 0.01 and abs(g_y - y) < 0.01 and
                                                        abs(g_width - width) < 0.01 and abs(g_height - height) < 0.01):
                                                    title_element = title
                                                    break
                                            except (ValueError, TypeError):
                                                continue
                                        if title_element is not None:
                                            break
                                if title_element is not None:
                                    break

                # 解析title信息
                if title_element is not None:
                    title_text = title_element.text or ""
                    if title_text:
                        # 解析title格式，支持多种格式：
                        # 格式1：'函数名 (samples数, 百分比%)'
                        # 格式2：'函数名 (samples数 samples, 百分比%)'
                        # 例如：'C2_CompilerThre (3,019 samples, 73.22%)'
                        # 例如：'_build_request (derisk/util/api_utils.py:174) (4 samples, 0.12%)'
                        if '(' in title_text and ')' in title_text:
                            # 找到最后一个括号对，这通常包含采样信息
                            last_paren_start = title_text.rfind('(')
                            last_paren_end = title_text.rfind(')')

                            if last_paren_start < last_paren_end:
                                function_name = title_text[:last_paren_start].strip()
                                info_part = title_text[last_paren_start+1:last_paren_end]

                                try:
                                    # 解析samples数和百分比
                                    # 匹配格式：3,019 samples, 73.22% 或 4 samples, 0.12%
                                    samples_match = re.search(r'([\d,]+)\s*samples?', info_part)
                                    if samples_match:
                                        samples_str = samples_match.group(1).replace(',', '')
                                        samples = int(samples_str)
                                    elif original_width is not None:
                                        # 如果title中没有找到samples，使用fg:w的值
                                        samples = original_width

                                    # 解析百分比
                                    percentage_match = re.search(r'([\d.]+)%', info_part)
                                    if percentage_match:
                                        percentage = float(percentage_match.group(1))

                                except (ValueError, AttributeError) as e:
                                    logger.warning(f"解析title信息失败: {title_text}, 错误: {e}")
                                    # 如果解析失败，使用默认值或fg:w的值
                                    if original_width is not None:
                                        samples = original_width
                                    else:
                                        samples = 1
                                    percentage = width  # 在百分比格式中，width本身就是百分比
                            else:
                                # 没有有效的括号信息
                                function_name = title_text.strip()
                                if original_width is not None:
                                    samples = original_width
                                else:
                                    samples = 1
                                percentage = width  # 在百分比格式中，width本身就是百分比
                        else:
                            # 没有括号信息的title，直接作为函数名
                            function_name = title_text.strip()
                            if original_width is not None:
                                samples = original_width
                            else:
                                samples = 1
                            percentage = width  # 在百分比格式中，width本身就是百分比

                # 如果没有获取到函数名，尝试从text元素获取
                if not function_name:
                    # 查找与rect关联的text元素
                    text_element = None
                    for ns in ['', '{http://www.w3.org/2000/svg}']:
                        text_element = rect.find(f'.//{ns}text')
                        if text_element is not None:
                            break

                    if text_element is None:
                        parent = parent_map.get(rect)
                        if parent is not None:
                            for ns in ['', '{http://www.w3.org/2000/svg}']:
                                text_element = parent.find(f'.//{ns}text')
                                if text_element is not None:
                                    break

                    if text_element is not None and text_element.text:
                        function_name = text_element.text.strip()

                # 如果还是没有函数名，跳过这个元素
                if not function_name or function_name == "None":
                    continue

                # 修复层级计算 - 使用Y坐标映射到层级
                level = y_to_level.get(y, 1)

                functions.append({
                    'id': f"func_{i}",
                    'name': function_name,
                    'x': x,
                    'y': y,
                    'width': width,
                    'height': height,
                    'samples': samples,
                    'percentage': percentage,
                    'level': level,
                    'title': title_text,
                    'index': i,
                    'original_x': original_x,
                    'original_width': original_width
                })

            except (ValueError, TypeError) as e:
                logger.warning(f"解析rect元素 {i} 失败: {e}")
                continue

        logger.info(f"成功解析 {len(functions)} 个有效函数")

        # 计算总采样数
        total_samples = max(f['samples'] for f in functions) if functions else 0

        # 找到根节点（all函数，通常samples最多）
        root_function = None
        for func in functions:
            if func['name'] == 'all' or func['samples'] == total_samples:
                root_function = func
                total_samples = func['samples']
                break

        # 构建层级关系
        functions_by_level = defaultdict(list)
        for func in functions:
            functions_by_level[func['level']].append(func)

        return {
            'total_functions': len(functions),
            'total_samples': total_samples,
            'functions': functions,
            'functions_by_level': dict(functions_by_level),
            'max_level': max(functions_by_level.keys()) if functions_by_level else 0,
            'min_level': min(functions_by_level.keys()) if functions_by_level else 0,
            'svg_width': svg_width,
            'svg_height': svg_height,
            'root_function': root_function,
            'is_inverted': is_inverted
        }

    except ET.ParseError as e:
        logger.error(f"SVG解析失败: {e}")
        return {'error': f'SVG解析失败: {str(e)}'}


def _build_hierarchical_view(parsed_data: Dict, max_functions_per_level: int = 5, limit: int = 50) -> List[str]:
    """
    构建层级视图，从L1（底层）到最高层显示主要函数。
    
    将解析后的火焰图数据转换为可读的层级视图，按层级显示CPU占用最高的函数。
    合并相同名称的函数，并按采样数排序。
    
    Args:
        parsed_data: 解析后的火焰图数据字典
        max_functions_per_level: 每层最多显示的函数数，默认为5
        limit: 限制返回的层级数量，默认为50层
        
    Returns:
        层级视图字符串列表，格式为"L{level} {function_name} ({samples} samples, {percentage}%)"
        列表已按从高层到低层排序（倒序）
    """
    if 'error' in parsed_data:
        return []

    functions_by_level = parsed_data.get('functions_by_level', {})
    if not functions_by_level:
        return []

    result = []

    # 从L1开始到最高层
    for level in sorted(functions_by_level.keys()):
        level_functions = functions_by_level[level]

        # 合并相同名称的函数
        function_stats = defaultdict(lambda: {
            'samples': 0,
            'percentage': 0.0,
            'count': 0
        })

        for func in level_functions:
            name = func['name']
            # 过滤无意义的函数名
            if (name.startswith('unknown_func_') or
                    name.startswith('func_') or
                    name.startswith('parse_error_') or
                    len(name) <= 2 or
                    name == "None"):
                continue

            function_stats[name]['samples'] = max(function_stats[name]['samples'], func['samples'])
            function_stats[name]['percentage'] = max(function_stats[name]['percentage'], func['percentage'])
            function_stats[name]['count'] += 1

        # 转换为列表并排序（按samples降序）
        level_result = []
        for name, stats in function_stats.items():
            level_result.append({
                'name': name,
                'samples': stats['samples'],
                'percentage': stats['percentage']
            })

        level_result.sort(key=lambda x: x['samples'], reverse=True)

        # 构建该层级的显示字符串
        if level_result:
            # 如果是L1层且有all函数，只显示all
            if level == 1 and any(f['name'] == 'all' for f in level_result):
                all_func = next(f for f in level_result if f['name'] == 'all')
                result.append(
                    f"L{level} {all_func['name']} ({all_func['samples']} samples, {all_func['percentage']:.2f}%)")
            else:
                # 显示前N个函数
                func_strs = []
                for func in level_result[:max_functions_per_level]:
                    func_strs.append(f"{func['name']} ({func['samples']} samples, {func['percentage']:.2f}%)")
                result.append(f"L{level} {', '.join(func_strs)}")

        # 如果达到限制层级数量，停止添加
        if len(result) >= limit:
            break

    # 倒序显示（从高层到低层）
    result.reverse()

    return result



@tool(description="获取火焰图概览，按层级显示CPU占用最高的函数（从L1根节点向上展示）")
async def flamegraph_overview(
    profile_path: Annotated[str, Doc("火焰图SVG文件的路径，用于获取对应的火焰图数据")],
    max_functions_per_level: Annotated[int, Doc("每层最多显示的函数数，默认为5")] = 5,
    limit: Annotated[int, Doc("限制返回的层级数量，默认为50层（L1-L50）")] = 50
) -> str:
    """
    获取火焰图概览，按层级显示CPU占用最高的函数。
    
    解析火焰图SVG文件，提取函数调用层级信息，并按层级显示CPU占用最高的函数。
    从L1根节点（通常是'all'函数）开始向上展示，帮助快速了解性能瓶颈的分布。
    
    Args:
        profile_path: 火焰图SVG文件的路径
        max_functions_per_level: 每层最多显示的函数数，默认为5
        limit: 限制返回的层级数量，默认为50层
        
    Returns:
        JSON格式字符串，包含：
        - success: 是否成功
        - profile_path: 火焰图文件路径
        - summary: 摘要信息（总函数数、总采样数、总层级数、显示的层级数）
        - hierarchical_view: 层级视图列表，格式为"L{level} {function_name} ({samples} samples, {percentage}%)"
        - description: 结果描述
        如果失败，返回包含error字段的JSON
    """
    try:
        # 获取火焰图SVG数据
        svg_content = await _fetch_flamegraph_svg(profile_path)

        # 解析SVG数据
        parsed_data = _parse_flamegraph_svg(svg_content)

        if 'error' in parsed_data:
            return json.dumps({
                'success': False,
                'error': parsed_data['error']
            }, ensure_ascii=False, indent=2)

        # 构建层级视图，限制层级数量
        hierarchical_view = _build_hierarchical_view(parsed_data, max_functions_per_level, limit)

        # 构建返回结果
        result = {
            'success': True,
            'profile_path': profile_path,
            'summary': {
                'total_functions': parsed_data['total_functions'],
                'total_samples': parsed_data['total_samples'],
                'total_levels': parsed_data['max_level'],
                'displayed_levels': min(limit, parsed_data['max_level']),
            },
            'hierarchical_view': hierarchical_view,
            'description': f"火焰图层级视图（显示前{min(limit, parsed_data['max_level'])}层，从高层到低层，L1是根节点）"
        }

        return json.dumps(result, ensure_ascii=False, indent=2)

    except Exception as e:
        logger.error(f"火焰图概览分析失败: {e}")
        return json.dumps({
            'success': False,
            'error': f'火焰图概览分析失败: {str(e)}'
        }, ensure_ascii=False, indent=2)


@tool(description="深入分析指定函数，展示该函数及其上层（子函数）的CPU占用情况，支持精确匹配和模糊匹配")
async def flamegraph_drill_down(
    profile_path: Annotated[str, Doc("火焰图SVG文件的路径，用于获取对应的火焰图数据")],
    function_name: Annotated[str, Doc(
        "要分析的函数名称，通常从概览结果中选择感兴趣的函数。支持精确匹配和模糊匹配（当fuzzy_match=True时）"
    )],
    fuzzy_match: Annotated[bool, Doc(
        "是否使用模糊匹配，默认为False。当为True时，会匹配包含function_name的所有函数"
    )] = False,
    levels_to_show: Annotated[int, Doc("显示多少层子函数，默认为10层")] = 10
) -> str:
    """
    深入分析指定函数，展示该函数及其上层（子函数）的CPU占用情况。
    
    从火焰图中找到指定的函数，然后向上分析其子函数调用链，帮助理解该函数的
    性能瓶颈来源。支持精确匹配和模糊匹配两种模式。
    
    Args:
        profile_path: 火焰图SVG文件的路径
        function_name: 要分析的函数名称，通常从概览结果中选择感兴趣的函数
        fuzzy_match: 是否使用模糊匹配，默认为False。当为True时，会匹配包含function_name的所有函数
        levels_to_show: 显示多少层子函数，默认为10层
        
    Returns:
        JSON格式字符串，包含：
        - success: 是否成功
        - profile_path: 火焰图文件路径
        - target_function: 目标函数信息（名称、层级、采样数、百分比、出现次数、匹配模式）
        - hierarchical_view: 层级视图列表，从目标函数开始向上展示子函数调用链
        - description: 结果描述
        如果失败，返回包含error字段的JSON
    """
    try:
        # 获取火焰图SVG数据
        svg_content = await _fetch_flamegraph_svg(profile_path)

        # 解析SVG数据
        parsed_data = _parse_flamegraph_svg(svg_content)

        if 'error' in parsed_data:
            return json.dumps({
                'success': False,
                'error': parsed_data['error']
            }, ensure_ascii=False, indent=2)

        # 查找指定的函数 - 支持精确匹配和模糊匹配
        if fuzzy_match:
            # 模糊匹配：查找包含function_name的所有函数
            target_functions = [f for f in parsed_data['functions'] if function_name.lower() in f['name'].lower()]
        else:
            # 精确匹配
            target_functions = [f for f in parsed_data['functions'] if f['name'] == function_name]

        if not target_functions:
            match_type = "模糊匹配" if fuzzy_match else "精确匹配"
            return json.dumps({
                'success': False,
                'error': f'未找到函数: {function_name} ({match_type})'
            }, ensure_ascii=False, indent=2)

        # 选择采样数最多的实例
        target_function = max(target_functions, key=lambda x: x['samples'])
        target_level = target_function['level']
        target_x = target_function['x']
        target_width = target_function['width']
        target_x_end = target_x + target_width

        # 构建该函数的调用链视图 - 只包含目标函数及其子函数
        hierarchical_view = []

        # 添加目标函数本身
        match_info = f" [FUZZY MATCH]" if fuzzy_match else ""
        hierarchical_view.append(
            f"L{target_level} [TARGET]{match_info} {target_function['name']} ({target_function['samples']} samples, {target_function['percentage']:.2f}%)"
        )

        # 如果是模糊匹配且找到多个函数，显示所有匹配的函数
        if fuzzy_match and len(target_functions) > 1:
            other_matches = [f for f in target_functions if f != target_function]
            other_matches.sort(key=lambda x: x['samples'], reverse=True)

            hierarchical_view.append("--- 其他匹配的函数 ---")
            for i, func in enumerate(other_matches[:5]):  # 最多显示5个其他匹配
                hierarchical_view.append(
                    f"L{func['level']} [MATCH {i + 2}] {func['name']} ({func['samples']} samples, {func['percentage']:.2f}%)"
                )
            if len(other_matches) > 5:
                hierarchical_view.append(f"... 还有 {len(other_matches) - 5} 个匹配的函数")
            hierarchical_view.append("--- 主要函数的子函数调用链 ---")

        # 只查找上层函数（子函数），不查找下层函数
        for level_offset in range(1, levels_to_show + 1):
            child_level = target_level + level_offset
            if child_level > parsed_data['max_level']:
                break

            # 查找在目标函数范围内的子函数
            child_functions = []
            for func in parsed_data['functions']:
                if func['level'] == child_level:
                    # 子函数应该在目标函数的X坐标范围内
                    func_x = func['x']
                    func_x_end = func['x'] + func['width']
                    # 检查是否在目标函数的范围内（有重叠）
                    if (func_x >= target_x and func_x < target_x_end) or \
                            (func_x_end > target_x and func_x_end <= target_x_end) or \
                            (func_x <= target_x and func_x_end >= target_x_end):
                        child_functions.append(func)

            if child_functions:
                # 合并相同名称的函数
                function_stats = defaultdict(lambda: {
                    'samples': 0,
                    'percentage': 0.0
                })

                for func in child_functions:
                    name = func['name']
                    function_stats[name]['samples'] = max(function_stats[name]['samples'], func['samples'])
                    function_stats[name]['percentage'] = max(function_stats[name]['percentage'], func['percentage'])

                # 排序并构建显示字符串
                level_funcs = []
                for name, stats in sorted(function_stats.items(), key=lambda x: x[1]['samples'], reverse=True):
                    level_funcs.append(f"{name} ({stats['samples']} samples, {stats['percentage']:.2f}%)")

                # 只显示前5个最重要的函数
                hierarchical_view.append(f"L{child_level} {', '.join(level_funcs[:5])}")
            else:
                # 如果没有找到子函数，停止继续查找
                break

        # 构建返回结果
        result = {
            'success': True,
            'profile_path': profile_path,
            'target_function': {
                'name': function_name,
                'actual_name': target_function['name'],
                'level': target_level,
                'samples': target_function['samples'],
                'percentage': target_function['percentage'],
                'occurrences': len(target_functions),
                'fuzzy_match': fuzzy_match
            },
            'hierarchical_view': hierarchical_view,
            'description': f"函数 {function_name} 的子函数调用链分析（{'模糊匹配' if fuzzy_match else '精确匹配'}，从L{target_level}层开始向上{len([h for h in hierarchical_view if h.startswith('L') and '[TARGET]' in h or h.startswith('L') and '[TARGET]' not in h and '[MATCH' not in h]) - 1}层）"
        }

        return json.dumps(result, ensure_ascii=False, indent=2)

    except Exception as e:
        logger.error(f"火焰图钻取分析失败: {e}")
        return json.dumps({
            'success': False,
            'error': f'火焰图钻取分析失败: {str(e)}'
        }, ensure_ascii=False, indent=2)




if __name__ == '__main__':
    import asyncio

    # 测试火焰图分析工具
    test_profile_id = "./pilot/data/f162eff6-330b-4388-9a31-bf8777dcbd60.svg"

    print("=== 火焰图CPU性能分析工具测试 ===")

    # 1. 测试概览分析
    print("\n1. 测试火焰图概览:")
    overview_result = asyncio.run(flamegraph_overview(test_profile_id, 5, 50))
    print(overview_result)

    # 2. 测试精确匹配钻取分析
    print("\n2. 测试精确匹配钻取分析 - C2_CompilerThre:")
    drill_down_result = asyncio.run(flamegraph_drill_down(test_profile_id, "C2_CompilerThre", False, 5))
    print(drill_down_result)

    # 3. 测试模糊匹配钻取分析
    print("\n3. 测试模糊匹配钻取分析 - 搜索包含'Compiler'的函数:")
    drill_down_result2 = asyncio.run(flamegraph_drill_down(test_profile_id, "Compiler", True, 5))
    print(drill_down_result2)

    # 4. 测试另一个模糊匹配
    print("\n4. 测试模糊匹配钻取分析 - 搜索包含'Load'的函数:")
    drill_down_result3 = asyncio.run(flamegraph_drill_down(test_profile_id, "Load", True, 5))
    print(drill_down_result3)

    # 5. 测试精确匹配 - LoadNode::Value
    print("\n5. 测试精确匹配钻取分析 - LoadNode::Value:")
    drill_down_result4 = asyncio.run(flamegraph_drill_down(test_profile_id, "LoadNode::Value", False, 5))
    print(drill_down_result4)

    # 6. 测试Java方法的模糊匹配
    print("\n6. 测试模糊匹配钻取分析 - 搜索包含'doIntercept'的函数:")
    drill_down_result5 = asyncio.run(flamegraph_drill_down(test_profile_id, "doIntercept", True, 5))
    print(drill_down_result5)

    print("\n测试完成！")
