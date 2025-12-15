"""
模拟 CPU 性能瓶颈的测试脚本

这个脚本用于测试火焰图采集和分析功能。它包含多个明显的性能瓶颈：
1. CPU 密集型计算（斐波那契数列）
2. 递归调用
3. 低效的算法实现
4. 嵌套循环

运行方式：
    python examples/simulate_cpu_bottleneck.py

然后使用 py-spy 采集这个进程的火焰图进行分析。
"""

import time
import random
import math
import os
from typing import List, Optional


def fibonacci_recursive(n: int) -> int:
    """
    低效的递归实现斐波那契数列（性能瓶颈1）
    这是一个典型的 CPU 密集型函数，会在火焰图中显示为热点
    """
    if n <= 1:
        return n
    return fibonacci_recursive(n - 1) + fibonacci_recursive(n - 2)


def fibonacci_inefficient(n: int) -> int:
    """
    低效的迭代实现（性能瓶颈2）
    使用不必要的重复计算
    """
    if n <= 1:
        return n
    
    result = 0
    for i in range(n):
        for j in range(i):
            result += j * math.sqrt(i)
    
    return result % 1000


def bubble_sort_inefficient(arr: List[int]) -> List[int]:
    """
    低效的冒泡排序（性能瓶颈3）
    对大数据集进行排序会产生明显的 CPU 占用
    """
    arr = arr.copy()
    n = len(arr)
    
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    
    return arr


def matrix_multiplication_naive(a: List[List[float]], b: List[List[float]]) -> List[List[float]]:
    """
    低效的矩阵乘法（性能瓶颈4）
    使用 O(n^3) 的朴素算法
    """
    n = len(a)
    result = [[0.0] * n for _ in range(n)]
    
    for i in range(n):
        for j in range(n):
            for k in range(n):
                result[i][j] += a[i][k] * b[k][j]
    
    return result


def cpu_intensive_task():
    """
    主要的 CPU 密集型任务
    这个函数会调用多个性能瓶颈函数
    """
    # 瓶颈1: 递归斐波那契（最耗时）
    print("执行递归斐波那契计算...")
    for i in range(30, 35):
        result = fibonacci_recursive(i)
        print(f"fibonacci_recursive({i}) = {result}")
    
    # 瓶颈2: 低效斐波那契
    print("执行低效斐波那契计算...")
    for i in range(100, 150, 10):
        result = fibonacci_inefficient(i)
        print(f"fibonacci_inefficient({i}) = {result}")
    
    # 瓶颈3: 冒泡排序
    print("执行冒泡排序...")
    large_array = [random.randint(1, 1000) for _ in range(500)]
    sorted_array = bubble_sort_inefficient(large_array)
    print(f"排序完成，前10个元素: {sorted_array[:10]}")
    
    # 瓶颈4: 矩阵乘法
    print("执行矩阵乘法...")
    size = 50
    matrix_a = [[random.random() for _ in range(size)] for _ in range(size)]
    matrix_b = [[random.random() for _ in range(size)] for _ in range(size)]
    result_matrix = matrix_multiplication_naive(matrix_a, matrix_b)
    print(f"矩阵乘法完成，结果矩阵大小: {len(result_matrix)}x{len(result_matrix[0])}")


def simulate_workload(duration_seconds: Optional[int] = None):
    """
    模拟工作负载，持续运行指定时间或无限运行
    
    Args:
        duration_seconds: 运行时长（秒），如果为 None 则无限运行直到手动中断
    """
    if duration_seconds is None:
        print("开始模拟 CPU 性能瓶颈，将持续运行直到手动中断（Ctrl+C）...")
    else:
        print(f"开始模拟 CPU 性能瓶颈，将持续运行 {duration_seconds} 秒...")
    print(f"进程ID: {os.getpid()}")
    print("=" * 60)
    
    start_time = time.time()
    iteration = 0
    
    try:
        while True:
            # 如果指定了时长，检查是否超时
            if duration_seconds is not None:
                if time.time() - start_time >= duration_seconds:
                    break
            
            iteration += 1
            elapsed = time.time() - start_time
            
            if duration_seconds is None:
                print(f"\n[迭代 {iteration}] 已运行 {elapsed:.1f} 秒（无限运行，按 Ctrl+C 停止）")
            else:
                remaining = duration_seconds - elapsed
                print(f"\n[迭代 {iteration}] 已运行 {elapsed:.1f} 秒，剩余 {remaining:.1f} 秒")
            
            # 执行 CPU 密集型任务
            cpu_intensive_task()
            
            # 短暂休息，避免完全占用 CPU
            time.sleep(0.5)
            
    except KeyboardInterrupt:
        print("\n\n收到中断信号，正在退出...")
    
    print("\n" + "=" * 60)
    print("模拟完成！")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='模拟 CPU 性能瓶颈用于火焰图分析')
    parser.add_argument(
        '--duration',
        type=int,
        default=None,
        help='运行时长（秒），默认无限运行直到手动中断（Ctrl+C）'
    )
    parser.add_argument(
        '--pid',
        action='store_true',
        help='显示进程ID后退出（用于测试）'
    )
    
    args = parser.parse_args()
    
    if args.pid:
        print(f"进程ID: {os.getpid()}")
        return
    
    # 运行模拟工作负载
    simulate_workload(args.duration)


if __name__ == "__main__":
    main()

