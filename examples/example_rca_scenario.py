"""
Example RCA Scenarios

This script demonstrates multiple RCA scenarios from the OpenRCA dataset.
Each scenario represents a real fault case that needs root cause analysis.
"""

import sys
import os
import dotenv

dotenv.load_dotenv()

from langchain.messages import HumanMessage
from src.agents.rca_agents import create_rca_deep_agent


def setup_agent():
    """Initialize the RCA agent with Langfuse tracing."""
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
        
        rca_agent = create_rca_deep_agent(
            model=model,
            config={"dataset_path": "datasets/OpenRCA/Bank"}
        )
        
        return rca_agent, langfuse_handler
        
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


def scenario_1(agent, langfuse_handler):
    """
    Scenario 1: 2021年3月4日 14:30-15:00 故障时间定位
    
    任务：确定根因的具体发生时间
    """
    print("\n" + "=" * 80)
    print("场景 1: 故障时间定位")
    print("=" * 80)
    
    question = """
2021年3月4日，在00:30至01:00的时间范围内，系统中检测到一次故障。根因发生的确切时间未知。请确定根因的具体发生时间，具体组件以及根本原因。
    """
    print(f"\n问题:\n{question}")
    print("\n" + "-" * 80)
    print("开始分析...")
    print("-" * 80 + "\n")
    
    try:
        result = agent.invoke(
            input=dict(
                messages=[
                    HumanMessage(content=question)
                ]
            ),
            config={"callbacks": [langfuse_handler]}
        )
        print("\n" + "=" * 80)
        print("分析结果:")
        print("=" * 80)
        print(result.get("output", "No output"))
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


def scenario_2(agent, langfuse_handler):
    """
    Scenario 2: 2021年3月4日 18:00-18:30 根因组件和原因分析
    
    任务：找出导致故障的根本原因组件及其根本原因
    """
    print("\n" + "=" * 80)
    print("场景 2: 根因组件和原因分析")
    print("=" * 80)
    
    question = """
2021年3月4日18:00至18:30之间，系统中出现了一次故障。导致此次故障的具体组件尚不清楚，故障发生的原因也尚未确定。你的任务是找出导致此次故障的根本原因组件及其根本原因。
    """
    
    print(f"\n问题:\n{question}")
    print("\n" + "-" * 80)
    print("开始分析...")
    print("-" * 80 + "\n")
    
    try:
        result = agent.invoke(
            input=dict(
                messages=[
                    HumanMessage(content=question)
                ]
            ),
            config={"callbacks": [langfuse_handler]}
        )
        print("\n" + "=" * 80)
        print("分析结果:")
        print("=" * 80)
        print(result.get("output", "No output"))
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


def scenario_3(agent, langfuse_handler):
    """
    Scenario 3: 2021年3月6日 18:30-19:00 完整根因分析
    
    任务：确定根本原因组件、根本原因发生的具体时间以及根本原因
    """
    print("\n" + "=" * 80)
    print("场景 3: 完整根因分析")
    print("=" * 80)
    
    question = """
2021年3月6日18:30至19:00之间发生了故障。然而，目前尚不清楚故障的根本原因组件、根本原因发生的确切时间以及故障的根本原因。你的任务是确定根本原因组件、根本原因发生的具体时间以及根本原因的原因。
    """
    
    print(f"\n问题:\n{question}")
    print("\n" + "-" * 80)
    print("开始分析...")
    print("-" * 80 + "\n")
    
    try:
        result = agent.invoke(
            input=dict(
                messages=[
                    HumanMessage(content=question)
                ]
            ),
            config={"callbacks": [langfuse_handler]}
        )
        print("\n" + "=" * 80)
        print("分析结果:")
        print("=" * 80)
        print(result.get("output", "No output"))
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


def scenario_4(agent, langfuse_handler):
    """
    Scenario 4: 2021年3月25日 19:00-19:30 多故障根因分析
    
    任务：确定两次故障的根本原因组件、根本原因发生的具体时间以及根本原因
    """
    print("\n" + "=" * 80)
    print("场景 4: 多故障根因分析")
    print("=" * 80)
    
    question = """
在2021年3月25日19:00至19:30的指定时间段内，系统中检测到两次故障。导致这些故障的具体组件、发生的确切时间以及根本原因尚不清楚。你的任务是确定故障的根本原因组件、根本原因发生的具体时间以及根本原因。
    """
    
    print(f"\n问题:\n{question}")
    print("\n" + "-" * 80)
    print("开始分析...")
    print("-" * 80 + "\n")
    
    try:
        result = agent.invoke(
            input=dict(
                messages=[
                    HumanMessage(content=question)
                ]
            ),
            config={"callbacks": [langfuse_handler]}
        )
        print("\n" + "=" * 80)
        print("分析结果:")
        print("=" * 80)
        print(result.get("output", "No output"))
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Run all RCA scenarios in sequence."""
    print("=" * 80)
    print("RCA Agent - OpenRCA 数据集故障分析场景")
    print("=" * 80)
    
    # Initialize agent
    print("\n" + "=" * 80)
    print("初始化 RCA Agent...")
    print("=" * 80)
    
    agent, langfuse_handler = setup_agent()
    if agent is None or langfuse_handler is None:
        print("\n初始化失败，退出。")
        return
    
    print("\n✓ Agent 初始化成功")
    print("✓ Langfuse tracing 已启用")
    
    # Run scenarios in sequence
    print("\n" + "=" * 80)
    print("开始执行场景分析...")
    print("=" * 80)
    
    # Scenario 1: 故障时间定位
    scenario_1(agent, langfuse_handler)
    
    # # Scenario 2: 根因组件和原因分析
    # scenario_2(agent, langfuse_handler)
    
    # # Scenario 3: 完整根因分析
    # scenario_3(agent, langfuse_handler)
    
    # # Scenario 4: 多故障根因分析
    # scenario_4(agent, langfuse_handler)
    
    print("\n" + "=" * 80)
    print("所有场景分析完成")
    print("=" * 80)
    print("\n提示: 请访问 Langfuse 控制台查看详细的执行追踪信息")


if __name__ == "__main__":
    main()
