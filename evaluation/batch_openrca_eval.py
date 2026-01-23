import os
import sys
import csv
import uuid
import time
import dotenv
import logging

dotenv.load_dotenv(override=True)

from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Tuple, Optional, Dict, Any
import argparse
import concurrent.futures
from langfuse import get_client
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Setup a file logger that doesn't interfere with Rich
def _setup_file_logger(log_file: Path):
    logger = logging.getLogger("eval_logger")
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(log_file)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger

_LF_CLIENT = None
def _get_langfuse_client():
    global _LF_CLIENT
    if _LF_CLIENT is None:
        _LF_CLIENT = get_client()
    return _LF_CLIENT


def _check_langfuse_config() -> None:
    required = ["LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_HOST"]
    missing = [k for k in required if not os.getenv(k)]
    if missing:
        raise RuntimeError(f"Missing Langfuse config: {', '.join(missing)}")
    client = _get_langfuse_client()
    if client is None:
        raise RuntimeError("Langfuse client initialization failed")


def _build_question(start_time: str, end_time: str, num_faults: int = 1) -> str:
    start_display = str(start_time).replace("T", " ")
    end_display = str(end_time).replace("T", " ")
    if num_faults == 1:
        return (
            f"在{start_display}至{end_display}的时间范围内，系统中检测到一次故障。然而，目前尚不清楚故障的根本原因组件、根本原因发生的确切时间以及故障的根本原因。"
            "你的任务是确定根本原因组件、根本原因发生的具体时间以及根本原因。"
        )
    return (
        f"在{start_display}至{end_display}的指定时间段内，系统中检测到{num_faults}次故障。导致这些故障的具体组件、发生的确切时间以及根本原因尚不清楚。"
        "你的任务是确定故障的根本原因组件、根本原因发生的具体时间以及根本原因。"
    )


def _quantize_half_hour_window(dt: datetime) -> Tuple[str, str]:
    """
    Given a datetime, return the start/end of the containing 30-minute window.
    e.g. 10:14 -> 10:00-10:30
    """
    minute = dt.minute
    if minute < 30:
        start = dt.replace(minute=0, second=0, microsecond=0)
        end = start + timedelta(minutes=30)
    else:
        start = dt.replace(minute=30, second=0, microsecond=0)
        end = start + timedelta(minutes=30)
    return start.isoformat(), end.isoformat()

def _load_all_records_grouped(dataset_path: Optional[str] = None):
    """
    Loads records and groups them by time window.
    Returns:
        all_records: List of dicts (with 'start', 'end', 'raw')
        window_to_records: Dict[(start, end), List[dict]]
    """
    # Fix path to root evaluation-record.csv
    if dataset_path:
        base = Path(dataset_path)
        if base.is_file():
            path = base
        else:
            path = base / "evaluation-record.csv"
    else:
        # Default to checking root project directory first, then datasets
        root_path = Path(__file__).resolve().parent.parent / "evaluation-record.csv"
        if root_path.exists():
            path = root_path
        else:
             # Fallback to datasets folder
             path = Path(__file__).resolve().parent.parent / "datasets" / "OpenRCA" / "Bank" / "evaluation-record.csv"

    if not path.exists():
        raise FileNotFoundError(f"evaluation-record.csv not found: {path}")

    all_records = []
    window_to_records = defaultdict(list)

    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
             # Prefer 'datetime' field: build aligned half-hour window around it
            dt_raw = r.get("datetime")
            start = None
            end = None
            center = None
            
            if dt_raw:
                try:
                    # Handle numeric epoch seconds/milliseconds
                    if isinstance(dt_raw, (int, float)) or (isinstance(dt_raw, str) and dt_raw.replace('.', '', 1).isdigit()):
                        val = float(dt_raw)
                        # ms vs s heuristic
                        if val > 1e12:
                            val = val / 1000.0
                        center = datetime.fromtimestamp(val)
                    else:
                        # Try ISO-like string
                        s = str(dt_raw).strip().replace('Z', '')
                        try:
                            center = datetime.fromisoformat(s)
                        except Exception:
                            # Fallback: treat as timestamp seconds in string
                            center = datetime.fromtimestamp(float(s))
                    start, end = _quantize_half_hour_window(center)
                except Exception:
                    pass

            if not start or not end:
                # Try to find explicit start/end time columns
                start = (
                    r.get("start_utc") or r.get("start_time") or r.get("fault_start_time") or r.get("time_start") or r.get("start")
                )
                end = (
                    r.get("end_utc") or r.get("end_time") or r.get("fault_end_time") or r.get("time_end") or r.get("end")
                )
            
            if not start or not end:
                 # Fallback: if only one time is available
                t = r.get("time") or r.get("fault_time") or r.get("timestamp")
                if t:
                    try:
                        if isinstance(t, (int, float)) or (isinstance(t, str) and t.replace('.', '', 1).isdigit()):
                            val = float(t)
                            if val > 1e12:
                                val = val / 1000.0
                            center = datetime.fromtimestamp(val)
                        else:
                            s = str(t).strip().replace('Z', '')
                            try:
                                center = datetime.fromisoformat(s)
                            except Exception:
                                center = datetime.fromtimestamp(float(s))
                        start, end = _quantize_half_hour_window(center)
                    except Exception:
                        pass
            
            record_entry = {
                "start": start,
                "end": end,
                "raw": r
            }
            all_records.append(record_entry)
            
            if start and end:
                window_key = (start, end)
                window_to_records[window_key].append(record_entry)
                
    return all_records, window_to_records

def _load_queries(
    dataset_path: Optional[str] = None, limit: int = 20
) -> List[Tuple[str, str]]:
    
    all_records, window_to_records = _load_all_records_grouped(dataset_path)

    # Second pass: Build questions based on limit
    rows: List[Tuple[str, str]] = []
    idx_counter = 0
    
    for record in all_records:
        start = record["start"]
        end = record["end"]
        
        if not start or not end:
            continue
            
        window_key = (start, end)
        records_in_window = window_to_records[window_key]
        count = len(records_in_window)
        
        question = _build_question(start, end, num_faults=count)
        idx_counter += 1
        rows.append((str(idx_counter), question))
        
        if len(rows) >= limit:
            break
            
    if len(rows) < limit:
        pass
        
    return rows


def _create_agent(dataset_path: str = "datasets/OpenRCA/Bank"):
    from langfuse.langchain import CallbackHandler
    from src.agents.rca_agents import create_rca_deep_agent
    from src.config.loader import load_config
    from src.utils.llm_factory import init_langchain_models_from_llm_config

    # Load config and init model
    config_data = load_config()
    models, default_model = init_langchain_models_from_llm_config(config_data.llm)
    
    if not default_model:
        raise ValueError("No valid LLM model found in config")
        
    model = default_model
    config = {
        "dataset_path": dataset_path,
        "metric_analyzer": {
            "dataset_path": dataset_path,
            "domain_adapter": "openrca",
            "dataloader": "openrca",
        },
        "log_analyzer": {
            "dataset_path": dataset_path,
            "domain_adapter": "openrca",
            "dataloader": "openrca",
        },
        "trace_analyzer": {
            "dataset_path": dataset_path,
            "domain_adapter": "openrca",
            "dataloader": "openrca",
        },
    }
    agent = create_rca_deep_agent(model=model, config=config)
    handler = CallbackHandler()
    return agent, handler


def _create_baseline_agent():
    from langfuse.langchain import CallbackHandler
    from src.openrca_baseline.rca_agent import RCA_Agent
    from src.openrca_baseline.prompt import agent_prompt
    from src.openrca_baseline.prompt import basic_prompt_Bank

    agent = RCA_Agent(agent_prompt=agent_prompt, basic_prompt=basic_prompt_Bank)
    handler = CallbackHandler()
    return agent, handler


def _invoke_agent(
    agent, handler, text: str, trace_context: Dict[str, Any], logger=None
) -> Tuple[str, float]:
    from langchain.messages import HumanMessage
    client = _get_langfuse_client()

    if logger:
        logger.info(f"[_invoke_agent] Starting invocation. Trace context: {trace_context}")
        logger.info(f"[_invoke_agent] Input text (first 100 chars): {text[:100]}...")

    start = time.time()
    with client.start_as_current_span(name="process-request", trace_context=trace_context):
        if logger:
            logger.info(f"[_invoke_agent] Calling agent.invoke()...")
        result = agent.invoke(
            {"messages": [HumanMessage(content=text)]},
            config={"callbacks": [handler]},
        )
        if logger:
            logger.info(f"[_invoke_agent] agent.invoke() returned.")
    end = time.time()
    latency = (end - start) * 1000.0
    content = ""
    if isinstance(result, dict) and "messages" in result and result["messages"]:
        last = result["messages"][-1]
        content = getattr(last, "content", "") or str(last)
    else:
        content = str(result)
    
    if logger:
        logger.info(f"[_invoke_agent] Invocation finished in {latency:.2f}ms")
    
    return content, latency


def _invoke_baseline_agent(
    agent, handler, text: str, trace_context: Dict[str, Any], logger=None
) -> Tuple[str, float]:
    client = _get_langfuse_client()
    
    if logger:
        logger.info(f"[_invoke_baseline_agent] Starting invocation. Trace context: {trace_context}")
    
    start = time.time()
    with client.start_as_current_span(name="process-request", trace_context=trace_context):
        if logger:
            logger.info(f"[_invoke_baseline_agent] Calling agent.run()...")
        
        # RCA_Agent.run returns (prediction, trajectory, prompt)
        # We need to pass callbacks to trace internal calls
        prediction, _, _ = agent.run(
            text, 
            logger=logger if logger else logging.getLogger("dummy"), 
            callbacks=[handler]
        )
        
        if logger:
            logger.info(f"[_invoke_baseline_agent] agent.run() returned.")
            
    end = time.time()
    latency = (end - start) * 1000.0
    
    return str(prediction), latency



def _log_extraction_error(log_path: Path, error_msg: str, prompt: str, raw_response: str):
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        entry = (
            f"[{timestamp}] Extraction Error\n"
            f"Reason: {error_msg}\n"
            f"Prompt:\n{prompt}\n"
            f"Raw Response:\n{raw_response}\n"
            f"{'-'*80}\n"
        )
        with log_path.open("a", encoding="utf-8") as f:
            f.write(entry)
    except Exception:
        pass


def _extract_fields_llm(
    text: str, error_log_path: Optional[Path] = None
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    from langchain_deepseek import ChatDeepSeek

    model = ChatDeepSeek(model="deepseek-chat")
    
    valid_components = [
        "apache01", "apache02", "IG01", "IG02", "MG01", "MG02",
        "Mysql01", "Mysql02", "Redis01", "Redis02",
        "Tomcat01", "Tomcat02", "Tomcat03", "Tomcat04"
    ]
    valid_reasons = [
        "high CPU usage", "high disk I/O read usage", "high disk space usage",
        "high JVM CPU load", "high memory usage", "JVM Out of Memory (OOM) Heap",
        "network latency", "network packet loss"
    ]

    prompt = (
        "请从以下文本中提取并返回JSON格式列表："
        '[{"component":"","reason":"","time":""}, ...]。\n'
        f"component必须是以下之一：{', '.join(valid_components)}。\n"
        f"reason必须是以下之一：{', '.join(valid_reasons)}。\n"
        "time使用原文中的日期时间字符串。文本如下：\n"
        f"{text}"
    )
    try:
        resp = model.invoke(prompt)
        content = (
            getattr(resp, "content", "") if hasattr(resp, "content") else str(resp)
        )
        data = []
        try:
            import json
            
            # Cleanup markdown code blocks if present
            cleaned_content = content.strip()
            if cleaned_content.startswith("```json"):
                cleaned_content = cleaned_content[7:]
            elif cleaned_content.startswith("```"):
                cleaned_content = cleaned_content[3:]
            if cleaned_content.endswith("```"):
                cleaned_content = cleaned_content[:-3]
            cleaned_content = cleaned_content.strip()

            parsed = json.loads(cleaned_content)
            if isinstance(parsed, dict):
                data = [parsed]
            elif isinstance(parsed, list):
                data = parsed
            else:
                data = []
        except Exception as e:
            if error_log_path:
                _log_extraction_error(error_log_path, f"JSON Parse Error: {str(e)}", prompt, content)
            data = []
        
        # Serialize to JSON string if list is not empty, else empty string
        # To handle multiple root causes in CSV, we can just dump the list as a JSON string
        if not data:
            if error_log_path:
                 # Also log if we got empty data (might be useful if model returned something valid but we failed to parse or it was empty)
                 # Only log if it wasn't already logged as exception above.
                 # Actually, if exception happened, data is [], so we logged it.
                 # If no exception but data is empty/invalid format (e.g. not dict/list), we didn't log specific error.
                 pass
            return None, None, None
        
        if len(data) == 1:
            item = data[0]
            return item.get("component"), item.get("reason"), item.get("time")
        
        # Multiple root causes: return concatenated strings or JSON dump
        # For CSV readability, maybe concatenated is better? 
        # But JSON dump is more precise. Let's use JSON dump for "component" and "reason" if multiple.
        # Actually, let's keep it simple: if multiple, join with ';'
        
        comps = [str(d.get("component", "")) for d in data]
        reasons = [str(d.get("reason", "")) for d in data]
        times = [str(d.get("time", "")) for d in data]
        
        return "; ".join(comps), "; ".join(reasons), "; ".join(times)

    except Exception as e:
        if error_log_path:
            _log_extraction_error(error_log_path, f"Invocation/General Error: {str(e)}", prompt, "N/A")
        return None, None, None


def _fetch_langfuse_metrics(
    trace_id: str,
) -> Tuple[Optional[int], Optional[int], Optional[int], Optional[int], Optional[float]]:
    client = _get_langfuse_client()
    if client is None:
        return None, None, None, None, None
    try:
        trace_obj = client.api.trace.get(trace_id)
        # Direct access to TraceWithFullDetails attributes
        latency_ms = (
            trace_obj.latency * 1000 if trace_obj.latency else None
        )  # Convert to milliseconds

        # Aggregate usage from observations
        input_tokens = 0
        input_cache_tokens = 0
        output_tokens = 0
        total_tokens = 0

        for obs in trace_obj.observations:
            usage_details = obs.usage_details or {}
            # DeepSeek cache usage handling
            input_val = usage_details.get("input", 0)
            cache_val = usage_details.get("input_cache_read", 0)
            
            input_tokens += (input_val + cache_val)
            input_cache_tokens += cache_val
            output_tokens += usage_details.get("output", 0)
            total_tokens += usage_details.get("total", 0)

        return (
            input_tokens if input_tokens > 0 else None,
            input_cache_tokens if input_cache_tokens > 0 else None,
            output_tokens if output_tokens > 0 else None,
            total_tokens if total_tokens > 0 else None,
            latency_ms,
        )
    except Exception:
        return None, None, None, None, None


def _append_csv(row: Dict[str, Any], out_csv: Path) -> None:
    exists = out_csv.exists()
    with out_csv.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "task_index",
                "run_index",
                "trace_id",
                "component",
                "reason",
                "time",
                "input_tokens",
                "input_cache_tokens",
                "output_tokens",
                "total_tokens",
                "latency_ms",
                "saved_path",
            ],
        )
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _load_existing_progress(csv_path: Path) -> Dict[str, set]:
    """
    Load existing progress from CSV.
    Returns a dict mapping task_index to a set of completed run_indices.
    """
    if not csv_path.exists():
        return {}
    
    completed_map = {}
    try:
        with csv_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                task_idx = row.get("task_index")
                run_idx_str = row.get("run_index")
                if task_idx and run_idx_str:
                    try:
                        run_idx = int(run_idx_str)
                        if task_idx not in completed_map:
                            completed_map[task_idx] = set()
                        completed_map[task_idx].add(run_idx)
                    except ValueError:
                        pass
    except Exception:
        pass
    return completed_map


def main():
    _check_langfuse_config()
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-trace", type=str, default=None)
    parser.add_argument(
        "--mode", 
        type=str, 
        choices=["rca", "baseline", "eval"], 
        required=True,
        help="Evaluation mode: 'rca', 'baseline', or 'eval' (evaluate existing results)"
    )
    parser.add_argument(
        "--results-file",
        type=str,
        default=None,
        help="Path to results CSV file (required for mode='eval')"
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="Path to dataset root or evaluation-record.csv"
    )
    parser.add_argument(
        "--changes",
        type=str,
        default="-",
        help="Description of changes for this run (e.g. 'Optimized prompt')"
    )
    args = parser.parse_args()

    # Special handling for 'eval' mode
    if args.mode == "eval":
        console = Console()
        if not args.results_file:
            console.print("[red]Error: --results-file is required for mode='eval'[/red]")
            sys.exit(1)
            
        csv_path = Path(args.results_file)
        if not csv_path.exists():
            console.print(f"[red]Error: File not found: {csv_path}[/red]")
            sys.exit(1)

        # Setup logger
        logger = logging.getLogger("eval_logger")
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        logger.addHandler(handler)

        console.print(f"[bold cyan]Running Independent Evaluation[/bold cyan]")
        console.print(f"[dim]Results File[/dim]: {csv_path}")

        console.print("[bold cyan]Refreshing Langfuse Data...[/bold cyan]")
        try:
            _refresh_langfuse_data(csv_path, logger)
        except Exception as e:
            logger.error(f"Failed to refresh Langfuse data: {e}")
            console.print(f"[red]Failed to refresh Langfuse data: {e}[/red]")

        console.print("[bold cyan]Running Evaluation Metrics...[/bold cyan]")
        try:
            _run_evaluation_metrics(csv_path, args.dataset_path, logger, changes=args.changes)
        except Exception as e:
            logger.error(f"Failed to run evaluation metrics: {e}")
            console.print(f"[red]Failed to run evaluation metrics: {e}[/red]")
        
        return

    if args.verify_trace:
        # (Verification logic omitted for brevity, keeping existing)
        in_toks, in_cache_toks, out_toks, tot_toks, lf_latency = _fetch_langfuse_metrics(
            args.verify_trace
        )
        if all(v is None for v in [in_toks, in_cache_toks, out_toks, tot_toks, lf_latency]):
            client = _get_langfuse_client()
            obj = client.api.trace.get(args.verify_trace) if client else None

            def _to_dict(o):
                if isinstance(o, dict):
                    return o
                for attr in ["model_dump", "dict", "to_dict"]:
                    fn = getattr(o, attr, None)
                    if callable(fn):
                        try:
                            return fn()
                        except Exception:
                            pass
                try:
                    return o.__dict__
                except Exception:
                    return {"value": o}

            data = _to_dict(obj) or {}
            if isinstance(data, dict) and "data" in data and data["data"] is not None:
                data = _to_dict(data["data"]) or {}
            print(f"raw_keys={list(data.keys())}")
        else:
            print(
                f"tokens_in={in_toks}, tokens_in_cache={in_cache_toks}, tokens_out={out_toks}, tokens_total={tot_toks}, latency_ms={lf_latency}"
            )
        return

    console = Console()
    
    # 1. Setup timestamped directory
    timestamp = datetime.now().strftime("%Y%m%d")
    eval_root = Path.cwd() / "evaluation" / "history" / timestamp
    eval_root.mkdir(parents=True, exist_ok=True)
    
    # Setup paths
    save_dir = eval_root / "md"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    out_csv_rca = eval_root / "rca_results.csv"
    out_csv_baseline = eval_root / "baseline_results.csv"
    
    log_file = eval_root / "execution.log"
    extraction_errors_file = eval_root / "extraction_errors.log"
    logger = _setup_file_logger(log_file)
    
    logger.info(f"Starting evaluation. Results will be saved to {eval_root}")
    console.print(f"[bold cyan]OpenRCA Batch Evaluation[/bold cyan]")
    console.print(f"[dim]Mode[/dim]: {args.mode}")
    console.print(f"[dim]Results Directory[/dim]: {eval_root}")

    # 2. Load queries
    queries = _load_queries(limit=62)
    logger.info(f"Loaded {len(queries)} queries.")
    
    # Check existing progress
    target_csv_path = out_csv_rca if args.mode == "rca" else out_csv_baseline
    completed_progress = _load_existing_progress(target_csv_path)
    if completed_progress:
        logger.info(f"Found existing progress: {completed_progress}")
        console.print(f"[dim]Found existing progress, resuming...[/dim]")

    # 3. Create Agents
    agent_rca = None
    agent_baseline = None

    if args.mode == "rca":
        try:
            agent_rca, _ = _create_agent()
            logger.info("RCA Agent initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize RCA Agent: {e}")
            raise
    
    if args.mode == "baseline":
        try:
            agent_baseline, _ = _create_baseline_agent()
            logger.info("Baseline Agent initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize Baseline Agent: {e}")
            raise

    client = _get_langfuse_client()

    def _run_task(task_idx: str, run: int, instruction: str, mode: str) -> Dict[str, Any]:
        from langfuse.langchain import CallbackHandler
        
        # Determine agent and handler
        if mode == "rca":
            agent = agent_rca
            invoke_func = _invoke_agent
        else:
            agent = agent_baseline
            invoke_func = _invoke_baseline_agent
            
        trace_id = client.create_trace_id(seed=f"{task_idx}-{run}-{mode}-{uuid.uuid4().hex}")
        handler_local = CallbackHandler()
        
        try:
            content, latency = invoke_func(
                agent, handler_local, instruction, {"trace_id": trace_id}, logger=logger
            )
        except Exception as e:
            logger.error(f"Error in task {task_idx}-{run} ({mode}): {e}")
            raise e
            
        unique = uuid.uuid4().hex[:6]
        filename = f"rca_result_{task_idx}_{run}_{mode}_{unique}.md"
        save_path = save_dir / filename
        with save_path.open("w", encoding="utf-8") as f:
            f.write(content)
            
        comp, reason, t = _extract_fields_llm(content, error_log_path=extraction_errors_file)
        in_toks, in_cache_toks, out_toks, tot_toks, lf_latency = _fetch_langfuse_metrics(trace_id)
        
        return {
            "task_index": task_idx,
            "run_index": run,
            "mode": mode,
            "trace_id": trace_id,
            "component": comp or "",
            "reason": reason or "",
            "time": t or "",
            "input_tokens": in_toks if in_toks is not None else "",
            "input_cache_tokens": in_cache_toks if in_cache_toks is not None else "",
            "output_tokens": out_toks if out_toks is not None else "",
            "total_tokens": tot_toks if tot_toks is not None else "",
            "latency_ms": lf_latency if lf_latency is not None else round(latency, 2),
            "saved_path": filename,
        }

    # Calculate remaining tasks
    tasks_to_run = []
    skipped_count = 0
    
    for task_idx, instruction in queries:
        completed_runs_set = completed_progress.get(task_idx, set())
        
        # We want total 1 run per task
        for run in range(1, 2):
            if run in completed_runs_set:
                skipped_count += 1
                continue
            tasks_to_run.append((task_idx, run, instruction))

    total_tasks = len(tasks_to_run)
    console.print(f"[dim]Total Tasks to Run[/dim]: {total_tasks} (Skipped {skipped_count} completed runs)")
    
    futures = []
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=False,
    )
    
    with progress:
        task_id = progress.add_task("Processing", total=total_tasks)
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            for task_idx, run, instruction in tasks_to_run:
                futures.append(executor.submit(_run_task, task_idx, run, instruction, args.mode))
                     
            for fut in concurrent.futures.as_completed(futures):
                try:
                    row = fut.result()
                    target_csv = out_csv_rca if row["mode"] == "rca" else out_csv_baseline
                    
                    row_to_write = row.copy()
                    del row_to_write["mode"]
                    
                    _append_csv(row_to_write, target_csv)
                    progress.advance(task_id, 1)
                    
                    color = "green" if row["mode"] == "rca" else "blue"
                    console.print(
                        f"[{color}]✓[/] Task {row['task_index']}-{row['run_index']} ({row['mode']}) saved"
                    )
                except Exception as e:
                    progress.advance(task_id, 1)
                    console.print(f"[red]✗[/red] Task Failed: {e}")
                    logger.error(f"Task failed with error: {e}")

    # 5. Post-processing: Refresh Langfuse Data & Run Evaluation
    target_csv = out_csv_rca if args.mode == "rca" else out_csv_baseline
    if target_csv.exists():
        console.print("[bold cyan]Refreshing Langfuse Data...[/bold cyan]")
        try:
            _refresh_langfuse_data(target_csv, logger)
        except Exception as e:
            logger.error(f"Failed to refresh Langfuse data: {e}")
            console.print(f"[red]Failed to refresh Langfuse data: {e}[/red]")

        console.print("[bold cyan]Running Evaluation Metrics...[/bold cyan]")
        try:
            _run_evaluation_metrics(target_csv, args.dataset_path, logger, changes=args.changes)
        except Exception as e:
            logger.error(f"Failed to run evaluation metrics: {e}")
            console.print(f"[red]Failed to run evaluation metrics: {e}[/red]")

def _refresh_langfuse_data(csv_path: Path, logger=None):
    client = _get_langfuse_client()
    if not client:
        return

    # Read all rows
    rows = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames

    updated_rows = []
    changes_count = 0
    
    # Using a simple loop instead of rich progress for simplicity here, 
    # or could reuse the existing console if passed.
    for row in rows:
        tid = row.get("trace_id")
        if tid:
            try:
                # Attempt to fetch trace. 
                # Note: The Langfuse Python SDK (v2+) usually allows fetching via client.fetch_trace() or similar.
                # If the method doesn't exist, we'll catch the error.
                # We try to look for `get_trace` or `fetch_trace`.
                trace_obj = None
                if hasattr(client, "get_trace"):
                    trace_obj = client.get_trace(tid)
                elif hasattr(client, "fetch_trace"):
                    trace_obj = client.fetch_trace(tid)
                
                if trace_obj:
                    # Depending on the object structure (Pydantic model or dict)
                    # We look for totalCost, usage, or observations.
                    # This part is tricky without knowing the exact SDK version.
                    # We'll try to extract what we can.
                    
                    # Try to access as object attributes first
                    observations = getattr(trace_obj, "observations", [])
                    
                    # Recalculate usage from observations if possible
                    input_tokens = 0
                    output_tokens = 0
                    total_tokens = 0
                    latency = getattr(trace_obj, "latency", 0) or 0
                    
                    # If observations is a list of objects
                    found_obs = False
                    for obs in observations:
                        # Check for usage in observation
                        usage = getattr(obs, "usage", None)
                        if usage:
                            found_obs = True
                            input_tokens += getattr(usage, "input", 0) or getattr(usage, "promptTokens", 0) or 0
                            output_tokens += getattr(usage, "output", 0) or getattr(usage, "completionTokens", 0) or 0
                            total_tokens += getattr(usage, "total", 0) or getattr(usage, "totalTokens", 0) or 0
                    
                    if found_obs:
                        row["input_tokens"] = input_tokens
                        row["output_tokens"] = output_tokens
                        row["total_tokens"] = total_tokens
                    
                    # Update latency if available (convert to ms)
                    if latency:
                        row["latency_ms"] = latency * 1000.0
                    
                    changes_count += 1
            except Exception as e:
                if logger:
                    logger.warning(f"Error refreshing trace {tid}: {e}")
        
        updated_rows.append(row)

    if changes_count > 0:
        # Write back
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(updated_rows)
        if logger:
            logger.info(f"Refreshed {changes_count} rows with Langfuse data.")

def _run_evaluation_metrics(results_csv_path: Path, dataset_path: Optional[str] = None, logger=None, changes: str = "-"):
    # 1. Load Ground Truth Mapping
    all_records, window_to_records = _load_all_records_grouped(dataset_path)
    
    # Rebuild task_id -> ground_truth list
    task_id_to_gt = {}
    idx_counter = 0
    for record in all_records:
        start = record["start"]
        end = record["end"]
        if not start or not end: continue
        window_key = (start, end)
        records_in_window = window_to_records[window_key]
        idx_counter += 1
        
        # Determine the Ground Truths for this window
        # Each record in records_in_window is a dict with "raw" data
        gts = []
        for r in records_in_window:
            raw = r["raw"]
            # Extract Time
            dt_raw = raw.get("datetime")
            gt_time = None
            if dt_raw:
                try:
                    if isinstance(dt_raw, (int, float)) or (isinstance(dt_raw, str) and dt_raw.replace('.', '', 1).isdigit()):
                         val = float(dt_raw)
                         if val > 1e12: val /= 1000.0
                         gt_time = datetime.fromtimestamp(val)
                    else:
                        s = str(dt_raw).strip().replace('Z', '')
                        try: gt_time = datetime.fromisoformat(s)
                        except: gt_time = datetime.fromtimestamp(float(s))
                except: pass
            
            gts.append({
                "component": raw.get("component", "").strip(),
                "reason": raw.get("reason", "").strip(),
                "time": gt_time
            })
        
        task_id_to_gt[str(idx_counter)] = gts

    # 2. Load Predictions
    results = []
    with results_csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        results = list(reader)

    # 3. Calculate Metrics
    metrics = {
        "all_correct": 0,
        "comp_reason_correct": 0,
        "comp_correct": 0,
        "reason_correct": 0,
        "time_correct": 0,
        "total_tasks": 0,
        "latency_sum": 0.0,
        "tokens_sum": 0.0,
        "input_tokens_sum": 0.0,
        "input_cache_tokens_sum": 0.0,
        "output_tokens_sum": 0.0,
        "count_valid_latency": 0
    }

    processed_tasks = set()

    for row in results:
        task_idx = row.get("task_index")
        if not task_idx or task_idx not in task_id_to_gt:
            continue
            
        # Avoid double counting if multiple runs exist? 
        # Usually we evaluate per run. If CSV has multiple runs for same task, we might need to handle.
        # But for now let's assume one run or evaluate all lines.
        # Ideally we should group by run_index too if multiple runs.
        # Let's count every line as a sample.
        
        gt_list = task_id_to_gt[task_idx]
        if not gt_list: continue

        metrics["total_tasks"] += 1
        
        # Parse Prediction
        # "comp1; comp2"
        p_comps = [c.strip() for c in row.get("component", "").split(";")]
        p_reasons = [r.strip() for r in row.get("reason", "").split(";")]
        p_times_str = [t.strip() for t in row.get("time", "").split(";")]
        
        # Zip them. If lengths mismatch, use safe zip or truncate
        # We need to form (comp, reason, time) tuples
        max_len = max(len(p_comps), len(p_reasons), len(p_times_str))
        preds = []
        for i in range(max_len):
            c = p_comps[i] if i < len(p_comps) else ""
            r = p_reasons[i] if i < len(p_reasons) else ""
            t_str = p_times_str[i] if i < len(p_times_str) else ""
            
            p_time = None
            if t_str:
                try:
                    # Try various formats
                    try:
                        p_time = datetime.fromisoformat(t_str)
                    except:
                        p_time = datetime.strptime(t_str, "%Y-%m-%d %H:%M:%S")
                except:
                    pass
            
            preds.append({
                "component": c,
                "reason": r,
                "time": p_time
            })

        # --- Comparisons ---
        # We need to check if PredSet == GTSet for different criteria
        
        # Helper to check match
        def check_match(pred, gt, check_comp=True, check_reason=True, check_time=True):
            if check_comp and pred["component"] != gt["component"]:
                return False
            if check_reason and pred["reason"] != gt["reason"]:
                return False
            if check_time:
                if not pred["time"] or not gt["time"]:
                    return False
                diff = abs((pred["time"] - gt["time"]).total_seconds())
                if diff > 300: # 5 minutes
                    return False
                # Also check if within window? GT time is already in window. 
                # 5 min error margin around GT usually implies it's close enough.
            return True

        def is_set_equal(check_comp=True, check_reason=True, check_time=True):
            # Check Recall: For every GT, exists a match in Preds
            # Check Precision: For every Pred, exists a match in GTs
            # (And mapped 1-to-1 ideally, but simple set check is usually enough for RCA)
            
            # Since we might have duplicates in component/reason, we need to handle list matching carefully.
            # Let's use a copy of lists to match one-by-one.
            
            p_remaining = list(preds)
            gt_remaining = list(gt_list)
            
            # Match GTs
            for gt in list(gt_remaining):
                matched_p = None
                for p in p_remaining:
                    if check_match(p, gt, check_comp, check_reason, check_time):
                        matched_p = p
                        break
                if matched_p:
                    p_remaining.remove(matched_p)
                    gt_remaining.remove(gt)
                else:
                    return False # GT not found
            
            # If any Pred remaining, it's a False Positive -> Not Exact Match
            if len(p_remaining) > 0:
                return False
                
            return True

        if is_set_equal(True, True, True): metrics["all_correct"] += 1
        if is_set_equal(True, True, True):
            print(f"Task {task_idx} is exact match")
        if is_set_equal(True, True, False): metrics["comp_reason_correct"] += 1
        if is_set_equal(True, False, False): metrics["comp_correct"] += 1
        if is_set_equal(False, True, False): metrics["reason_correct"] += 1
        if is_set_equal(False, False, True): metrics["time_correct"] += 1
        
        # Latency/Tokens
        try:
            lat = float(row.get("latency_ms", 0))
            if lat > 0:
                metrics["latency_sum"] += lat
                metrics["count_valid_latency"] += 1
            metrics["tokens_sum"] += float(row.get("total_tokens", 0))
            metrics["input_tokens_sum"] += float(row.get("input_tokens", 0))
            metrics["input_cache_tokens_sum"] += float(row.get("input_cache_tokens", 0))
            metrics["output_tokens_sum"] += float(row.get("output_tokens", 0))
        except: pass

    # Print Report
    total = metrics["total_tasks"]
    if total == 0:
        print("No results to evaluate.")
        return

    print("\n" + "="*50)
    print(f"EVALUATION REPORT ({total} tasks)")
    print("="*50)
    print(f"Avg Latency: {metrics['latency_sum']/metrics['count_valid_latency']:.2f} ms" if metrics['count_valid_latency'] else "Avg Latency: N/A")
    print(f"Avg Total Tokens:  {metrics['tokens_sum']/total:.2f}")
    print(f"Avg Input Tokens:  {metrics['input_tokens_sum']/total:.2f}")
    print(f"Avg Input Cache Tokens: {metrics['input_cache_tokens_sum']/total:.2f}")
    print(f"Avg Output Tokens: {metrics['output_tokens_sum']/total:.2f}")
    print("-" * 30)
    print(f"Component + Reason + Time Correct: {metrics['all_correct']} ({metrics['all_correct']/total:.2%})")
    print(f"Component + Reason Correct:      {metrics['comp_reason_correct']} ({metrics['comp_reason_correct']/total:.2%})")
    print(f"Component Correct:               {metrics['comp_correct']} ({metrics['comp_correct']/total:.2%})")
    print(f"Reason Correct:                  {metrics['reason_correct']} ({metrics['reason_correct']/total:.2%})")
    print(f"Time Correct:                    {metrics['time_correct']} ({metrics['time_correct']/total:.2%})")
    print("="*50 + "\n")

    # Update README
    try:
        _update_readme(metrics, "baseline" if "baseline" in str(results_csv_path) else "rca", changes)
    except Exception as e:
        print(f"Failed to update README: {e}")

def _update_readme(metrics: Dict[str, Any], mode: str, changes: str = "-"):
    readme_path = Path(__file__).parent / "README.md"
    if not readme_path.exists():
        return

    today = datetime.now().strftime("%Y-%m-%d")
    total = metrics["total_tasks"]
    if total == 0: return

    acc_full = metrics['all_correct']/total * 100
    acc_cr = metrics['comp_reason_correct']/total * 100
    acc_c = metrics['comp_correct']/total * 100
    acc_r = metrics['reason_correct']/total * 100
    acc_t = metrics['time_correct']/total * 100
    avg_lat = (metrics['latency_sum']/metrics['count_valid_latency'] / 1000) if metrics['count_valid_latency'] else 0
    avg_tok = metrics['tokens_sum']/total / 1000 # in k
    avg_input_k = metrics['input_tokens_sum']/total/1000
    avg_cache_k = metrics['input_cache_tokens_sum']/total/1000
    avg_output_k = metrics['output_tokens_sum']/total/1000

    # Read content
    content = readme_path.read_text(encoding="utf-8")
    
    # --- 1. Append New Row to Table ---
    # Header signature to locate the table
    table_header_sig = "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |"
    
    new_row = f"| **{today}** | {mode} | {changes} | {avg_lat:.0f}s | {avg_tok:.0f}k | {avg_input_k:.0f}k | {avg_cache_k:.0f}k | {avg_output_k:.0f}k | {acc_full:.2f}% | {acc_cr:.2f}% | {acc_c:.2f}% | {acc_r:.2f}% | {acc_t:.2f}% |"
    
    if table_header_sig in content:
        parts = content.split(table_header_sig)
        # Append new row immediately after header separator
        content = parts[0] + table_header_sig + "\n" + new_row + parts[1]
    
    # --- 2. Parse All Data for Charts ---
    # We need to re-read the table from 'content' (which now includes the new row)
    # to generate the full history for charts.
    
    import re
    
    # Regex to find the table rows
    # Look for lines starting with | and containing the date pattern or just non-header lines
    # Skip the header lines
    
    lines = content.splitlines()
    data_rows = []
    
    in_table = False
    header_found = False
    
    for line in lines:
        if "| 日期 | Agent模式 |" in line:
            in_table = True
            continue
        if ":---" in line and in_table:
            header_found = True
            continue
        
        if in_table and header_found:
            if not line.strip().startswith("|"):
                in_table = False # End of table
                continue
            
            # Parse row
            # | **2026-01-23** | baseline | ...
            cols = [c.strip() for c in line.strip().split('|') if c.strip()]
            if len(cols) >= 13: # Ensure we have enough columns
                # Columns:
                # 0: Date, 1: Mode, 2: Changes, 3: Latency, 4: TotalTok, 
                # 5: InTok, 6: CacheTok, 7: OutTok, 
                # 8: Full%, 9: CR%, 10: Comp%, 11: Reason%, 12: Time%
                
                try:
                    row_data = {
                        "date": cols[0].replace("**", ""),
                        "mode": cols[1],
                        "changes": cols[2],
                        "acc_full": float(cols[8].strip('%')),
                        "acc_cr": float(cols[9].strip('%')),
                        "acc_c": float(cols[10].strip('%')),
                        "acc_r": float(cols[11].strip('%')),
                        "acc_t": float(cols[12].strip('%')),
                        "tokens": float(cols[4].strip('k')),
                        "avg_lat": float(cols[3].strip('s'))
                    }
                    data_rows.append(row_data)
                except ValueError:
                    continue # Skip malformed rows
    
    # Sort data by date (Oldest to Newest)
    # The table usually has newest on top (if we prepend). 
    # Let's assume we want chronological order for charts.
    # Our 'new_row' logic prepends, so the list 'data_rows' (read from top to bottom) is Newest -> Oldest.
    # We need to reverse it.
    data_rows.reverse()
    
    # Filter Data
    baselines = [r for r in data_rows if r['mode'] == 'baseline']
    
    # Determine Baseline Constants
    if baselines:
        base_acc_full = baselines[-1]['acc_full']
        base_acc_cr = baselines[-1]['acc_cr']
        base_acc_time = baselines[-1]['acc_t']
        base_lat = baselines[-1]['avg_lat']
    else:
        base_acc_full = 0
        base_acc_cr = 0
        base_acc_time = 0
        base_lat = 0
    
    # Prepare Chart Data
    x_labels = []
    y_acc_full = []
    y_acc_cr = []
    y_acc_comp = []
    y_acc_reason = []
    y_acc_time = []
    y_lat = []
    y_tokens = []
    
    for r in data_rows:
        label = r['date'][5:] # Remove Year "2026-" -> "01-23"
        x_labels.append(f'"{label}"')
        y_acc_full.append(r['acc_full'])
        y_acc_cr.append(r['acc_cr'])
        y_acc_comp.append(r['acc_c'])
        y_acc_reason.append(r['acc_r'])
        y_acc_time.append(r['acc_t'])
        y_lat.append(r['avg_lat'])
        y_tokens.append(r['tokens'])

    if not x_labels:
        # If no data yet
        x_labels = ["Pending"]
        y_acc_full = [0]
        y_acc_cr = [0]
        y_acc_comp = [0]
        y_acc_reason = [0]
        y_acc_time = [0]
        y_lat = [0]
        y_tokens = [0]

    # Helper to generate Baseline Bar Data
    base_full_data = [base_acc_full] * len(x_labels)
    base_cr_data = [base_acc_cr] * len(x_labels)
    base_time_data = [base_acc_time] * len(x_labels)
    base_lat_data = [base_lat] * len(x_labels)

    # --- 3. Regenerate Chart Sections ---
    
    # Chart 1: Full Accuracy
    chart1_code = f"""```mermaid
xychart-beta
    title "准确率趋势 (Baseline: {base_acc_full:.2f}%)"
    x-axis [{", ".join(x_labels)}]
    y-axis "准确率 (%)" 0 --> 100
    bar [{", ".join(f"{v:.2f}" for v in base_full_data)}]
    line [{", ".join(f"{v:.2f}" for v in y_acc_full)}]
```"""

    # Chart 2: Component + Reason Accuracy
    chart2_code = f"""```mermaid
xychart-beta
    title "组件+原因 准确率 (Baseline: {base_acc_cr:.2f}%)"
    x-axis [{", ".join(x_labels)}]
    y-axis "准确率 (%)" 0 --> 100
    bar [{", ".join(f"{v:.2f}" for v in base_cr_data)}]
    line [{", ".join(f"{v:.2f}" for v in y_acc_cr)}]
```"""

    # Chart 3: Component vs Reason (Internal Comparison)
    chart3_code = f"""```mermaid
xychart-beta
    title "诊断细分: 仅组件 vs 仅原因"
    x-axis [{", ".join(x_labels)}]
    y-axis "准确率 (%)" 0 --> 100
    bar [{", ".join(f"{v:.2f}" for v in y_acc_comp)}]
    line [{", ".join(f"{v:.2f}" for v in y_acc_reason)}]
```"""

    # Chart 4: Time Correctness
    chart4_code = f"""```mermaid
xychart-beta
    title "时间定位准确率 (Baseline: {base_acc_time:.2f}%)"
    x-axis [{", ".join(x_labels)}]
    y-axis "准确率 (%)" 0 --> 100
    bar [{", ".join(f"{v:.2f}" for v in base_time_data)}]
    line [{", ".join(f"{v:.2f}" for v in y_acc_time)}]
```"""

    # Chart 5: Latency
    # Adjust Y-axis max dynamically? 
    max_lat = max(y_lat + [base_lat]) if y_lat else 100
    max_lat_axis = int(max_lat * 1.2)
    chart5_code = f"""```mermaid
xychart-beta
    title "平均耗时 (Baseline: {base_lat:.0f}s)"
    x-axis [{", ".join(x_labels)}]
    y-axis "耗时 (s)" 0 --> {max_lat_axis}
    bar [{", ".join(f"{v:.0f}" for v in base_lat_data)}]
    line [{", ".join(f"{v:.0f}" for v in y_lat)}]
```"""

    # Chart 6: Token Usage
    max_tok = max(y_tokens) if y_tokens else 100
    max_tok_axis = int(max_tok * 1.2)
    chart6_code = f"""```mermaid
xychart-beta
    title "平均 Token 消耗 (千)"
    x-axis [{", ".join(x_labels)}]
    y-axis "Tokens (k)" 0 --> {max_tok_axis}
    bar [{", ".join(f"{v:.0f}" for v in y_tokens)}]
```"""

    # Apply updates
    trend_header = "## 📈 趋势分析"
    next_header = "## 🤖 模型与配置"
    
    if trend_header in content and next_header in content:
        start_idx = content.find(trend_header)
        end_idx = content.find(next_header)
        
        new_section = f"""{trend_header}

> **说明**: 
> *   `baseline`: 性能通常保持稳定（图中柱状背景/水平线）。
> *   `rca`: 随着Agent的迭代优化，性能曲线应呈现波动上升趋势。

### 核心准确率趋势 (RCA vs Baseline)

{chart1_code}

### 组件+原因 准确率 (RCA vs Baseline)

{chart2_code}

### 诊断细分 (仅组件 vs 仅原因)

{chart3_code}

### 时间定位准确率 (RCA vs Baseline)

{chart4_code}

### 平均耗时 (RCA vs Baseline)

{chart5_code}

### 资源消耗 (Resource Usage)

{chart6_code}

---

"""
        content = content[:start_idx] + new_section + content[end_idx:]
    
    readme_path.write_text(content, encoding="utf-8")
    print(f"Updated README.md with new results and charts.")

if __name__ == "__main__":
    main()
