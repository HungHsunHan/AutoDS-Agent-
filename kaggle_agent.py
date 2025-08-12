import io
import json
import logging
import os
import re
import shutil
import sys
from contextlib import redirect_stderr, redirect_stdout
from typing import Annotated, Dict, List, TypedDict

import pandas as pd
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langsmith import traceable

# Import prompt definitions
from prompts import (
    DATA_ANALYST_PROMPT,
    FEATURE_ENGINEER_PROMPT,
    MODEL_ARCHITECT_PROMPT,
    PROJECT_MANAGER_PROMPT,
    REPORT_GENERATOR_PROMPT,
    STRATEGIST_PROMPT,
)


# Utility to safely escape curly braces in prompts when used with ChatPromptTemplate
# to prevent accidental variable detection (e.g., JSON examples). Keeps specified placeholders.
def _escape_braces_for_template(text: str, keep: List[str] | None = None) -> str:
    if not text:
        return text
    keep = keep or []
    placeholders = {k: f"__PLACEHOLDER_{i}__" for i, k in enumerate(keep)}
    # Temporarily replace kept placeholders
    for k, ph in placeholders.items():
        text = text.replace(f"{{{k}}}", ph)
    # Escape all remaining braces
    text = text.replace("{", "{{").replace("}", "}}")
    # Restore kept placeholders
    for k, ph in placeholders.items():
        text = text.replace(ph, f"{{{k}}}")
    return text


# Load environment variables from .env file
load_dotenv(override=True)

# Configure LangSmith
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY", "")
os.environ["LANGSMITH_PROJECT"] = os.getenv("LANGSMITH_PROJECT", "KaggleAgent")
os.environ["LANGSMITH_TRACING"] = os.getenv("LANGSMITH_TRACING", "true")

# Standard LLM instance
# z-ai/glm-4.5-air:free, moonshotai/kimi-k2:free, deepseek/deepseek-chat-v3-0324:free, qwen/qwen3-coder
# google/gemini-2.0-flash-001
model = "openai/gpt-4.1-mini"
llm = ChatOpenAI(
    model=model,
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    temperature=0.1,  # Lower temperature for more consistent output
    model_kwargs={
        "response_format": {"type": "json_object"}  # Enable JSON mode when supported
    },
)

# JSON-optimized LLM instance for structured outputs
json_llm_base = ChatOpenAI(
    model=model,
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    temperature=0,  # Very low temperature for structured output
    # max_tokens=40000,  # Reduced from 4000 to prevent truncation errors
    model_kwargs={"response_format": {"type": "json_object"}},
)


# Helper function to create structured LLM with schema
def create_structured_llm(schema: dict):
    """Create a structured output LLM with the given schema and JSON validation"""
    return json_llm_base.with_structured_output(schema)


# Validation helper function
def validate_json_response(response, schema_title: str) -> bool:
    """Validate if the response matches expected structure"""
    try:
        # Handle string responses that might be JSON
        if isinstance(response, str):
            logger.warning(
                f"⚠️  Received string response for {schema_title}: {response[:100]}..."
            )
            try:
                import json
                import re

                # Clean the response before parsing
                cleaned_response = response.strip()
                # Fix common JSON formatting issues with newlines in field names/values
                cleaned_response = re.sub(r'\n\s+(".*?":)', r" \1", cleaned_response)
                # Remove standalone newlines in JSON values
                cleaned_response = re.sub(
                    r'"\s*\n\s*([^"]*)\s*\n\s*"', r'"\1"', cleaned_response
                )

                parsed = json.loads(cleaned_response)
                if isinstance(parsed, dict):
                    logger.debug(
                        f"✅ Successfully parsed string JSON response for {schema_title}"
                    )
                    return True
            except json.JSONDecodeError as e:
                logger.warning(
                    f"⚠️  String response is not valid JSON for {schema_title}: {e}"
                )
                logger.debug(f"Original response: {response[:200]}...")
                return False

        if hasattr(response, "__dict__") or isinstance(response, dict):
            logger.debug(f"✅ Valid JSON response for {schema_title}")
            return True
        else:
            logger.warning(
                f"⚠️  Invalid JSON response structure for {schema_title}, type: {type(response)}"
            )
            return False
    except Exception as e:
        logger.error(f"❌ JSON validation error for {schema_title}: {e}")
        return False


# JSON mode effectiveness monitoring
def log_json_mode_stats(stats: dict = None):
    """Log statistics about JSON mode effectiveness"""
    if stats is None:
        stats = {
            "total_requests": 0,
            "json_success": 0,
            "fallback_used": 0,
            "validation_failures": 0,
        }

    success_rate = (
        (stats["json_success"] / stats["total_requests"] * 100)
        if stats["total_requests"] > 0
        else 0
    )

    logger.debug("=" * 50)
    logger.debug("📊 JSON模式效能統計")
    logger.debug(f"總請求數: {stats['total_requests']}")
    logger.debug(f"JSON成功: {stats['json_success']} ({success_rate:.1f}%)")
    logger.debug(f"後備模式: {stats['fallback_used']}")
    logger.debug(f"驗證失敗: {stats['validation_failures']}")
    logger.debug("=" * 50)

    if success_rate < 80:
        logger.warning("⚠️  JSON模式成功率低於80%，建議檢查LLM配置")
    elif success_rate > 95:
        logger.debug("🎉 JSON模式運作優異！")

    return stats


# Response sanitization and validation for routing
def validate_and_sanitize_strategist_response(
    response, fallback_agent="Feature_Engineer_Agent"
):
    """
    Validate and sanitize strategist response to ensure valid routing.

    Args:
        response: The strategist response object
        fallback_agent: Default agent to route to if validation fails

    Returns:
        dict: Sanitized response with guaranteed valid next_step
    """
    allowed_next_steps = {
        "Data_Analysis_Agent",
        "Feature_Engineer_Agent",
        "Model_Architect_Agent",
        "Report_Generator_Agent",
        "END",
    }

    # Handle different response types
    sanitized_response = {}

    try:
        # Extract response data
        if hasattr(response, "model_dump") and callable(
            getattr(response, "model_dump")
        ):
            sanitized_response = response.model_dump()
        elif hasattr(response, "dict") and callable(getattr(response, "dict")):
            sanitized_response = response.dict()
        elif isinstance(response, dict):
            sanitized_response = response.copy()
        else:
            logger.warning(f"⚠️  Unexpected response type: {type(response)}")
            sanitized_response = {
                "next_step": fallback_agent,
                "feedback": str(response),
            }

        # Validate and sanitize next_step
        next_step = sanitized_response.get("next_step", "").strip()

        # Handle common malformed values
        if not next_step or next_step in ["", " ", ":", ": ", "None", "null"]:
            logger.warning(
                f"⚠️  Empty or malformed next_step '{next_step}', using fallback: {fallback_agent}"
            )
            next_step = fallback_agent
        elif next_step not in allowed_next_steps:
            logger.warning(
                f"⚠️  Invalid next_step '{next_step}', using fallback: {fallback_agent}"
            )
            next_step = fallback_agent

        # Ensure required fields exist
        sanitized_response["next_step"] = next_step
        if "feedback" not in sanitized_response or not sanitized_response["feedback"]:
            sanitized_response["feedback"] = (
                f"Proceeding to {next_step} based on current workflow state."
            )

        # Sanitize other fields
        for field in ["validation_score", "test_score"]:
            if field in sanitized_response and sanitized_response[field] is not None:
                try:
                    sanitized_response[field] = float(sanitized_response[field])
                except (ValueError, TypeError):
                    sanitized_response[field] = None

        logger.debug(f"✅ Strategist response validated: next_step='{next_step}'")
        return sanitized_response

    except Exception as e:
        logger.error(f"❌ Response validation failed: {e}")
        return {
            "next_step": fallback_agent,
            "feedback": f"Validation error occurred, proceeding to {fallback_agent}. Error: {str(e)}",
            "validation_score": None,
            "test_score": None,
            "submission_file_path": "",
            "performance_analysis": "",
            "should_continue": True,
            "error_analysis": f"Response validation error: {str(e)}",
            "confidence_level": 0.5,
        }


# Debug and monitoring utilities
def debug_workflow_state(state: dict, checkpoint_name: str = "unknown"):
    """
    Debug utility to log workflow state at key checkpoints.
    """
    logger.debug(f"=== 工作流程狀態檢查點: {checkpoint_name} ===")

    # Log key state information
    key_fields = [
        "target_column",
        "validation_score",
        "test_score",
        "submission_file_path",
        "last_code_generating_agent",
        "error_count",
        "next_node_after_triage",
    ]

    for field in key_fields:
        value = state.get(field)
        if value is not None:
            logger.debug(f"  {field}: {value}")

    # Check strategist decision
    strategist_decision = state.get("strategist_decision")
    if strategist_decision:
        next_step = strategist_decision.get("next_step")
        logger.debug(f"  strategist_next_step: {next_step}")

    # Check available files
    available_files = state.get("available_files", {})
    if available_files:
        total_files = sum(len(files) for files in available_files.values())
        logger.debug(f"  total_available_files: {total_files}")
        for folder, files in available_files.items():
            if files:
                logger.debug(f"    {folder}: {len(files)} files")

    logger.debug("=" * 50)


def validate_llm_response_structure(response, expected_schema: str, agent_name: str):
    """
    Enhanced validation for LLM responses with detailed debugging.
    """
    logger.debug(f"🔍 Validating {agent_name} response structure for {expected_schema}")

    # Check response type
    logger.debug(f"  Response type: {type(response)}")

    # Try to extract content
    if hasattr(response, "content"):
        logger.debug(f"  Response has content: {len(str(response.content))} chars")
        if isinstance(response.content, str):
            # Check if it looks like JSON
            content = response.content.strip()
            if content.startswith("{") and content.endswith("}"):
                logger.debug("  Content appears to be JSON format")
            else:
                logger.warning("  Content does not appear to be JSON format")

    # Check for structured output attributes
    if hasattr(response, "model_dump"):
        logger.debug("  Response has model_dump method (Pydantic v2)")
    elif hasattr(response, "dict"):
        logger.debug("  Response has dict method (Pydantic v1)")
    elif isinstance(response, dict):
        logger.debug("  Response is already a dict")
    else:
        logger.warning(f"  Unknown response structure: {dir(response)}")

    return True


# Enhanced error tracking
def track_agent_error(agent_name: str, error_type: str, error_details: str):
    """
    Enhanced error tracking with better categorization.
    """
    increment_error_stats(error_type, agent_name)

    # Log structured error information
    logger.error(f"🚨 {agent_name} Error Detected:")
    logger.error(f"   Type: {error_type}")
    logger.error(f"   Details: {error_details[:200]}...")

    # Add to global error tracking
    current_time = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    error_entry = {
        "timestamp": current_time,
        "agent": agent_name,
        "type": error_type,
        "details": error_details[:500],  # Truncate long errors
    }

    # Store in global error log (if needed for analysis)
    if not hasattr(track_agent_error, "error_log"):
        track_agent_error.error_log = []
    track_agent_error.error_log.append(error_entry)

    # Keep only last 50 errors to prevent memory bloat
    if len(track_agent_error.error_log) > 50:
        track_agent_error.error_log = track_agent_error.error_log[-50:]


# 配置日志系统
def setup_logging():
    """設置增強的日志記錄系統 - 專注於錯誤檢測"""
    # 创建日志记录器
    logger = logging.getLogger("kaggle_agent")
    logger.setLevel(logging.DEBUG)  # 設置為DEBUG以捕獲所有級別

    # 如果logger已经有handlers，先清除
    if logger.handlers:
        logger.handlers.clear()

    # 創建文件處理器 - 保存詳細日志用於調試
    file_handler = logging.FileHandler(
        "kaggle_agent.log", mode="w", encoding="utf-8"
    )  # 使用'w'模式重新開始
    file_handler.setLevel(logging.DEBUG)

    # 創建控制台處理器 - 只顯示警告和錯誤
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)  # 只顯示WARNING及以上級別

    # 創建不同的格式化器
    # 文件使用詳細格式
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 控制台使用簡潔格式，重點突出錯誤
    console_formatter = logging.Formatter("%(levelname)s: %(message)s")

    file_handler.setFormatter(file_formatter)
    console_handler.setFormatter(console_formatter)

    # 添加处理器到logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


# 初始化日志系统
logger = setup_logging()

# 錯誤統計追蹤
error_stats = {
    "total_errors": 0,
    "syntax_errors": 0,
    "import_errors": 0,
    "runtime_errors": 0,
    "file_errors": 0,
    "last_agent_errors": {},
}


def log_error_stats():
    """記錄錯誤統計摘要"""
    if error_stats["total_errors"] > 0:
        logger.warning(f"🚨 錯誤統計總結:")
        logger.warning(f"   總錯誤數: {error_stats['total_errors']}")
        logger.warning(f"   語法錯誤: {error_stats['syntax_errors']}")
        logger.warning(f"   導入錯誤: {error_stats['import_errors']}")
        logger.warning(f"   運行時錯誤: {error_stats['runtime_errors']}")
        logger.warning(f"   文件錯誤: {error_stats['file_errors']}")
        if error_stats["last_agent_errors"]:
            logger.warning(f"   各代理錯誤次數: {error_stats['last_agent_errors']}")


def increment_error_stats(error_type: str, agent_name: str):
    """增加錯誤統計計數"""
    error_stats["total_errors"] += 1

    # 按錯誤類型分類
    if "syntax" in error_type:
        error_stats["syntax_errors"] += 1
    elif "import" in error_type:
        error_stats["import_errors"] += 1
    elif "file" in error_type:
        error_stats["file_errors"] += 1
    else:
        error_stats["runtime_errors"] += 1

    # 按代理分類
    if agent_name not in error_stats["last_agent_errors"]:
        error_stats["last_agent_errors"][agent_name] = 0
    error_stats["last_agent_errors"][agent_name] += 1


# --- JSON Schema Definitions for Structured Output ---

# Project Manager Schema
PROJECT_MANAGER_SCHEMA = {
    "title": "ProjectManagerResponse",
    "type": "object",
    "properties": {
        "target_column": {
            "title": "Target Column",
            "type": "string",
            "description": "推斷的目標欄位名稱",
        },
        "plan": {
            "title": "Project Plan",
            "type": "string",
            "description": "詳細的專案計畫",
        },
        "problem_type": {
            "title": "Problem Type",
            "type": "string",
            "description": "問題類型（分類、回歸等）",
        },
        "evaluation_metric": {
            "title": "Evaluation Metric",
            "type": "string",
            "description": "評估指標（AUC、Accuracy等）",
        },
        "next_task_description": {
            "title": "Next Task Description",
            "type": "string",
            "description": "給下一個代理的任務描述",
        },
    },
    "required": ["target_column", "plan", "next_task_description"],
}

# Code-Generating Agent Schema
CODE_AGENT_SCHEMA = {
    "title": "CodeAgentResponse",
    "type": "object",
    "properties": {
        "code_to_execute": {
            "title": "Code to Execute",
            "type": "string",
            "description": "要執行的Python程式碼",
        },
        "description": {
            "title": "Code Description",
            "type": "string",
            "description": "程式碼功能描述",
        },
        "expected_outputs": {
            "title": "Expected Outputs",
            "type": "array",
            "items": {"type": "string"},
            "description": "預期的輸出檔案或結果",
        },
        "dependencies": {
            "title": "Dependencies",
            "type": "array",
            "items": {"type": "string"},
            "description": "所需的Python套件",
        },
        "data_sources": {
            "title": "Data Sources",
            "type": "array",
            "items": {"type": "string"},
            "description": "使用的資料來源檔案",
        },
        "output_files": {
            "title": "Output Files",
            "type": "array",
            "items": {"type": "string"},
            "description": "將要生成的檔案路徑",
        },
    },
    "required": ["code_to_execute", "description"],
}

# Enhanced Strategist Schema (extending the existing one)
ENHANCED_STRATEGIST_SCHEMA = {
    "title": "StrategistDecision",
    "type": "object",
    "properties": {
        "next_step": {
            "title": "Next Step",
            "type": "string",
            "enum": [
                "Data_Analysis_Agent",
                "Feature_Engineer_Agent",
                "Model_Architect_Agent",
                "Report_Generator_Agent",
                "END",
            ],
            "description": "下一個要呼叫的代理或END結束流程",
        },
        "feedback": {"title": "Feedback", "type": "string"},
        "validation_score": {
            "title": "Validation Score",
            "type": ["number", "null"],
            "description": "驗證分數，從執行輸出中解析出來，如果沒有找到則為null",
        },
        "test_score": {
            "title": "Test Score",
            "type": ["number", "null"],
            "description": "測試分數，從執行輸出中解析出來，如果沒有找到則為null",
        },
        "submission_file_path": {
            "title": "Submission File Path",
            "type": "string",
            "description": "提交檔案路徑，從執行輸出中解析出來，如果沒有找到則為空字串",
        },
        "performance_analysis": {
            "title": "Performance Analysis",
            "type": "string",
            "description": "對模型性能的分析，包括驗證分數和測試分數的評估、是否過擬合、是否需要優化等",
        },
        "should_continue": {
            "title": "Should Continue",
            "type": "boolean",
            "description": "基於性能分析，判斷是否應該繼續優化模型還是結束流程",
        },
        "error_analysis": {
            "title": "Error Analysis",
            "type": "string",
            "description": "對錯誤的分析和建議的解決方案",
        },
        "confidence_level": {
            "title": "Confidence Level",
            "type": "number",
            "minimum": 0,
            "maximum": 1,
            "description": "對決策的信心程度（0-1）",
        },
    },
    "required": ["next_step", "feedback"],
}

# Report Generator Schema
REPORT_GENERATOR_SCHEMA = {
    "title": "ReportGeneratorResponse",
    "type": "object",
    "properties": {
        "code_to_execute": {
            "title": "Code to Execute",
            "type": "string",
            "description": "要執行的Python程式碼來生成報告",
        },
        "report_title": {
            "title": "Report Title",
            "type": "string",
            "description": "報告標題",
        },
        "report_summary": {
            "title": "Report Summary",
            "type": "string",
            "description": "報告摘要",
        },
        "key_findings": {
            "title": "Key Findings",
            "type": "array",
            "items": {"type": "string"},
            "description": "主要發現列表",
        },
        "recommendations": {
            "title": "Recommendations",
            "type": "array",
            "items": {"type": "string"},
            "description": "建議列表",
        },
        "charts_analyzed": {
            "title": "Charts Analyzed",
            "type": "array",
            "items": {"type": "string"},
            "description": "分析的圖表列表",
        },
    },
    "required": ["code_to_execute", "report_title", "report_summary"],
}


class KaggleWorkflowState(TypedDict):
    """
    描述工作流程中每一步的狀態。
    """

    problem_statement: str
    data_path: str
    target_column: str  # 目標欄位名稱

    plan: str
    eda_report: str
    feature_plan: str
    model_plan: str

    code_to_execute: str
    execution_stdout: str
    execution_stderr: str

    validation_score: float
    test_score: float
    submission_file_path: str

    iteration_history: List[str]
    last_code_generating_agent: str
    current_task_description: str
    strategist_decision: Dict

    # 新增的錯誤處理與流程控制欄位
    error_count: int  # 計數連續錯誤的次數
    next_node_after_triage: str  # 分流節點決定的下一步

    # 新增的路徑追蹤欄位
    workspace_paths: Dict  # 包含所有工作目錄的路徑資訊
    available_files: Dict  # 當前可用的檔案清單

    # 額外的模型相關參數
    feature_columns: List[str]  # 處理後的特徵欄位列表
    model_performance: Dict  # 存儲模型性能指標
    best_hyperparameters: Dict  # 最佳超參數
    preprocessing_steps: List[str]  # 記錄前處理步驟


# --- 2. 建立核心工具：程式碼執行沙箱 (Create the Core Tool: Code Executor) ---
# **V2 更新**: 新增了 _clean_code 輔助函式來移除 LLM 可能產生的 Markdown 標籤。


def _clean_code(code: str) -> str:
    """
    輔助函式，用於清理程式碼字串。
    在JSON模式下主要用於後備處理，移除 Markdown 標籤。
    """
    # 移除 ```python, ```, etc. (後備清理，JSON模式下應該不需要)
    if "```python" in code:
        try:
            code = re.search(r"```python\n(.*)```", code, re.DOTALL).group(1)
        except AttributeError:
            # 如果正則表達式失敗，嘗試簡單替換
            code = code.replace("```python", "").replace("```", "")
    elif "```" in code:
        code = code.replace("```", "")
    return code.strip()


def _categorize_execution_errors(stderr: str, agent_name: str = None) -> Dict:
    """
    分類和分析代碼執行錯誤
    """
    if not stderr:
        return {
            "has_error": False,
            "error_type": "none",
            "error_details": "",
            "formatted_error": "",
        }

    stderr_lower = stderr.lower()

    # 錯誤類型檢測
    error_patterns = {
        "syntax_error": ["syntaxerror", "invalid syntax", "unexpected token"],
        "import_error": ["importerror", "modulenotfounderror", "no module named"],
        "file_error": ["filenotfounderror", "no such file", "permission denied"],
        "memory_error": ["memoryerror", "out of memory"],
        "key_error": ["keyerror", "key not found"],
        "attribute_error": ["attributeerror", "has no attribute"],
        "type_error": ["typeerror", "unsupported operand", "argument"],
        "value_error": ["valueerror", "invalid literal", "cannot convert"],
        "index_error": ["indexerror", "list index out of range"],
        "zero_division": ["zerodivisionerror", "division by zero"],
        "runtime_error": ["runtimeerror", "runtime error"],
        "assertion_error": ["assertionerror"],
        "pandas_error": ["keyerror", "dataframeerror", "series"],
        "sklearn_error": ["sklearn", "fit", "transform", "predict"],
    }

    # 警告模式檢測
    warning_patterns = [
        "warning",
        "deprecat",
        "future",
        "userwarning",
        "deprecationwarning",
        "futurewarning",
        "pendingdeprecationwarning",
    ]

    # 首先檢查是否只是警告
    is_only_warning = True
    error_keywords = ["error", "exception", "traceback", "failed"]

    for keyword in error_keywords:
        if keyword in stderr_lower:
            # 檢查是否在警告上下文中
            if not any(warn in stderr_lower for warn in warning_patterns):
                is_only_warning = False
                break

    if is_only_warning:
        return {
            "has_error": False,
            "error_type": "warning",
            "error_details": stderr.strip(),
            "formatted_error": "",
        }

    # 確定錯誤類型
    detected_error_type = "unknown_error"
    for error_type, patterns in error_patterns.items():
        if any(pattern in stderr_lower for pattern in patterns):
            detected_error_type = error_type
            break

    # 提取關鍵錯誤信息
    error_lines = stderr.split("\n")
    key_error_lines = []
    for line in error_lines:
        line_lower = line.lower().strip()
        if any(
            keyword in line_lower for keyword in ["error:", "exception:", "traceback"]
        ):
            key_error_lines.append(line.strip())
        elif line.strip() and not any(warn in line_lower for warn in warning_patterns):
            # 包含非空的非警告行
            if len(key_error_lines) < 3:  # 限制關鍵行數
                key_error_lines.append(line.strip())

    error_summary = "\n".join(key_error_lines) if key_error_lines else stderr.strip()

    return {
        "has_error": True,
        "error_type": detected_error_type,
        "error_details": error_summary,
        "formatted_error": stderr.strip(),
    }


def setup_workspace_structure(base_path: str = "./kaggle_workspace") -> Dict:
    """
    設置工作區的資料夾結構並返回路徑資訊。
    """
    logger.debug("正在設置工作區結構")

    # 創建基礎工作目錄
    os.makedirs(base_path, exist_ok=True)

    # 定義所有需要的子資料夾，使用 os.path.join 確保路徑正確
    folders = {
        "workspace": os.path.abspath(base_path),
        "data": os.path.abspath(os.path.join(base_path, "data")),
        "image": os.path.abspath(os.path.join(base_path, "image")),
        "model": os.path.abspath(os.path.join(base_path, "model")),
        "after_preprocessing": os.path.abspath(
            os.path.join(base_path, "after_preprocessing")
        ),
    }

    # 創建所有資料夾
    for folder_name, folder_path in folders.items():
        os.makedirs(folder_path, exist_ok=True)
        logger.debug(f"已創建/確認資料夾: {folder_name} -> {folder_path}")

    return folders


def scan_available_files(workspace_paths: Dict) -> Dict:
    """
    Enhanced file scanning with intelligent file detection and categorization.
    """
    available_files = {
        "data": [],
        "image": [],
        "model": [],
        "after_preprocessing": [],
        "workspace": [],
    }

    file_categories = {
        "csv_files": [],
        "image_files": [],
        "model_files": [],
        "processed_files": [],
        "analysis_files": [],
    }

    for folder_name, folder_path in workspace_paths.items():
        if os.path.exists(folder_path):
            try:
                files = [
                    f
                    for f in os.listdir(folder_path)
                    if os.path.isfile(os.path.join(folder_path, f))
                ]
                available_files[folder_name] = files

                # Categorize files by type and purpose
                for file in files:
                    file_lower = file.lower()
                    file_path = os.path.join(folder_path, file)

                    if file_lower.endswith((".csv")):
                        file_categories["csv_files"].append(
                            {
                                "name": file,
                                "path": file_path,
                                "folder": folder_name,
                                "size": (
                                    os.path.getsize(file_path)
                                    if os.path.exists(file_path)
                                    else 0
                                ),
                            }
                        )

                        if "processed" in file_lower:
                            file_categories["processed_files"].append(
                                {"name": file, "path": file_path, "folder": folder_name}
                            )

                    elif file_lower.endswith((".png", ".jpg", ".jpeg", ".gif", ".svg")):
                        file_categories["image_files"].append(
                            {"name": file, "path": file_path, "folder": folder_name}
                        )

                    elif file_lower.endswith(
                        (".pkl", ".joblib", ".model", ".h5", ".pt")
                    ):
                        file_categories["model_files"].append(
                            {"name": file, "path": file_path, "folder": folder_name}
                        )

                    elif file_lower.endswith((".json", ".txt", ".md")):
                        file_categories["analysis_files"].append(
                            {"name": file, "path": file_path, "folder": folder_name}
                        )

            except PermissionError:
                available_files[folder_name] = []

    # Add file categories to the return dict
    available_files["file_categories"] = file_categories
    return available_files


def generate_file_context_string(workspace_paths: Dict, available_files: Dict) -> str:
    """
    Generate a comprehensive file context string for agents.
    """
    context_parts = []

    # Add workspace structure
    context_parts.append("=== CURRENT WORKSPACE STRUCTURE ===")
    for folder_name, folder_path in workspace_paths.items():
        files = available_files.get(folder_name, [])
        context_parts.append(f"{folder_name.upper()} FOLDER: {folder_path}")
        if files:
            for file in files[:10]:  # Limit to first 10 files per folder
                context_parts.append(f"  - {file}")
            if len(files) > 10:
                context_parts.append(f"  ... and {len(files) - 10} more files")
        else:
            context_parts.append("  - (empty)")

    # Add categorized file information
    file_cats = available_files.get("file_categories", {})
    context_parts.append("\n=== AVAILABLE DATA FILES ===")

    csv_files = file_cats.get("csv_files", [])
    if csv_files:
        context_parts.append("CSV Files:")
        for file_info in csv_files:
            size_mb = file_info["size"] / (1024 * 1024) if file_info["size"] > 0 else 0
            context_parts.append(
                f"  - {file_info['name']} ({file_info['folder']} folder, {size_mb:.1f}MB)"
            )

    processed_files = file_cats.get("processed_files", [])
    if processed_files:
        context_parts.append("Processed Data Files:")
        for file_info in processed_files:
            context_parts.append(
                f"  - {file_info['name']} ({file_info['folder']} folder)"
            )

    context_parts.append("\n=== AVAILABLE VISUALIZATION FILES ===")
    image_files = file_cats.get("image_files", [])
    if image_files:
        for file_info in image_files[:15]:  # Limit to first 15 images
            context_parts.append(
                f"  - {file_info['name']} ({file_info['folder']} folder)"
            )
        if len(image_files) > 15:
            context_parts.append(f"  ... and {len(image_files) - 15} more images")

    context_parts.append("\n=== PATH USAGE GUIDELINES ===")
    context_parts.append("- Always use relative paths from the workspace root")
    context_parts.append("- Data files: Use 'data/filename.csv'")
    context_parts.append(
        "- Processed data: Use 'after_preprocessing/filename.csv' or 'processed/filename.csv'"
    )
    context_parts.append("- Save images to: 'image/filename.png'")
    context_parts.append("- Save models to: 'model/filename.pkl'")
    context_parts.append("- For analysis outputs: Check existing files first")

    return "\n".join(context_parts)


def validate_and_suggest_file_paths(
    code: str, workspace_paths: Dict, available_files: Dict
) -> Dict:
    """
    Validate file paths in code and suggest corrections.
    """
    suggestions = []
    corrections = []

    import re

    # Common file path patterns in Python code
    path_patterns = [
        r'pd\.read_csv\([\'"]([^\'"]+)[\'"]',  # pandas read_csv
        r'\.to_csv\([\'"]([^\'"]+)[\'"]',  # pandas to_csv
        r'plt\.savefig\([\'"]([^\'"]+)[\'"]',  # matplotlib savefig
        r'open\([\'"]([^\'"]+)[\'"]',  # file open
        r'[\'"]([^\'"]*.csv)[\'"]',  # any .csv reference
        r'[\'"]([^\'"]*.png)[\'"]',  # any .png reference
        r'[\'"]([^\'"]*.pkl)[\'"]',  # any .pkl reference
    ]

    found_paths = set()
    for pattern in path_patterns:
        matches = re.findall(pattern, code)
        found_paths.update(matches)

    # Check each found path
    file_cats = available_files.get("file_categories", {})
    all_available_files = {}

    # Build a lookup of available files
    for cat_name, files in file_cats.items():
        for file_info in files:
            filename = file_info["name"]
            folder = file_info["folder"]
            all_available_files[filename] = {
                "folder": folder,
                "correct_path": f"{folder}/{filename}",
                "full_path": file_info["path"],
            }

    for path in found_paths:
        path_clean = path.strip()
        if not path_clean:
            continue

        # Check if path exists as specified
        full_path = None
        if os.path.isabs(path_clean):
            full_path = path_clean
        else:
            # Try relative to workspace
            workspace_root = workspace_paths.get("workspace", "")
            full_path = os.path.join(workspace_root, path_clean)

        if full_path and os.path.exists(full_path):
            # Path is valid
            continue

        # Path doesn't exist, try to suggest correction
        filename = os.path.basename(path_clean)

        if filename in all_available_files:
            file_info = all_available_files[filename]
            suggested_path = file_info["correct_path"]

            suggestions.append(
                {
                    "original_path": path_clean,
                    "suggested_path": suggested_path,
                    "reason": f"File exists in {file_info['folder']} folder",
                    "correction": path_clean != suggested_path,
                }
            )

            if path_clean != suggested_path:
                corrections.append({"from": path_clean, "to": suggested_path})
        else:
            # Try fuzzy matching
            similar_files = []
            for available_file in all_available_files.keys():
                if (
                    filename.lower() in available_file.lower()
                    or available_file.lower() in filename.lower()
                ):
                    similar_files.append(available_file)

            if similar_files:
                best_match = similar_files[0]
                file_info = all_available_files[best_match]
                suggestions.append(
                    {
                        "original_path": path_clean,
                        "suggested_path": file_info["correct_path"],
                        "reason": f"Similar file '{best_match}' found in {file_info['folder']} folder",
                        "correction": True,
                    }
                )

    return {
        "suggestions": suggestions,
        "corrections": corrections,
        "validation_passed": len(corrections) == 0,
    }


def apply_path_corrections(code: str, corrections: List[Dict]) -> str:
    """
    Apply path corrections to code.
    """
    corrected_code = code
    for correction in corrections:
        # Use word boundaries to avoid partial matches
        pattern = re.escape(correction["from"])
        replacement = correction["to"]
        corrected_code = re.sub(
            rf"['\"]\\s*{pattern}\\s*['\"]", f'"{replacement}"', corrected_code
        )

    return corrected_code


def detect_file_changes(old_files: Dict, new_files: Dict) -> Dict:
    """
    Detect changes in file system between two file states.
    """
    changes = {
        "new_files": [],
        "deleted_files": [],
        "modified_files": [],
        "summary": "",
    }

    # Check each folder for changes
    all_folders = set(old_files.keys()).union(set(new_files.keys()))

    for folder in all_folders:
        if folder == "file_categories":  # Skip the categories metadata
            continue

        old_folder_files = set(old_files.get(folder, []))
        new_folder_files = set(new_files.get(folder, []))

        # New files
        new_in_folder = new_folder_files - old_folder_files
        for file in new_in_folder:
            changes["new_files"].append(f"{folder}/{file}")

        # Deleted files
        deleted_in_folder = old_folder_files - new_folder_files
        for file in deleted_in_folder:
            changes["deleted_files"].append(f"{folder}/{file}")

    # Generate summary
    total_changes = (
        len(changes["new_files"])
        + len(changes["deleted_files"])
        + len(changes["modified_files"])
    )
    if total_changes > 0:
        changes["summary"] = (
            f"{len(changes['new_files'])} new, {len(changes['deleted_files'])} deleted, {len(changes['modified_files'])} modified"
        )
    else:
        changes["summary"] = "no changes"

    return changes


def create_file_state_backup(state: Dict) -> Dict:
    """
    Create a backup of current file state for rollback purposes.
    """
    backup = {
        "available_files": state.get("available_files", {}),
        "workspace_paths": state.get("workspace_paths", {}),
        "timestamp": pd.Timestamp.now().isoformat(),
    }
    return backup


def restore_file_state_from_backup(backup: Dict) -> Dict:
    """
    Restore file state from backup (for rollback scenarios).
    """
    return {
        "available_files": backup.get("available_files", {}),
        "workspace_paths": backup.get("workspace_paths", {}),
    }


@traceable(name="execute_code_node")
def execute_code(state: KaggleWorkflowState) -> Dict:
    """
    在一個受控環境中執行程式碼的工具節點。
    支援傳統執行和 Docker 安全執行兩種模式。
    """
    logger.debug("正在執行程式碼")

    code = state.get("code_to_execute", "")
    if not code:
        return {"execution_stdout": "", "execution_stderr": "沒有提供程式碼。"}

    # 智能程式碼清理：JSON模式下通常不需要，但保留作為後備
    cleaned_code = _clean_code(code)

    # 記錄是否需要清理（監控JSON模式效果）
    if cleaned_code != code:
        logger.warning(f"⚠️  程式碼需要清理，可能JSON模式未正常工作")
    else:
        logger.debug("✅ 程式碼已為純淨格式，JSON模式運作正常")

    # Enhanced path validation and correction
    workspace_paths = state.get("workspace_paths", {})
    available_files = state.get("available_files", {})

    if workspace_paths and available_files:
        validation_result = validate_and_suggest_file_paths(
            cleaned_code, workspace_paths, available_files
        )

        if not validation_result["validation_passed"]:
            logger.warning(
                f"⚠️  發現 {len(validation_result['corrections'])} 個路徑問題，正在自動修正"
            )
            for suggestion in validation_result["suggestions"]:
                if suggestion["correction"]:
                    logger.debug(
                        f"路徑修正: {suggestion['original_path']} -> {suggestion['suggested_path']}"
                    )
                    logger.debug(f"原因: {suggestion['reason']}")

            # Apply corrections
            cleaned_code = apply_path_corrections(
                cleaned_code, validation_result["corrections"]
            )
            logger.debug("✅ 路徑修正完成")
        else:
            logger.debug("✅ 所有檔案路徑均有效")

    return _execute_code_traditional(state, cleaned_code)


def _execute_code_traditional(state: KaggleWorkflowState, cleaned_code: str) -> Dict:
    """使用傳統方式執行程式碼，增強錯誤檢測和分類"""
    import time

    start_time = time.time()

    # 使用狀態中的工作目錄路徑
    workspace_paths = state.get("workspace_paths", {})
    work_dir_abs = workspace_paths.get(
        "workspace", os.path.abspath("./kaggle_workspace")
    )

    # 獲取當前執行的代理信息用於錯誤上下文
    current_agent = state.get("last_code_generating_agent", "Unknown_Agent")

    # 更新可用檔案清單
    available_files = scan_available_files(workspace_paths)
    code_with_context = cleaned_code

    original_cwd = os.getcwd()
    os.chdir(work_dir_abs)

    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()

    try:
        # 創建執行環境，包含save_report_file函數
        exec_globals = globals().copy()
        exec_globals["save_report_file"] = (
            lambda filename, content, ws_paths=workspace_paths: save_report_file(
                filename, content, ws_paths
            )
        )
        exec_globals["workspace_paths"] = workspace_paths

        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            exec(code_with_context, exec_globals)

        stdout = stdout_capture.getvalue()
        stderr = stderr_capture.getvalue()
        execution_time = time.time() - start_time

        # 增強的錯誤檢測和分類
        error_info = _categorize_execution_errors(stderr, current_agent)
        has_real_error = error_info["has_error"]

        # 只在有錯誤時記錄詳細信息
        if has_real_error:
            increment_error_stats(error_info["error_type"], current_agent)
            logger.error(f"🚨 代碼執行錯誤 - {current_agent}")
            logger.error(f"錯誤類型: {error_info['error_type']}")
            logger.error(f"錯誤詳情: {error_info['error_details']}")
            logger.error(f"執行時間: {execution_time:.2f}秒")
        elif stderr:
            # 只有警告時使用warning級別
            logger.warning(f"⚠️  代碼執行警告 - {current_agent}: {stderr.strip()}")

        # 更新可用檔案清單並記錄變化
        updated_files = scan_available_files(workspace_paths)

        # Log file changes for monitoring
        old_files = state.get("available_files", {})
        file_changes = detect_file_changes(old_files, updated_files)

        if (
            file_changes["new_files"]
            or file_changes["deleted_files"]
            or file_changes["modified_files"]
        ):
            logger.debug("📁 檔案系統變化檢測:")
            if file_changes["new_files"]:
                logger.debug(f"  新增檔案: {file_changes['new_files']}")
            if file_changes["deleted_files"]:
                logger.debug(f"  刪除檔案: {file_changes['deleted_files']}")
            if file_changes["modified_files"]:
                logger.debug(f"  修改檔案: {file_changes['modified_files']}")

        return {
            "execution_stdout": stdout,
            "execution_stderr": error_info["formatted_error"] if has_real_error else "",
            "available_files": updated_files,
            "file_changes": file_changes,
        }

    except SyntaxError as e:
        execution_time = time.time() - start_time
        error_message = f"語法錯誤: {e.msg} (行 {e.lineno})"
        increment_error_stats("syntax_error", current_agent)
        logger.error(f"🚨 語法錯誤 - {current_agent}: {error_message}")
        logger.error(f"執行時間: {execution_time:.2f}秒")
        return {
            "execution_stdout": stdout_capture.getvalue(),
            "execution_stderr": error_message,
            "available_files": available_files,
            "file_changes": {
                "new_files": [],
                "deleted_files": [],
                "modified_files": [],
                "summary": "error occurred",
            },
        }
    except ImportError as e:
        execution_time = time.time() - start_time
        error_message = f"導入錯誤: {str(e)}"
        increment_error_stats("import_error", current_agent)
        logger.error(f"🚨 導入錯誤 - {current_agent}: {error_message}")
        logger.error(f"執行時間: {execution_time:.2f}秒")
        return {
            "execution_stdout": stdout_capture.getvalue(),
            "execution_stderr": error_message,
            "available_files": available_files,
            "file_changes": {
                "new_files": [],
                "deleted_files": [],
                "modified_files": [],
                "summary": "error occurred",
            },
        }
    except FileNotFoundError as e:
        execution_time = time.time() - start_time
        error_message = f"文件未找到: {str(e)}"
        increment_error_stats("file_error", current_agent)
        logger.error(f"🚨 文件錯誤 - {current_agent}: {error_message}")
        logger.error(f"執行時間: {execution_time:.2f}秒")
        return {
            "execution_stdout": stdout_capture.getvalue(),
            "execution_stderr": error_message,
            "available_files": available_files,
            "file_changes": {
                "new_files": [],
                "deleted_files": [],
                "modified_files": [],
                "summary": "error occurred",
            },
        }
    except Exception as e:
        execution_time = time.time() - start_time
        stderr = stderr_capture.getvalue()
        error_type = type(e).__name__
        error_message = f"運行時錯誤 ({error_type}): {str(e)}"
        if stderr:
            error_message += f"\n詳細信息:\n{stderr}"

        increment_error_stats("runtime_error", current_agent)
        logger.error(f"🚨 運行時錯誤 - {current_agent}: {error_type}")
        logger.error(f"錯誤詳情: {str(e)}")
        logger.error(f"執行時間: {execution_time:.2f}秒")

        return {
            "execution_stdout": stdout_capture.getvalue(),
            "execution_stderr": error_message,
            "available_files": available_files,
            "file_changes": {
                "new_files": [],
                "deleted_files": [],
                "modified_files": [],
                "summary": "error occurred",
            },
        }
    finally:
        os.chdir(original_cwd)


# --- 3. 工具函式：保存報告檔案 (Tool Function: Save Report File) ---


def save_report_file(filename: str, content: str, workspace_paths: Dict) -> str:
    """
    保存報告檔案到工作區。
    """
    try:
        # 使用工作區根目錄作為保存位置
        workspace_dir = workspace_paths.get("workspace", "./kaggle_workspace")
        file_path = os.path.join(workspace_dir, filename)

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)

        logger.info(f"報告已保存至: {file_path}")
        return f"Report saved successfully to: {file_path}"
    except Exception as e:
        error_msg = f"Error saving report file: {str(e)}"
        logger.error(error_msg)
        return error_msg


# --- 4. 定義 AI 代理 (Define the AI Agents) ---


def create_agent_node(system_prompt: str, agent_name: str):
    @traceable(name=agent_name)
    def agent_node(state: KaggleWorkflowState) -> Dict:
        logger.debug(f"正在呼叫代理: {agent_name}")
        task_description = state.get("current_task_description", "")

        # Enhanced context with file awareness
        workspace_paths = state.get("workspace_paths", {})
        available_files = state.get("available_files", {})

        # Generate file context for the agent
        file_context = generate_file_context_string(workspace_paths, available_files)

        # Enhance task description with file context
        enhanced_task_description = f"""
{task_description}

{file_context}

IMPORTANT: Use the file paths exactly as shown above. Always check the available files before referencing them in your code.
"""

        # Create structured LLM for code-generating agents
        structured_llm = create_structured_llm(CODE_AGENT_SCHEMA)

        # 為Model Architect注入目標欄位資訊
        if agent_name == "Model_Architect_Agent":
            target_column = state.get("target_column", "target")
            # 替換所有的模板變數，避免ChatPromptTemplate錯誤
            enhanced_prompt = system_prompt.replace("{target_column}", target_column)
            # 處理示例代碼中的四重大括號變數（將它們轉換為雙重大括號以便在f-string中正確顯示）
            enhanced_prompt = enhanced_prompt.replace(
                "{{{{target_col}}}}", "{{target_col}}"
            )
            enhanced_prompt = enhanced_prompt.replace(
                "{{{{possible_targets}}}}", "{{possible_targets}}"
            )

            try:
                # Try structured output first
                response = structured_llm.invoke(
                    [
                        SystemMessage(content=enhanced_prompt),
                        HumanMessage(content=enhanced_task_description),
                    ]
                )

                if validate_json_response(response, agent_name):
                    logger.debug(f"✅ {agent_name} 成功產生結構化回應")
                    return {
                        "code_to_execute": response.get("code_to_execute", ""),
                        "last_code_generating_agent": agent_name,
                    }
                else:
                    raise ValueError("Invalid JSON response")

            except Exception as e:
                logger.error(f"❌ {agent_name} JSON回應失敗: {e}")
                logger.debug(f"🔄 {agent_name} 降級為傳統文本模式")

                # Fallback to original approach
                sanitized_system_prompt = _escape_braces_for_template(
                    enhanced_prompt, keep=["current_task_description"]
                )
                prompt_template = ChatPromptTemplate.from_messages(
                    [
                        ("system", sanitized_system_prompt),
                        ("human", "{current_task_description}"),
                    ]
                )
                agent = prompt_template | llm
                response = agent.invoke(
                    {"current_task_description": enhanced_task_description}
                )

                return {
                    "code_to_execute": response.content,
                    "last_code_generating_agent": agent_name,
                }
        else:
            try:
                # Try structured output first
                response = structured_llm.invoke(
                    [
                        SystemMessage(content=system_prompt),
                        HumanMessage(content=enhanced_task_description),
                    ]
                )

                if validate_json_response(response, agent_name):
                    logger.debug(f"✅ {agent_name} 成功產生結構化回應")
                    return {
                        "code_to_execute": response.get("code_to_execute", ""),
                        "last_code_generating_agent": agent_name,
                    }
                else:
                    raise ValueError("Invalid JSON response")

            except Exception as e:
                logger.error(f"❌ {agent_name} JSON回應失敗: {e}")
                logger.debug(f"🔄 {agent_name} 降級為傳統文本模式")

                # Fallback to original approach
                sanitized_system_prompt = _escape_braces_for_template(
                    system_prompt, keep=["current_task_description"]
                )
                prompt_template = ChatPromptTemplate.from_messages(
                    [
                        ("system", sanitized_system_prompt),
                        ("human", "{current_task_description}"),
                    ]
                )
                agent = prompt_template | llm
                response = agent.invoke(
                    {"current_task_description": enhanced_task_description}
                )

                return {
                    "code_to_execute": response.content,
                    "last_code_generating_agent": agent_name,
                }

    return agent_node


# 代理的 Prompt 維持不變...


def project_manager_node(state: KaggleWorkflowState) -> Dict:
    """Project manager agent node with robust structured response handling.

    Fixes:
    - Safely handle Pydantic BaseModel (uses model_dump / dict)
    - Handle raw string (attempt JSON extraction / fallback)
    - Avoid calling .get on BaseModel directly
    - Trim and sanitize target_column
    - Provide clearer debug logs for troubleshooting
    """
    logger.debug("正在呼叫代理: 專案經理 (增強版)")
    # NOTE: Avoid using .format on the prompt because it contains JSON braces which
    # would trigger KeyError (e.g., '{"target_column"'). If data_path injection is needed
    # explicitly add a placeholder like '{data_path}' and escape other braces. For now we
    # just use the static prompt.
    prompt = PROJECT_MANAGER_PROMPT  # previously used .format causing KeyError

    structured_llm = create_structured_llm(PROJECT_MANAGER_SCHEMA)

    def _response_to_dict(resp):
        """Normalize different response object types into a plain dict."""
        try:
            # Already dict
            if isinstance(resp, dict):
                return resp
            # Pydantic v2
            if hasattr(resp, "model_dump") and callable(getattr(resp, "model_dump")):
                return resp.model_dump()
            # Pydantic v1
            if hasattr(resp, "dict") and callable(getattr(resp, "dict")):
                return resp.dict()
            # LangChain message with .content
            if hasattr(resp, "content") and isinstance(resp.content, (str, dict)):
                inner = resp.content
                if isinstance(inner, dict):
                    return inner
                # Try to parse JSON from string
                parsed = _extract_json(inner)
                if parsed:
                    return parsed
                return {"raw_text": inner}
            # Raw string
            if isinstance(resp, str):
                parsed = _extract_json(resp)
                if parsed:
                    return parsed
                return {"raw_text": resp}
        except Exception as e:  # noqa: BLE001
            logger.error(f"❌ 轉換回應為字典失敗: {e}")
        return {}

    def _extract_json(text: str):
        import json
        import re

        if not text:
            return None
        # Try direct load
        try:
            return json.loads(text)
        except Exception:
            pass
        # Extract largest JSON object with braces
        try:
            match = re.search(r"\{[\s\S]*\}", text)
            if match:
                candidate = match.group(0)
                # Clean common issues (newline before key, trailing commas)
                candidate = re.sub(r'\n\s+(".*?":)', r" \1", candidate)
                candidate = re.sub(r",\s*([}\]])", r"\1", candidate)
                return json.loads(candidate)
        except Exception as e:  # noqa: BLE001
            logger.debug(f"JSON抽取失敗: {e}")
        return None

    try:
        response = structured_llm.invoke(
            [
                SystemMessage(content=prompt),
                HumanMessage(content=state.get("problem_statement", "")),
            ]
        )

        logger.debug(f"Project Manager raw response type: {type(response)}")
        logger.debug(
            f"Project Manager dir: {dir(response) if hasattr(response, '__dir__') else 'n/a'}"
        )
        if hasattr(response, "__dict__"):
            logger.debug(f"Project Manager __dict__: {vars(response)}")

        resp_dict = _response_to_dict(response)
        logger.debug(f"Normalized response dict: {resp_dict}")

        if not resp_dict:
            raise ValueError("Empty or unparseable response from structured_llm")

        # If schema fields nested under data key (rare), flatten
        if all(
            k in resp_dict.get("data", {})
            for k in ["target_column", "plan", "next_task_description"]
        ):
            resp_dict = resp_dict["data"]

        # Validate required keys
        missing = [
            k
            for k in ["target_column", "plan", "next_task_description"]
            if k not in resp_dict
        ]
        if missing:
            raise KeyError(
                f"Missing required keys in response: {missing}; raw keys: {list(resp_dict.keys())}"
            )

        target_column = resp_dict.get("target_column") or "target"
        if isinstance(target_column, str):
            target_column = target_column.strip().replace("\n", "").replace("\r", "")
        else:
            target_column = str(target_column)

        plan = resp_dict.get("plan", "")
        next_task = resp_dict.get("next_task_description", plan)

        logger.debug(f"✅ 專案經理成功識別目標欄位: {target_column}")
        logger.debug(f"問題類型: {resp_dict.get('problem_type')}")
        logger.debug(f"評估指標: {resp_dict.get('evaluation_metric')}")

        return {
            "current_task_description": next_task,
            "plan": plan,
            "target_column": target_column,
        }

    except Exception as e:  # noqa: BLE001
        logger.error(f"❌ 專案經理結構化處理失敗: {e}")
        logger.debug("🔄 降級為傳統文本解析模式")
        # Fallback: simple LLM invocation and heuristic parsing
        fallback_resp = llm.invoke(
            [
                SystemMessage(content=prompt),
                HumanMessage(content=state.get("problem_statement", "")),
            ]
        )
        raw_text = getattr(fallback_resp, "content", str(fallback_resp))

        # Try to extract target column heuristically
        heur_target = "target"
        patterns = [
            r"target_column\s*[:=]\s*['\"]([^'\"]+)['\"]",
            r"TARGET_COLUMN:\s*(\w+)",
        ]
        import re

        for pat in patterns:
            m = re.search(pat, raw_text, re.IGNORECASE)
            if m:
                heur_target = m.group(1)
                break
        heur_target = heur_target.strip().replace("\n", "")
        logger.debug(f"降級模式推斷目標欄位: {heur_target}")
        return {
            "current_task_description": raw_text,
            "plan": raw_text,
            "target_column": heur_target or "target",
        }


data_analysis_agent = create_agent_node(DATA_ANALYST_PROMPT, "Data_Analysis_Agent")

feature_engineer_agent = create_agent_node(
    FEATURE_ENGINEER_PROMPT, "Feature_Engineer_Agent"
)

model_architect_agent = create_agent_node(
    MODEL_ARCHITECT_PROMPT, "Model_Architect_Agent"
)

# 報告撰寫代理


def report_generator_node(state: KaggleWorkflowState) -> Dict:
    """專門處理報告生成的節點，具備增強的SHAP分析和自動化洞察提取"""
    logger.debug("正在呼叫代理: 報告撰寫代理")

    # Create structured LLm for Report Generator
    structured_llm = create_structured_llm(REPORT_GENERATOR_SCHEMA)

    # Enhanced context with file awareness
    workspace_paths = state.get("workspace_paths", {})
    available_files = state.get("available_files", {})

    # 準備上下文資訊
    context_info = []

    # 解析執行輸出中的關鍵信息
    execution_stdout = state.get("execution_stdout", "")

    # 提取SHAP分析結果
    shap_insights = ""
    if "=== SHAP Analysis Insights ===" in execution_stdout:
        shap_start = execution_stdout.find("=== SHAP Analysis Insights ===")
        shap_end = execution_stdout.find(
            "=== Automated Insights & Recommendations ===", shap_start
        )
        if shap_end == -1:
            shap_insights = execution_stdout[shap_start:]
        else:
            shap_insights = execution_stdout[shap_start:shap_end]

    # 提取自動化洞察和建議
    automated_insights = ""
    if "=== Automated Insights & Recommendations ===" in execution_stdout:
        insights_start = execution_stdout.find(
            "=== Automated Insights & Recommendations ==="
        )
        automated_insights = execution_stdout[insights_start:]

    # 添加執行輸出（包含解析的關鍵部分）
    if execution_stdout:
        context_info.append(f"Model Training Output:\n{execution_stdout}\n")

    if shap_insights:
        context_info.append(f"Extracted SHAP Insights:\n{shap_insights}\n")

    if automated_insights:
        context_info.append(f"Extracted Automated Insights:\n{automated_insights}\n")

    # 添加分數資訊
    if state.get("validation_score"):
        context_info.append(f"Validation Score: {state['validation_score']}\n")
    if state.get("test_score"):
        context_info.append(f"Test Score: {state['test_score']}\n")

    # 智能分析可用檔案
    available_files = state.get("available_files", {})
    image_files = available_files.get("image", [])

    # 分類圖表類型
    chart_analysis = {
        "shap_charts": [],
        "traditional_charts": [],
        "performance_charts": [],
    }

    for img in image_files:
        if "shap" in img.lower():
            chart_analysis["shap_charts"].append(img)
        elif "confusion_matrix" in img.lower() or "roc" in img.lower():
            chart_analysis["performance_charts"].append(img)
        else:
            chart_analysis["traditional_charts"].append(img)

    context_info.append(f"Available Chart Analysis:\n")
    context_info.append(f"- SHAP Charts: {chart_analysis['shap_charts']}\n")
    context_info.append(
        f"- Performance Charts: {chart_analysis['performance_charts']}\n"
    )
    context_info.append(
        f"- Traditional Charts: {chart_analysis['traditional_charts']}\n"
    )

    # 添加迭代歷史
    if state.get("iteration_history"):
        context_info.append("Workflow History:\n")
        for item in state["iteration_history"][-5:]:  # 只取最近5個
            context_info.append(f"- {item}\n")

    context_str = "\n".join(context_info)

    # Generate file context for the report generator
    file_context = generate_file_context_string(workspace_paths, available_files)

    # 構建完整的任務描述
    task_description = f"""
Based on the analysis workflow results, generate a comprehensive data science report with enhanced SHAP interpretability and automated insights.

Current State Information:
{context_str}

Current File System State:
{file_context}

Special Instructions for Enhanced Reporting:
1. Prioritize SHAP visualizations in the report structure
2. Extract and interpret automated insights from the execution output
3. Include quantitative analysis of feature importance comparisons
4. Generate business-actionable recommendations based on SHAP patterns
5. Address model deployment readiness and risk assessment

Please respond in JSON format with:
- code_to_execute: Python code to generate the report
- report_title: Title of the analysis report
- report_summary: Executive summary of findings
- key_findings: List of main discoveries
- recommendations: List of actionable recommendations
- charts_analyzed: List of charts that will be analyzed in the report
"""

    try:
        # Try structured output first
        response = structured_llm.invoke(
            [
                SystemMessage(content=REPORT_GENERATOR_PROMPT),
                HumanMessage(content=task_description),
            ]
        )

        if validate_json_response(response, "ReportGenerator"):
            logger.debug("✅ 報告生成代理成功產生結構化回應")
            return {
                "code_to_execute": response.get("code_to_execute", ""),
                "last_code_generating_agent": "Report_Generator_Agent",
            }
        else:
            raise ValueError("Invalid JSON response")

    except Exception as e:
        logger.error(f"❌ 報告生成代理JSON回應失敗: {e}")
        logger.debug("🔄 報告生成代理降級為傳統文本模式")

        # Fallback to original approach
        sanitized_report_prompt = _escape_braces_for_template(
            REPORT_GENERATOR_PROMPT, keep=["task_description"]
        )
        prompt_template = ChatPromptTemplate.from_messages(
            [("system", sanitized_report_prompt), ("human", "{task_description}")]
        )
        agent = prompt_template | llm
        response = agent.invoke({"task_description": task_description})

        return {
            "code_to_execute": response.content,
            "last_code_generating_agent": "Report_Generator_Agent",
        }


# **V2 更新**: 增強了策略師的 Prompt，使其能夠處理來自 Triage 節點的升級問題。


def chief_strategist_node(state: KaggleWorkflowState) -> Dict:
    logger.debug("正在呼叫代理: 首席策略師")
    context_list = []
    for key, value in state.items():
        if key not in ["strategist_decision"] and value:
            if isinstance(value, list) and value:
                context_list.append(f"- {key}:\n  - " + "\n  - ".join(map(str, value)))
            elif isinstance(value, str) and value.strip():
                context_list.append(f"- {key}:\n{value}\n")
            elif isinstance(value, (float, int)):
                context_list.append(f"- {key}: {value}")
    context_str = "\n".join(context_list)
    prompt = STRATEGIST_PROMPT.format(context_str=context_str)
    logger.debug(f"策略師的strategist_prompt: {context_str}")
    # Use the enhanced strategist schema with better validation
    json_llm = create_structured_llm(ENHANCED_STRATEGIST_SCHEMA)

    try:
        response = json_llm.invoke(
            [
                SystemMessage(content=prompt),
                HumanMessage(
                    content="請根據以上上下文，做出你的下一步決策。特別注意解析執行輸出中的 'Validation Score:', 'Test Score:' 和 'Submission file saved to:' 資訊。"
                ),
            ]
        )

        if not validate_json_response(response, "StrategistDecision"):
            raise ValueError("Invalid JSON response structure")

        logger.debug(f"✅ 策略師成功產生結構化決策: {response}")

    except Exception as e:
        logger.error(f"❌ 策略師JSON回應失敗: {e}")
        logger.debug("🔄 策略師降級為基礎結構化輸出模式")

        # Fallback to basic structured output
        fallback_schema = {
            "title": "BasicStrategistDecision",
            "type": "object",
            "properties": {
                "next_step": {"title": "Next Step", "type": "string"},
                "feedback": {"title": "Feedback", "type": "string"},
                "validation_score": {
                    "title": "Validation Score",
                    "type": ["number", "null"],
                    "description": "驗證分數",
                },
                "test_score": {
                    "title": "Test Score",
                    "type": ["number", "null"],
                    "description": "測試分數",
                },
                "submission_file_path": {
                    "title": "Submission File Path",
                    "type": "string",
                    "description": "提交檔案路徑",
                },
                "performance_analysis": {
                    "title": "Performance Analysis",
                    "type": "string",
                    "description": "性能分析",
                },
                "should_continue": {
                    "title": "Should Continue",
                    "type": "boolean",
                    "description": "是否繼續",
                },
            },
            "required": ["next_step", "feedback"],
        }

        fallback_llm = json_llm_base.with_structured_output(fallback_schema)
        response = fallback_llm.invoke(
            [
                SystemMessage(content=prompt),
                HumanMessage(
                    content="請根據以上上下文，做出你的下一步決策。特別注意解析執行輸出中的 'Validation Score:', 'Test Score:' 和 'Submission file saved to:' 資訊。"
                ),
            ]
        )
        logger.debug(f"🔄 策略師降級決策: {response}")

    # Validate and sanitize response to prevent routing errors
    response = validate_and_sanitize_strategist_response(response)
    logger.debug(f"✅ 策略師回應已驗證和清理: {response}")

    new_history = state.get("iteration_history", [])

    # 使用驗證後的結果
    validation_score = response.get("validation_score")
    test_score = response.get("test_score")
    submission_file_path = response.get("submission_file_path")
    performance_analysis = response.get("performance_analysis", "")
    # should_continue 目前未使用，可在未來擴展中實現自動化流程控制

    # 更新歷史記錄和返回值
    result = {
        "strategist_decision": response,
        "current_task_description": response["feedback"],
        "iteration_history": new_history,
    }

    if validation_score is not None:
        result["validation_score"] = validation_score
        new_history.append(f"模型訓練完成，驗證分數: {validation_score}")

    if test_score is not None:
        result["test_score"] = test_score
        new_history.append(f"測試分數: {test_score}")

    if submission_file_path:
        result["submission_file_path"] = submission_file_path
        new_history.append(f"提交檔案已保存: {submission_file_path}")

    if performance_analysis:
        new_history.append(f"性能分析: {performance_analysis}")

    if not any([validation_score, test_score, submission_file_path]):
        new_history.append(
            f"策略師決策: {response['next_step']} - {response['feedback'][:50]}..."
        )

    result["iteration_history"] = new_history
    return result


# --- 4. 定義流程圖的邊和條件邏輯 (Define Graph Edges & Conditional Logic) ---
# **V2 更新**: 新增了 Triage 節點和對應的 Router，取代了舊的條件式邊。


def triage_node(state: KaggleWorkflowState) -> Dict:
    """
    分析執行結果，決定下一步是修正、評估還是升級問題。
    """
    if state.get("execution_stderr"):
        logger.debug("偵測到程式碼錯誤，進行分流")
        error_count = state.get("error_count", 0) + 1

        # 如果連續錯誤達到 2 次，將問題升級給策略師
        if error_count >= 2:
            logger.warning(f"錯誤次數達到 {error_count}，將問題升級給策略師")
            feedback = (
                f"代理 '{state['last_code_generating_agent']}' 連續多次無法修正其程式碼錯誤。\n"
                f"這是最後一次的錯誤訊息：\n{state['execution_stderr']}\n"
                f"請分析根本原因，並制定一個全新的計畫來打破僵局。"
            )
            return {
                "error_count": 0,  # 重置計數器
                "current_task_description": feedback,
                "execution_stderr": "",  # 清除錯誤，因為現在是策略問題
                "next_node_after_triage": "Chief_Strategist_Agent",
            }
        # 如果錯誤次數尚在容許範圍，返回原代理修正
        else:
            logger.debug(f"第 {error_count} 次錯誤，返回修正")
            feedback = (
                f"你的上一段程式碼執行失敗，請修正它。\n"
                f"這是第 {error_count} 次嘗試。\n"
                f"錯誤訊息如下：\n{state['execution_stderr']}"
            )
            return {
                "error_count": error_count,
                "current_task_description": feedback,
                "next_node_after_triage": state["last_code_generating_agent"],
            }
    else:
        logger.debug("程式碼執行成功，交由策略師評估")
        return {
            "error_count": 0,  # 成功後重置錯誤計數器
            "next_node_after_triage": "Chief_Strategist_Agent",
        }


def router_after_triage(state: KaggleWorkflowState):
    """根據分流節點的決定，導向到下一個節點。"""
    destination = state.get("next_node_after_triage")
    logger.debug(f"分流結果: 前往 {destination}")
    return destination


def router_after_strategy(state: KaggleWorkflowState):
    """根據首席策略師的決策，決定下一個節點。
    增強版: 強化錯誤處理和驗證，完全防止 KeyError 和路由失敗。
    推斷邏輯:
      1. 若已有處理後資料 (after_preprocessing/*.csv) -> 進入建模階段 Model_Architect_Agent
      2. 若已有驗證與測試分數 (且 >0) -> 進入報告生成 Report_Generator_Agent
      3. 否則 (僅完成 EDA) -> 進入特徵工程 Feature_Engineer_Agent
    """
    allowed = {
        "Data_Analysis_Agent",
        "Feature_Engineer_Agent",
        "Model_Architect_Agent",
        "Report_Generator_Agent",
        "END",
    }

    # 預設安全的fallback
    safe_fallback = "Feature_Engineer_Agent"

    try:
        # 獲取策略師決策，確保安全存取
        strategist_decision = state.get("strategist_decision")
        if not strategist_decision or not isinstance(strategist_decision, dict):
            logger.warning(
                f"⚠️  Missing or invalid strategist_decision, using fallback: {safe_fallback}"
            )
            return safe_fallback

        next_step = strategist_decision.get("next_step")

        # 清理和驗證 next_step 值
        if isinstance(next_step, str):
            next_step = next_step.strip()

        # 處理常見的無效值
        invalid_values = ["", " ", ":", ": ", "None", "null", None]
        if next_step in invalid_values:
            logger.warning(
                f"⚠️  Invalid next_step value '{next_step}', inferring from workflow state"
            )
            next_step = None

        # 驗證是否為允許的值
        if next_step and next_step not in allowed:
            logger.warning(
                f"⚠️  Unrecognized next_step '{next_step}', inferring from workflow state"
            )
            next_step = None

        # 如果需要推斷下一步
        if not next_step:
            try:
                available_files = state.get("available_files", {}) or {}
                after_pre_files = available_files.get("after_preprocessing", []) or []
                has_processed = any(
                    f.endswith("_processed.csv") for f in after_pre_files
                )
                val_score = state.get("validation_score", 0) or 0
                test_score = state.get("test_score", 0) or 0

                # 智能推斷下一步
                if (val_score and val_score > 0) and (test_score and test_score > 0):
                    next_step = "Report_Generator_Agent"
                    logger.info(f"🤖 Inferred next step: {next_step} (has scores)")
                elif has_processed:
                    next_step = "Model_Architect_Agent"
                    logger.info(
                        f"🤖 Inferred next step: {next_step} (has processed data)"
                    )
                else:
                    next_step = "Feature_Engineer_Agent"
                    logger.info(f"🤖 Inferred next step: {next_step} (default)")

                # 更新狀態以記錄推斷結果
                strategist_decision["next_step"] = next_step
                state["strategist_decision"] = strategist_decision

            except Exception as e:
                logger.error(f"❌ Error during next_step inference: {e}")
                next_step = safe_fallback

        # 最終驗證 - 確保返回值絕對安全
        if next_step not in allowed:
            logger.error(
                f"❌ Final validation failed for next_step '{next_step}', using safe fallback: {safe_fallback}"
            )
            next_step = safe_fallback

        logger.debug(f"✅ Router decision: {next_step}")
        return END if next_step == "END" else next_step

    except Exception as e:
        logger.error(f"❌ Critical error in router_after_strategy: {e}")
        logger.warning(f"🚨 Using emergency fallback: {safe_fallback}")
        return safe_fallback


# --- 5. 組裝 LangGraph 流程圖 (Assemble the Graph) ---
# **V2 更新**: 修改了圖的結構，加入了 Triage 節點。

workflow = StateGraph(KaggleWorkflowState)

workflow.add_node("Project_Manager_Agent", project_manager_node)
workflow.add_node("Data_Analysis_Agent", data_analysis_agent)
workflow.add_node("Feature_Engineer_Agent", feature_engineer_agent)
workflow.add_node("Model_Architect_Agent", model_architect_agent)
workflow.add_node("Report_Generator_Agent", report_generator_node)  # 新增報告撰寫代理
workflow.add_node("Code_Executor_Node", execute_code)
workflow.add_node("Triage_Node", triage_node)  # 新增分流節點
workflow.add_node("Chief_Strategist_Agent", chief_strategist_node)

workflow.set_entry_point("Project_Manager_Agent")

workflow.add_edge("Project_Manager_Agent", "Data_Analysis_Agent")
workflow.add_edge("Data_Analysis_Agent", "Code_Executor_Node")
workflow.add_edge("Feature_Engineer_Agent", "Code_Executor_Node")
workflow.add_edge("Model_Architect_Agent", "Code_Executor_Node")
workflow.add_edge(
    "Report_Generator_Agent", "Code_Executor_Node"
)  # 報告撰寫代理也需要執行程式碼

# 程式碼執行後，總是先到 Triage 節點進行分流
workflow.add_edge("Code_Executor_Node", "Triage_Node")

# Triage 節點後的條件式路由
workflow.add_conditional_edges(
    "Triage_Node",
    router_after_triage,
    {
        "Data_Analysis_Agent": "Data_Analysis_Agent",
        "Feature_Engineer_Agent": "Feature_Engineer_Agent",
        "Model_Architect_Agent": "Model_Architect_Agent",
        "Report_Generator_Agent": "Report_Generator_Agent",
        "Chief_Strategist_Agent": "Chief_Strategist_Agent",
    },
)

# 策略師節點後的條件式路由
workflow.add_conditional_edges(
    "Chief_Strategist_Agent",
    router_after_strategy,
    {
        "Data_Analysis_Agent": "Data_Analysis_Agent",
        "Feature_Engineer_Agent": "Feature_Engineer_Agent",
        "Model_Architect_Agent": "Model_Architect_Agent",
        "Report_Generator_Agent": "Report_Generator_Agent",
        END: END,
    },
)

app = workflow.compile()


# --- 6. 執行工作流程 (Run the Workflow) ---


def setup_titanic_dataset():
    logger.debug("正在準備範例資料集 (鐵達尼號)")
    data_dir = os.path.abspath("./kaggle_workspace/data")

    print(f"Deleting existing directory: {data_dir}")
    if os.path.exists(os.path.abspath("./kaggle_workspace")):
        shutil.rmtree(os.path.abspath("./kaggle_workspace"))

    os.makedirs(data_dir, exist_ok=True)
    train_url = (
        "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    )
    df = pd.read_csv(train_url)
    from sklearn.model_selection import train_test_split

    # First split: separate test set (20% of total data)
    train_val_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["Survived"]
    )

    # Second split: separate train and validation from remaining 80%
    # This gives us 60% train, 20% validation, 20% test
    train_df, val_df = train_test_split(
        train_val_df, test_size=0.25, random_state=42, stratify=train_val_df["Survived"]
    )

    # Save all three datasets with target variable intact
    train_path = os.path.join(data_dir, "train.csv")
    val_path = os.path.join(data_dir, "validation.csv")
    test_path = os.path.join(data_dir, "test.csv")

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    logger.info(f"訓練資料已儲存至: {train_path} (樣本數: {len(train_df)})")
    logger.info(f"驗證資料已儲存至: {val_path} (樣本數: {len(val_df)})")
    logger.info(f"測試資料已儲存至: {test_path} (樣本數: {len(test_df)})")
    logger.info(f"總樣本數: {len(df)}")
    return data_dir


def setup_dataset(file_path: str, base_path: str):
    logger.debug("正在準備特定資料集")
    data_dir = os.path.abspath(base_path)
    if os.path.exists(data_dir):
        print(f"Deleting existing directory: {data_dir}")
        shutil.rmtree(data_dir)

    os.makedirs(data_dir, exist_ok=True)
    df = pd.read_csv(file_path)
    # df = df.dropna(subset=["ir"])
    from sklearn.model_selection import train_test_split

    # First split: separate test set (20% of total data)
    train_val_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        # stratify=df["ir"],
    )

    # Second split: separate train and validation from remaining 80%
    # This gives us 60% train, 20% validation, 20% test
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=0.25,
        random_state=42,
        # stratify=train_val_df["ir"],
    )

    # Save all three datasets with target variable intact
    train_path = os.path.join(data_dir, "train.csv")
    val_path = os.path.join(data_dir, "validation.csv")
    test_path = os.path.join(data_dir, "test.csv")

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    logger.info(f"訓練資料已儲存至: {train_path} (樣本數: {len(train_df)})")
    logger.info(f"驗證資料已儲存至: {val_path} (樣本數: {len(val_df)})")
    logger.info(f"測試資料已儲存至: {test_path} (樣本數: {len(test_df)})")
    logger.info(f"總樣本數: {len(df)}")
    return data_dir


if __name__ == "__main__":
    USE_TANICS_DATASET = True  # 設置為 True 以使用鐵達尼號資料集，否則使用自定義資料集

    if USE_TANICS_DATASET:
        data_directory = setup_titanic_dataset()
        problem = "你好，你的任務是分析鐵達尼號資料集，預測哪些乘客能夠生還。這是一個二元分類問題，請建立一個模型並產出分析報告。target_column is 'Survived'。Don't use one-hot encoding."
        workspace_paths = setup_workspace_structure()

    # 掃描初始可用檔案
    available_files = scan_available_files(workspace_paths)

    # 設置預設目標欄位
    default_target = "target"

    initial_state = {
        "problem_statement": problem,
        "data_path": workspace_paths["data"],  # 使用設置好的資料路徑
        "target_column": default_target,  # 設置預設目標欄位
        "workspace_paths": workspace_paths,  # 添加所有路徑資訊
        "available_files": available_files,  # 添加初始檔案清單
        "iteration_history": [],
        "error_count": 0,
        "validation_score": 0.0,
        "test_score": 0.0,
        "submission_file_path": "",
        "feature_columns": [],
        "model_performance": {},
        "best_hyperparameters": {},
        "preprocessing_steps": [],
    }

    logger.info("=== 工作區初始化完成 ===")
    logger.info("可用資料夾:")
    for folder_name, folder_path in workspace_paths.items():
        logger.info(f"  {folder_name}: {folder_path}")
    logger.info("開始執行 Kaggle 工作流程...")

    final_state = None
    # 增加 recursion_limit 以應對可能的重試
    try:
        for s in app.stream(initial_state, {"recursion_limit": 60}):
            logger.debug("---")
            node_name = list(s.keys())[0]
            logger.debug(f"節點: {node_name}")
            logger.debug(f"{s[node_name]}")
            final_state = s[node_name]
    except Exception as e:
        import traceback

        print(traceback.format_exc())
        logger.error(f"工作流程執行失敗: {e}")
        logger.info("===Reach recursion_limit 75===")

    # 記錄最終結果和錯誤統計
    log_error_stats()
    logger.info("=== 工作流程執行完畢 ===")

    if final_state is not None:
        logger.info(f"最終驗證分數: {final_state.get('validation_score')}")
        logger.info(f"最終測試分數: {final_state.get('test_score')}")
        logger.info(f"提交檔案路徑: {final_state.get('submission_file_path')}")
        logger.info("迭代歷史:")
        for item in final_state.get("iteration_history", []):
            logger.info(f"- {item}")
    else:
        logger.warning("工作流程執行未能完成，沒有最終狀態資訊")
