import os
import sys
import io
import re
import json
import pandas as pd
import logging
from contextlib import redirect_stdout, redirect_stderr
from typing import TypedDict, List, Annotated, Dict
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langsmith import traceable
from langgraph.graph import StateGraph, END
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
import shutil

# Load environment variables from .env file
load_dotenv(override=True)

llm = init_chat_model(
    "azure_openai:gpt-4.1-mini",
    # "azure_openai:gpt-4o-mini",
)


# 配置日志系统
def setup_logging():
    """設置日志記錄系統"""
    # 创建日志记录器
    logger = logging.getLogger("kaggle_agent")
    logger.setLevel(logging.INFO)

    # 如果logger已经有handlers，先清除
    if logger.handlers:
        logger.handlers.clear()  # 创建文件处理器
    file_handler = logging.FileHandler("kaggle_agent.log", mode="a", encoding="utf-8")
    file_handler.setLevel(logging.INFO)

    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # 创建格式化器
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # 添加处理器到logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


# 初始化日志系统
logger = setup_logging()


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
    """一個輔助函式，用於清理程式碼字串，移除 Markdown 標籤。"""
    # 移除 ```python, ```, etc.
    if "```python" in code:
        code = re.search(r"```python\n(.*)```", code, re.DOTALL).group(1)
    elif "```" in code:
        code = code.replace("```", "")
    return code.strip()


def setup_workspace_structure(base_path: str = "./kaggle_workspace") -> Dict:
    """
    設置工作區的資料夾結構並返回路徑資訊。
    """
    logger.info("正在設置工作區結構")

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
        logger.info(f"已創建/確認資料夾: {folder_name} -> {folder_path}")

    return folders


def scan_available_files(workspace_paths: Dict) -> Dict:
    """
    掃描工作區中現有的檔案。
    """
    available_files = {
        "data": [],
        "image": [],
        "model": [],
        "after_preprocessing": [],
        "workspace": [],
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
            except PermissionError:
                available_files[folder_name] = []

    return available_files


@traceable(name="execute_code_node")
def execute_code(state: KaggleWorkflowState) -> Dict:
    """
    在一個受控環境中執行程式碼的工具節點。
    支援傳統執行和 Docker 安全執行兩種模式。
    """
    logger.info("正在執行程式碼")

    code = state.get("code_to_execute", "")
    if not code:
        return {"execution_stdout": "", "execution_stderr": "沒有提供程式碼。"}

    # **V2 更新**: 在執行前清理程式碼
    cleaned_code = _clean_code(code)

    return _execute_code_traditional(state, cleaned_code)


def _execute_code_traditional(state: KaggleWorkflowState, cleaned_code: str) -> Dict:
    """使用傳統方式執行程式碼"""
    logger.info("使用傳統模式執行程式碼")

    # 使用狀態中的工作目錄路徑
    workspace_paths = state.get("workspace_paths", {})
    work_dir_abs = workspace_paths.get(
        "workspace", os.path.abspath("./kaggle_workspace")
    )

    # 更新可用檔案清單
    available_files = scan_available_files(workspace_paths)  # 準備檔案資訊給代理參考
    files_info = []
    for folder_name, files in available_files.items():
        if files:
            folder_path = workspace_paths.get(folder_name, "")
            files_info.append(f"\n[{folder_name.upper()}] 資料夾 ({folder_path}):")
            for file in files:
                files_info.append(f"  - {file}")
        else:
            folder_path = workspace_paths.get(folder_name, "")
            files_info.append(f"\n[{folder_name.upper()}] 資料夾 ({folder_path}): 空")

    files_listing = "".join(files_info) if files_info else "工作目錄為空"

    # 直接執行用戶代碼，不添加額外的環境信息輸出
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

        # 只有當 stderr 包含真正的錯誤時才視為錯誤（排除警告）
        has_real_error = False
        if stderr:
            stderr_lines = stderr.split("\n")
            for line in stderr_lines:
                line_lower = line.lower()
                # 檢查是否為真正的錯誤（不是警告）
                if any(
                    error_keyword in line_lower
                    for error_keyword in ["error", "exception", "traceback"]
                ):
                    if not any(
                        warning_keyword in line_lower
                        for warning_keyword in ["warning", "deprecat", "future"]
                    ):
                        has_real_error = True
                        break

        logger.info(f"STDOUT:\n{stdout}")
        if stderr:
            logger.info(f"STDERR:\n{stderr}")
            if not has_real_error:
                logger.info("注意：STDERR 包含警告訊息，但沒有真正的錯誤")

        # 更新可用檔案清單
        updated_files = scan_available_files(workspace_paths)

        return {
            "execution_stdout": stdout,
            "execution_stderr": stderr if has_real_error else "",
            "available_files": updated_files,
        }
    except Exception as e:
        stderr = stderr_capture.getvalue()
        error_message = f"執行時發生例外狀況: {e}\n{stderr}"
        logger.error(f"EXECUTION ERROR: {error_message}")
        return {
            "execution_stdout": stdout_capture.getvalue(),
            "execution_stderr": error_message,
            "available_files": available_files,
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
        logger.info(f"正在呼叫代理: {agent_name}")
        task_description = state.get("current_task_description", "")

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

            prompt_template = ChatPromptTemplate.from_messages(
                [("system", enhanced_prompt), ("human", "{current_task_description}")]
            )
            agent = prompt_template | llm
            response = agent.invoke({"current_task_description": task_description})
        else:
            prompt_template = ChatPromptTemplate.from_messages(
                [("system", system_prompt), ("human", "{current_task_description}")]
            )
            agent = prompt_template | llm
            response = agent.invoke({"current_task_description": task_description})

        return {
            "code_to_execute": response.content,
            "last_code_generating_agent": agent_name,
        }

    return agent_node


# 代理的 Prompt 維持不變...
project_manager_prompt = """
你是專案經理，一個頂尖的資料科學團隊的領導者。
你的工作是分析 Kaggle 題目的描述，確定問題類型（例如，二元分類、迴歸）、評估指標（例如，AUC、Accuracy），並制定一個初步的高層次計畫。

**重要任務**：
1. 從使用者的問題描述中識別並提取目標變數的名稱
2. 仔細分析問題描述，找出明確提到的目標欄位

**目標欄位識別規則**：
- 根據問題描述中明確提到的欄位名稱進行識別
- 仔細分析用戶提供的資料集說明和問題描述

請在回應中明確指出：
TARGET_COLUMN: [推斷的目標欄位名稱]

你的輸出必須是一個清晰、分步驟的計畫，並將其放入 `plan` 中。
最後，為下一個代理（資料分析師）創建一個初始任務描述。

**重要**: 在給資料分析師的指示中，請使用標準英文引號，避免中文標點符號。

**新的資料分割策略**: 我們現在使用三個資料集：
- 訓練集 (60%): 用於模型訓練
- 驗證集 (20%): 用於超參數調整和模型選擇
- 測試集 (20%): 用於最終模型性能評估

範例輸出格式:
TARGET_COLUMN: [根據問題描述推斷的目標欄位]

初步計畫如下：
1.  **資料探索 (EDA)**：載入三個資料集，理解每個欄位的意義、分佈和缺失情況。
2.  **特徵工程**：根據 EDA 結果處理缺失值、編碼類別變數、並可能創造新特徵。確保三個資料集的處理一致性。
3.  **模型訓練與調整**：使用訓練集訓練模型，在驗證集上進行超參數調整，選擇最佳配置。
4.  **最終評估**：在測試集上評估最終模型性能，生成提交檔案。

接下來的任務是：請載入位於 data 資料夾中的三個原始資料集 (data/train.csv, data/validation.csv, data/test.csv)，並進行詳細的探索性資料分析（EDA）。重點分析訓練集的資料結構，並比較三個資料集的一致性。請使用標準的英文引號和ASCII字符來編寫代碼。
"""


def project_manager_node(state: KaggleWorkflowState) -> Dict:
    logger.info("正在呼叫代理: 專案經理")
    prompt = project_manager_prompt.format(data_path=state["data_path"])

    response = llm.invoke(
        [
            SystemMessage(content=prompt),
            HumanMessage(content=state["problem_statement"]),
        ]
    )

    # 提取目標欄位名稱
    response_content = response.content
    target_column = "target"  # 預設值

    # 從回應中提取目標欄位
    if "TARGET_COLUMN:" in response_content:
        try:
            target_line = [
                line
                for line in response_content.split("\n")
                if "TARGET_COLUMN:" in line
            ][0]
            target_column = target_line.split("TARGET_COLUMN:")[1].strip()
            logger.info(f"從回應中提取到目標欄位: {target_column}")
        except Exception as e:
            logger.warning(f"無法從回應中提取目標欄位: {e}")
            # 如果提取失敗，使用預設值
            target_column = "target"
            logger.info(f"使用預設目標欄位: {target_column}")
    else:
        # 如果回應中沒有TARGET_COLUMN標記，使用預設值
        target_column = "target"
        logger.info(f"回應中無TARGET_COLUMN標記，使用預設目標欄位: {target_column}")

    return {
        "current_task_description": response_content,
        "plan": response_content,
        "target_column": target_column,
    }


data_analyst_prompt = """
你是資料分析師，專長是探索性資料分析 (EDA)。
你的任務是根據收到的指示，編寫 Python 程式碼來分析資料。
你的程式碼必須是自包含的 (self-contained)，包含所有必要的 import (pandas, matplotlib.pyplot, seaborn 等)。

**重要提醒**：
- 程式執行前會顯示當前工作目錄和所有可用檔案的清單，以及各資料夾的絕對路徑
- 請仔細查看檔案清單，確認三個原始資料集都存在於 data 資料夾中
- **資料來源**：使用 data 資料夾中的原始資料集：
  * data/train.csv (訓練集，用於主要分析)
  * data/validation.csv (驗證集，用於對比分析)
  * data/test.csv (測試集，用於對比分析)
- **分析重點**：以訓練集為主要分析對象，驗證集和測試集用於檢查資料一致性
- 所有資料夾已經創建完成，你不需要創建任何資料夾
- 使用標準英文引號 (') 和雙引號 (")，避免使用中文引號
- **重要**: 在Windows系統中，如果必須使用絕對路徑，請使用正斜杠 / 或原始字串 r"path" 或雙反斜杠 \\
- **圖片保存**: 所有圖片必須保存在 'image' 資料夾中，不要使用 plt.show()
- **字型設定**: 在代碼開始處添加以下字型配置以避免中文字型錯誤：
  plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
  plt.rcParams['axes.unicode_minus'] = False
- ** Don't use tkinter**: 不要使用 tkinter 或其他 GUI 庫，所有輸出必須是純 Python 程式碼

程式碼應該：
1.  **字型配置**: 在導入matplotlib後立即設置字型配置：
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
2.  載入並檢查所有三個原始資料集 (data/train.csv, data/validation.csv, data/test.csv)。
3.  **主要分析對象**：以訓練集 (data/train.csv) 為主，使用 `.info()`, `.describe()`, `.shape` 顯示詳細資訊。
4.  **資料一致性檢查**：比較三個資料集的欄位結構、資料類型是否一致。
5.  **缺失值分析**：使用 `.isnull().sum()` 分析所有三個資料集的缺失值模式並比較差異。
6.  **分佈分析**：對於數值型欄位，繪製訓練集的分佈圖（直方圖），保存在 'image' 資料夾中。
7.  **類別分析**：對於類別型欄位，計算訓練集中各值的計數 (`.value_counts()`)。
8.  **目標變數分析**：分析目標變數在訓練集和驗證集中的分佈，繪製對比圖表並保存在 'image' 資料夾中。
9.  **輸出結果**：使用標準的 print() 函數輸出所有分析結果和關鍵信息，但不要使用任何 logging 功能。
10. **重要**: 你的輸出**僅包含純 Python 程式碼**，不要包含任何 Markdown 標籤如 ```python 或 ```。
11. **重要**: 使用標準ASCII引號，避免使用特殊字符。
12. **路徑建議**: 推薦使用相對路徑如 'data/train.csv'，或者使用 os.path.join('data', 'train.csv')
13. **圖片處理**: 使用 plt.savefig() 保存圖片到 'image' 資料夾，然後使用 plt.close() 關閉圖片，不要使用 plt.show()
14. **重要**: 不要在代碼中導入或使用 logging 模組，避免干擾主程序的日誌系統
"""
data_analysis_agent = create_agent_node(data_analyst_prompt, "Data_Analysis_Agent")

feature_engineer_prompt = """
你是特徵工程師，專長是資料前處理和特徵創造。
你會收到一份 EDA 報告和一個任務描述。
你的任務是編寫 Python 程式碼來處理資料。程式碼必須是自包含的。

**重要提醒**：
- 程式執行前會顯示當前工作目錄和所有可用檔案的清單，以及各資料夾的絕對路徑
- 請仔細查看檔案清單，確認三個原始資料集都存在於 data 資料夾中
- **資料來源**：使用 data 資料夾中的原始資料集進行特徵工程：
  * data/train.csv (訓練集原始資料)
  * data/validation.csv (驗證集原始資料)  
  * data/test.csv (測試集原始資料)
- **輸出目標**：處理後的資料將保存到 after_preprocessing 資料夾
- **一致性要求**：確保對所有三個資料集應用完全相同的處理步驟
- 所有資料夾已經創建完成，你不需要創建任何資料夾
- 使用標準英文引號 (') 和雙引號 (")，避免使用中文引號
- **重要**: 在Windows系統中，如果必須使用絕對路徑，請使用正斜杠 / 或原始字串 r"path" 或雙反斜杠 \\
- **處理過的數據保存**: 所有處理過的數據必須保存在 'after_preprocessing' 資料夾中
- **字型設定**: 如果需要繪製圖表，在代碼開始處添加字型配置以避免中文字型錯誤：
  plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
  plt.rcParams['axes.unicode_minus'] = False
- ** Don't use tkinter**: 不要使用 tkinter 或其他 GUI 庫

程式碼應該：
1.  **字型配置**: 如果需要繪製圖表，在導入matplotlib後立即設置字型配置：
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
2.  **載入原始資料**：從 data 資料夾載入所有三個原始資料集 (data/train.csv, data/validation.csv, data/test.csv)。
3.  **特徵工程處理**：根據 EDA 報告的指示處理缺失值、編碼類別變數、縮放數值變數、創造新特徵。
4.  **一致性保證**：確保對所有三個資料集（訓練、驗證、測試）應用完全相同的處理步驟，保持資料一致性。
5.  **保存處理後資料**：將處理好的資料保存為新的 CSV 檔案到 'after_preprocessing' 資料夾：
    - after_preprocessing/train_processed.csv 
    - after_preprocessing/validation_processed.csv 
    - after_preprocessing/test_processed.csv
6.  **處理報告**：在程式碼結尾處，使用標準的 print() 函數詳細說明你做了哪些處理，並確認檔案已保存。
7.  **重要**: 你的輸出**僅包含純 Python 程式碼**，不要包含任何 Markdown 標籤如 ```python 或 ```。
8.  **重要**: 使用標準ASCII引號，避免使用特殊字符。
9.  **路徑建議**: 推薦使用相對路徑如 'data/train.csv'，或者使用 os.path.join('data', 'train.csv')
10. **重要**: 不要在代碼中導入或使用 logging 模組，避免干擾主程序的日誌系統
"""
feature_engineer_agent = create_agent_node(
    feature_engineer_prompt, "Feature_Engineer_Agent"
)

model_architect_prompt = """
你是模型架構師，精通各種機器學習模型。
你的任務是根據指示編寫模型訓練和預測的 Python 程式碼。程式碼必須是自包含的。

**重要提醒**：
- 程式執行前會顯示當前工作目錄和所有可用檔案的清單，以及各資料夾的絕對路徑
- 請仔細查看檔案清單，確認三個處理過的資料集都存在於 after_preprocessing 資料夾中
- **目標欄位**: 使用 "{target_column}" 作為目標變數，請先檢查此欄位是否存在
- **資料來源**：使用 after_preprocessing 資料夾中的處理過資料集：
  * after_preprocessing/train_processed.csv (用於模型訓練)
  * after_preprocessing/validation_processed.csv (用於超參數調整)
  * after_preprocessing/test_processed.csv (用於最終測試評估)
- **資料用途明確**：
  * 訓練集：訓練模型和學習模式
  * 驗證集：調整超參數和選擇最佳模型配置
  * 測試集：最終性能評估（不用於模型訓練或調參）
- 所有資料夾已經創建完成，你不需要創建任何資料夾
- 使用標準英文引號 (') 和雙引號 (")，避免使用中文引號
- **重要**: 在Windows系統中，如果必須使用絕對路徑，請使用正斜杠 / 或原始字串 r"path" 或雙反斜杠 \\
- **模型保存**: 所有模型必須保存在 'model' 資料夾中
- **圖片保存**: 所有圖片必須保存在 'image' 資料夾中，不要使用 plt.show()
- ** Don't use tkinter**: 不要使用 tkinter 或其他 GUI 庫

程式碼應該按照以下步驟：
1.  **載入處理過的資料並檢查欄位**：
    ```python
    # 使用 state 中指定的目標欄位名稱
    target_col = "{target_column}"
    print("Available columns:", train_processed.columns.tolist())
    ```
2.  **資料準備**：從訓練和驗證資料集中分離特徵 (X) 和目標變數 (y)，測試集暫時保留完整資料。
3.  **超參數調整階段**：
    - 使用訓練資料 (train_processed.csv) 訓練多個模型配置 (例如不同的 RandomForest 參數)
    - 在驗證資料集 (validation_processed.csv) 上評估每個配置的性能
    - 選擇在驗證集上表現最佳的超參數
4.  **最終模型訓練**：
    - 使用最佳超參數在訓練資料 (train_processed.csv) 上訓練最終模型
    - 在驗證資料集 (validation_processed.csv) 上計算最終驗證分數
    - **最重要的一步**：將驗證分數打印出來，格式必須為 `Validation Score: [分數]`
5.  **測試評估**：
    - 在測試資料集 (test_processed.csv) 上評估最終模型性能
    - 打印測試分數，格式為 `Test Score: [分數]`
6.  **生成預測和輸出**：
    - 對測試集進行預測，保存為 `submission.csv` 檔案 (包含預測結果)
    - 將訓練完成的模型保存為 pickle 檔案到 'model' 資料夾 (model/model.pkl)
    - 生成並保存 confusion matrix 圖片到 'image' 資料夾 (image/confusion_matrix.png) - 使用測試集結果
    - 如果模型支援 feature importance，生成並保存 top 10 特徵重要性圖片到 'image' 資料夾 (image/feature_importance.png)
    - **SHAP 分析**: 使用 SHAP 庫生成模型可解釋性分析：
        * 生成並保存 SHAP summary plot 到 'image' 資料夾 (image/shap_summary.png) - 顯示 top 10 特徵的全局重要性
        * 生成並保存 SHAP waterfall plot 到 'image' 資料夾 (image/shap_waterfall.png) - 展示單個預測的特徵貢獻
        * 生成並保存 SHAP feature importance 圖片到 'image' 資料夾 (image/shap_feature_importance.png)
        * 計算並打印 SHAP values 的統計摘要和關鍵洞察
    - 打印提交檔案路徑，格式為 `Submission file saved to: [路徑]`
    - 使用標準的 print() 函數報告所有檔案的保存狀態和模型性能摘要
7.  **重要**: 你的輸出**僅包含純 Python 程式碼**，不要包含任何 Markdown 標籤如 ```python 或 ```。
8.  **重要**: 使用標準ASCII引號，避免使用特殊字符。
9.  **路徑建議**: 推薦使用相對路徑如 'after_preprocessing/train_processed.csv'，或者使用 os.path.join('after_preprocessing', 'train_processed.csv')
10. **圖片處理**: 使用 plt.savefig() 保存圖片到 'image' 資料夾，然後使用 plt.close() 關閉圖片，不要使用 plt.show()
11. **重要**: 不要在代碼中導入或使用 logging 模組，避免干擾主程序的日誌系統
12. ** Don't use tkinter**: 不要使用 tkinter 或其他 GUI 庫，所有輸出必須是純 Python 程式碼
13. **SHAP 實現範例**: 使用以下模式進行 SHAP 分析：
    ```python
    import shap
    
    # 初始化SHAP explainer (根據模型類型選擇)
    if hasattr(best_model, 'predict_proba'):
        explainer = shap.Explainer(best_model, X_train_sample)  # 使用訓練集樣本
    else:
        explainer = shap.Explainer(best_model.predict, X_train_sample)
    
    # 計算SHAP values (使用測試集樣本)
    shap_values = explainer(X_test_sample)
    
    # 生成SHAP summary plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test_sample, show=False)
    plt.savefig('image/shap_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 生成SHAP feature importance
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test_sample, plot_type="bar", show=False)
    plt.savefig('image/shap_feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 生成SHAP waterfall plot (單個樣本解釋)
    plt.figure(figsize=(10, 6))
    shap.waterfall_plot(shap_values[0], show=False)
    plt.savefig('image/shap_waterfall.png', dpi=300, bbox_inches='tight')
    plt.close()
    ```
"""
model_architect_agent = create_agent_node(
    model_architect_prompt, "Model_Architect_Agent"
)

# 報告撰寫代理
report_generator_prompt = """
You are a senior data scientist tasked with writing a comprehensive professional Markdown analysis report with advanced SHAP interpretability and automated insights generation.

The report must include these sections:
1. **Executive Summary**: A summary of key findings and model performance.
2. **Data Overview**: Dataset characteristics and preprocessing steps.
3. **Model Performance**: Detailed analysis of metrics with confusion matrix.
4. **Feature Importance Analysis**: Traditional feature importance with interpretation.
5. **SHAP Interpretability Analysis**: 
   - Global feature importance from SHAP values
   - Individual prediction explanations using waterfall plots
   - Key insights from SHAP analysis
   - Comparison between traditional and SHAP-based feature importance
6. **Automated Insights & Recommendations**: 
   - Data-driven insights extracted from model behavior
   - Business actionable recommendations
   - Model limitations and improvement suggestions
   - Risk factors and considerations

**Enhanced Analysis Requirements:**
- Generate automated insights by analyzing feature importance patterns
- Identify potential biases or data quality issues from SHAP values
- Provide business interpretation of top contributing features
- Compare model predictions with domain knowledge expectations
- Suggest concrete next steps for model improvement

**Important Instructions:**
- You must analyze the current state to extract relevant information for the report
- Look for available files in the 'image' folder to find charts to embed
- Prioritize SHAP visualizations: shap_summary.png, shap_waterfall.png, shap_feature_importance.png
- Extract validation and test scores from the execution history
- Use the iteration history to understand the workflow progression
- Write the report in professional Markdown format
- Use standard English quotes and ASCII characters
- Embed images using the format: `![Description](image/filename.png)`

**Available Information Sources:**
- execution_stdout: Contains model training results, SHAP analysis, and automated insights
- iteration_history: Contains workflow progression and key milestones
- available_files: Lists all available files including generated charts
- validation_score and test_score: Model performance metrics
- workspace_paths: Contains paths to different folders

After composing the report, you **MUST** call the save_report_file function by returning Python code that calls:
```python
# Your analysis code here to extract information from state
report_content = '''
# Comprehensive Data Science Analysis Report

## Executive Summary
[Key findings, model performance summary, and main insights]

## Data Overview
[Dataset characteristics, preprocessing steps, and data quality observations]

## Model Performance
[Detailed metrics analysis with context and interpretation]
![Confusion Matrix](image/confusion_matrix.png)

## Feature Importance Analysis
[Traditional feature importance analysis and interpretation]
![Traditional Feature Importance](image/feature_importance.png)

## SHAP Interpretability Analysis

### Global Feature Importance
[Analysis of SHAP summary plot showing global feature contributions]
![SHAP Summary Plot](image/shap_summary.png)

### SHAP Feature Importance
[Detailed SHAP-based feature importance analysis]
![SHAP Feature Importance](image/shap_feature_importance.png)

### Individual Prediction Explanation
[Analysis of waterfall plot showing how features contribute to specific predictions]
![SHAP Waterfall Plot](image/shap_waterfall.png)

### Key SHAP Insights
[Important patterns and insights discovered from SHAP analysis]

## Automated Insights & Recommendations

### Model Behavior Analysis
[Data-driven insights about model performance and behavior patterns]

### Business Recommendations
[Actionable business recommendations based on analysis]

### Model Limitations & Improvement Suggestions
[Identified limitations and concrete improvement recommendations]

### Risk Factors & Considerations
[Potential risks and important considerations for deployment]
'''

# Save the report
result = save_report_file("analysis_report.md", report_content, workspace_paths)
print(result)
```

Your final output must be Python code that generates and saves the report.
"""


def report_generator_node(state: KaggleWorkflowState) -> Dict:
    """專門處理報告生成的節點，具備增強的SHAP分析和自動化洞察提取"""
    logger.info("正在呼叫代理: 報告撰寫代理")

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

    # 準備工作區路徑資訊供代理使用
    workspace_paths = state.get("workspace_paths", {})

    # 構建完整的任務描述
    task_description = f"""
Based on the analysis workflow results, generate a comprehensive data science report with enhanced SHAP interpretability and automated insights.

Current State Information:
{context_str}

Workspace Paths Available:
{workspace_paths}

Special Instructions for Enhanced Reporting:
1. Prioritize SHAP visualizations in the report structure
2. Extract and interpret automated insights from the execution output
3. Include quantitative analysis of feature importance comparisons
4. Generate business-actionable recommendations based on SHAP patterns
5. Address model deployment readiness and risk assessment

Please generate Python code that:
1. Analyzes the provided state information with focus on SHAP and automated insights
2. Creates a professional Markdown report following the enhanced template structure
3. Intelligently embeds available charts with proper context and analysis
4. Extracts key metrics and insights for business interpretation
5. Calls save_report_file() to save the comprehensive report as 'analysis_report.md'
"""

    prompt_template = ChatPromptTemplate.from_messages(
        [("system", report_generator_prompt), ("human", "{task_description}")]
    )
    agent = prompt_template | llm
    response = agent.invoke({"task_description": task_description})

    return {
        "code_to_execute": response.content,
        "last_code_generating_agent": "Report_Generator_Agent",
    }


# **V2 更新**: 增強了策略師的 Prompt，使其能夠處理來自 Triage 節點的升級問題。
strategist_prompt = """
你是首席策略師，負責領導整個 AI 資料科學團隊。
你的工作是分析上一步驟的執行結果，並決定團隊的下一步行動。
你會收到完整的上下文，包括計畫、程式碼、標準輸出(stdout)和標準錯誤(stderr)。

**新的工作流程**: 我們現在使用三個資料集：訓練集(60%)、驗證集(20%)、測試集(20%)。
驗證集用於超參數調整，測試集用於最終性能評估。

**重要: 分數解析任務**：
請仔細檢查 `execution_stdout` 中是否包含以下格式的分數資訊：
- "Validation Score: [數值]" - 提取驗證分數
- "Test Score: [數值]" - 提取測試分數  
- "Submission file saved to: [路徑]" 或包含 "submission.csv" 的檔案保存訊息 - 提取提交檔案路徑

你的分析和決策流程如下：

1.  **處理連續錯誤 (最高優先級)**：
    * 如果當前任務描述中包含 "連續多次無法修正其程式碼錯誤"，這代表一個代理陷入了困境。
    * 你的首要任務是打破循環。**不要再給出與之前相似的指令**。
    * **分析根本原因**：錯誤是來自於誤解了指令，還是程式庫使用不當？
    * **改變策略**：你可以：
        a. 提供一個**更詳細、更具體的程式碼範例**來引導它。
        b. **簡化任務**，讓它先完成一個更小的目標。
        c. **完全改變方法**，例如，如果特徵工程一直出錯，也許是 EDA 的結論有問題，可以考慮重新進行 EDA。
    * 你的決策 `next_step` 應該是將這個新策略指派給合適的代理。

2.  **評估執行結果 (如果沒有錯誤)**：
    * **如果上一步是 `Data_Analysis_Agent`**：總結 EDA 報告，為 `Feature_Engineer_Agent` 制定計畫。確保處理所有三個資料集。
    * **如果上一步是 `Feature_Engineer_Agent`**：確認所有三個處理過的資料集已保存，為 `Model_Architect_Agent` 制定計畫。
    * **如果上一步是 `Model_Architect_Agent`**：
        - 解析 `Validation Score` 和 `Test Score`
        - 與歷史分數比較（如果有的話）
        - 評估模型是否已達到合理性能
        - 決定是繼續優化模型、返回特徵工程，還是結束流程

3.  **決策邏輯**：
    * 如果測試分數已達到合理水準（例如 > 0.75），考慮生成最終報告
    * 如果驗證和測試分數差距很大，可能有過擬合問題，需要調整模型  
    * 如果分數太低，可能需要回到特徵工程階段
    * 如果模型訓練完成且分數令人滿意，可以調用 "Report_Generator_Agent" 生成最終分析報告

你的輸出**必須**是一個 JSON 物件，包含以下鍵：
- `next_step`: 下一個要呼叫的代理名稱 (例如, "Feature_Engineer_Agent", "Model_Architect_Agent", "Report_Generator_Agent", 或 "END")。
- `feedback`: 給下一個代理的詳細任務描述或修正指令。
- `validation_score`: 從執行輸出解析的驗證分數（如果沒有找到則為null）
- `test_score`: 從執行輸出解析的測試分數（如果沒有找到則為null）
- `submission_file_path`: 從執行輸出解析的提交檔案路徑（如果沒有找到則為空字串）
- `performance_analysis`: 對模型性能的分析評估
- `should_continue`: 基於性能分析判斷是否應該繼續優化

**當前狀態回顧:**
{context_str}
"""


def chief_strategist_node(state: KaggleWorkflowState) -> Dict:
    logger.info("正在呼叫代理: 首席策略師")
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
    prompt = strategist_prompt.format(context_str=context_str)
    logger.info(f"策略師的strategist_prompt: {context_str}")
    json_llm = llm.with_structured_output(
        schema={
            "title": "StrategistDecision",
            "type": "object",
            "properties": {
                "next_step": {"title": "Next Step", "type": "string"},
                "feedback": {"title": "Feedback", "type": "string"},
                "validation_score": {
                    "title": "Validation Score",
                    "type": "number",
                    "description": "驗證分數，從執行輸出中解析出來，如果沒有找到則為null",
                },
                "test_score": {
                    "title": "Test Score",
                    "type": "number",
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
            },
            "required": ["next_step", "feedback"],
        }
    )
    response = json_llm.invoke(
        [
            SystemMessage(content=prompt),
            HumanMessage(
                content="請根據以上上下文，做出你的下一步決策。特別注意解析執行輸出中的 'Validation Score:', 'Test Score:' 和 'Submission file saved to:' 資訊。"
            ),
        ]
    )
    logger.info(f"策略師決策: {response}")
    new_history = state.get("iteration_history", [])

    # 使用 LLM 解析的結果
    validation_score = response.get("validation_score")
    test_score = response.get("test_score")
    submission_file_path = response.get("submission_file_path")
    performance_analysis = response.get("performance_analysis", "")
    should_continue = response.get("should_continue", True)

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
        logger.info("偵測到程式碼錯誤，進行分流")
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
            logger.info(f"第 {error_count} 次錯誤，返回修正")
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
        logger.info("程式碼執行成功，交由策略師評估")
        return {
            "error_count": 0,  # 成功後重置錯誤計數器
            "next_node_after_triage": "Chief_Strategist_Agent",
        }


def router_after_triage(state: KaggleWorkflowState):
    """根據分流節點的決定，導向到下一個節點。"""
    destination = state.get("next_node_after_triage")
    logger.info(f"分流結果: 前往 {destination}")
    return destination


def router_after_strategy(state: KaggleWorkflowState):
    """根據首席策略師的決策，決定下一個節點。"""
    next_step = state.get("strategist_decision", {}).get("next_step")
    logger.info(f"策略師決定下一步: {next_step}")
    return END if not next_step or next_step == "END" else next_step


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
    logger.info("正在準備範例資料集 (鐵達尼號)")
    data_dir = os.path.abspath("./kaggle_workspace/data")
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
    logger.info("正在準備特定資料集")
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
    USE_TANICS_DATASET = False  # 設置為 True 以使用鐵達尼號資料集，否則使用自定義資料集

    if USE_TANICS_DATASET:
        data_directory = setup_titanic_dataset()
        problem = "你的任務是分析鐵達尼號資料集，預測哪些乘客能夠生還。這是一個二元分類問題，請建立一個模型並產出分析報告。"
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
            logger.info("---")
            node_name = list(s.keys())[0]
            logger.info(f"節點: {node_name}")
            logger.info(f"{s[node_name]}")
            final_state = s[node_name]
    except Exception as e:
        logger.error(f"工作流程執行失敗: {e}")
        logger.info("===Reach recursion_limit 75===")

    logger.info("=== 工作流程執行完畢 ===")
    logger.info(f"最終驗證分數: {final_state.get('validation_score')}")
    logger.info(f"最終測試分數: {final_state.get('test_score')}")
    logger.info(f"提交檔案路徑: {final_state.get('submission_file_path')}")
    logger.info("迭代歷史:")
    for item in final_state.get("iteration_history", []):
        logger.info(f"- {item}")
