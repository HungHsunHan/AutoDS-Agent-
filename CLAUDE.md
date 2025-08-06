# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a Kaggle Agent - an automated data science workflow system built with LangGraph that processes datasets through a multi-agent pipeline. The system uses AI agents to perform exploratory data analysis (EDA), feature engineering, and model training/evaluation with comprehensive error handling and recovery mechanisms.

## Running the System

**Main execution command:**
```bash
python kaggle_agent.py
```

The system runs entirely through this single Python file - there are no separate build, test, or lint commands configured.

## Configuration

### Dataset Selection
Configure the dataset by modifying variables in the main execution block (lines 1013-1033):

**Built-in Titanic Dataset:**
- Set `USE_TANICS_DATASET = True` (line 1014) to use the built-in Titanic dataset
- Uses automatic train/validation/test split (60%/20%/20%)

**Custom Dataset:**
- Set `USE_TANICS_DATASET = False` 
- Update `FILE_PATH` variable with your CSV file path
- Modify the `problem` description to match your dataset and prediction task
- Adjust workspace path in `setup_workspace_structure(base_path="./YOUR_workspace")`

### LLM Configuration
The system is configured to use Azure OpenAI (lines 24-36):
- Model: `gpt-4o-mini` via Azure OpenAI
- LangSmith tracing enabled for project "TCCI_kaggle_agent"
- API keys and endpoints configured via environment variables

## High-Level Architecture

The system implements a **multi-agent workflow** using LangGraph with comprehensive error handling and decision-making capabilities.

### Core Agents
1. **Project Manager** (`project_manager_node`): Analyzes problem statement, intelligently infers target column, creates high-level execution plan
2. **Data Analyst** (`data_analysis_agent`): Performs comprehensive EDA on all three datasets, generates statistical summaries and visualizations
3. **Feature Engineer** (`feature_engineer_agent`): Processes raw data consistently across all datasets, handles missing values, creates features
4. **Model Architect** (`model_architect_agent`): Trains models with hyperparameter tuning, evaluates performance, generates predictions and analysis visualizations
5. **Report Generator** (`report_generator_agent`): Creates comprehensive markdown analysis reports with intelligently embedded visualizations
6. **Chief Strategist** (`chief_strategist_node`): Evaluates execution results, extracts performance metrics, decides workflow progression

### Support Systems
- **Code Executor** (`execute_code`): Executes Python code in controlled sandbox environment with workspace management
- **Triage Node** (`triage_node`): Handles execution errors, implements retry logic, escalates persistent issues
- **Router Functions**: Manage workflow transitions based on execution results and strategic decisions

### Workflow Structure
The system follows a **three-dataset approach** with intelligent data splitting:
- **Training set (60%)**: Model training and pattern learning
- **Validation set (20%)**: Hyperparameter tuning and model selection
- **Test set (20%)**: Final performance evaluation (never used for training)

### Workspace Organization
```
./[DATASET]_workspace/
├── data/                    # Original datasets (train.csv, validation.csv, test.csv)
├── after_preprocessing/     # Processed datasets (*_processed.csv)
├── image/                   # Generated visualizations and plots
├── model/                   # Saved models (model.pkl)
├── submission.csv          # Final predictions
└── analysis_report.md      # Comprehensive analysis report with embedded images
```

### Advanced Error Handling
- **Automatic Retry**: Failed code execution triggers retry with error feedback
- **Escalation Logic**: After 2 consecutive failures, issues are escalated to Chief Strategist
- **Error Classification**: Distinguishes between warnings and actual errors
- **Recovery Strategies**: Strategist can change approach, simplify tasks, or provide detailed guidance

### Performance Tracking & Analysis
- **Score Extraction**: Automatically parses validation and test scores from execution output
- **Performance Comparison**: Tracks improvements across iterations
- **Decision Logic**: Determines when to continue optimization vs. generate final report
- **File Path Tracking**: Monitors submission and report file generation

## External Dependencies

The system requires:
- **LangChain/LangGraph**: Agent orchestration and workflow management
- **Azure OpenAI**: LLM capabilities (gpt-4o-mini model)
- **LangSmith**: Execution tracing and monitoring
- **Standard Data Science Stack**: pandas, scikit-learn, matplotlib, seaborn
- **Advanced Analytics**: SHAP library for feature importance analysis
- **Python Standard Library**: os, sys, io, json, logging, contextlib

## Important Implementation Details

### Code Generation & Execution
- All agents generate **pure Python code** without Markdown formatting
- Code cleaning function removes LLM-generated markdown tags before execution
- Sandboxed execution environment with proper working directory management
- Windows path compatibility using forward slashes and raw strings
- No interactive plotting (`plt.show()`) - all visualizations saved to files

### Logging & Monitoring
- Comprehensive logging to `kaggle_agent.log` with timestamps
- Execution stdout/stderr capture and analysis
- LangSmith integration for workflow tracing
- Error classification (warnings vs. actual errors)

### Agent Communication
- Structured state management via `KaggleWorkflowState` TypedDict
- JSON-based strategist decisions with structured output schema
- Context-aware prompting with state information injection
- Workflow history tracking for decision making

## Target Column Intelligence

The system features intelligent target column inference with multiple strategies:

### Automatic Detection Rules
- **Titanic/Survival**: "Survived" 
- **Price Prediction**: "Price", "SalePrice"
- **Sales Prediction**: "Sales"
- **Generic Classification**: "target", "label", "class"
- **IR/Self-Discharge**: "ir" (battery datasets)

### Detection Process
1. **Primary**: Extract from Project Manager's structured response (`TARGET_COLUMN: [name]`)
2. **Fallback**: Pattern matching against problem description keywords
3. **Default**: Use "target" as ultimate fallback

## Report Generation System

The system features an intelligent Report Generator that creates comprehensive analysis reports using workflow state data and smart image embedding.

### Content Extraction Strategy
- **Primary Source**: Workflow state fields (`eda_report`, `feature_engineering_report`, `model_training_report`)
- **Performance Data**: Extracted validation/test scores and file paths
- **Historical Context**: Iteration history and execution timeline

### Smart Image Classification
Images are automatically categorized and embedded based on filename patterns:
- `train_histogram_*.png` → EDA section (feature distributions)
- `*_distribution_train_vs_validation.png` → EDA section (target variable analysis)
- `correlation_heatmap.png` → EDA section (feature correlation analysis)
- `confusion_matrix.png` → Model Evaluation section
- `feature_importance.png` → Feature Importance section
- `shap_summary.png` → SHAP Analysis section (global feature importance)
- `shap_waterfall.png` → SHAP Analysis section (individual prediction explanation)
- `shap_feature_importance.png` → SHAP Analysis section (SHAP-based feature ranking)

### Enhanced Report Structure
1. **Executive Summary**: Key findings, model performance summary, and main insights
2. **Data Overview**: Dataset characteristics, preprocessing steps, and data quality observations  
3. **Model Performance**: Detailed metrics analysis with context and interpretation
4. **Feature Importance Analysis**: Traditional feature importance analysis with business interpretation
5. **SHAP Interpretability Analysis**: 
   - Global feature importance from SHAP values
   - Individual prediction explanations using waterfall plots
   - Key insights from SHAP analysis
   - Comparison between traditional and SHAP-based feature importance
6. **Automated Insights & Recommendations**: 
   - Data-driven insights extracted from model behavior
   - Business actionable recommendations
   - Model limitations and improvement suggestions
   - Risk factors and deployment considerations

## Enhanced SHAP Integration & Automated Insights

### SHAP Analysis Capabilities
The system now includes comprehensive SHAP (SHapley Additive exPlanations) integration for model interpretability:

**SHAP Visualizations Generated:**
- **Summary Plot** (`shap_summary.png`): Global feature importance showing how each feature affects predictions
- **Feature Importance Plot** (`shap_feature_importance.png`): SHAP-based feature ranking for top contributing features  
- **Waterfall Plot** (`shap_waterfall.png`): Individual prediction explanation showing feature contributions

**SHAP Analysis Features:**
- Automatic SHAP explainer initialization based on model type
- Top 10 SHAP feature importance ranking with quantitative scores
- Comparison between traditional and SHAP-based feature importance
- Sample-based SHAP value computation for efficiency

### Automated Insights Generation
The Model Architect agent now generates automated insights and recommendations:

**Performance Analysis:**
- Validation vs. test score comparison for overfitting detection
- Performance categorization (excellent >0.85, needs improvement <0.70)
- Automatic recommendations based on score patterns

**Feature Insights:**
- Traditional vs. SHAP feature importance comparison
- Top 3 features analysis from both methodologies
- Feature contribution patterns and business interpretation

**Business Recommendations:**
- Deployment readiness assessment
- Model complexity and risk level evaluation
- Data quality and process optimization suggestions
- Cross-validation recommendations based on performance gaps

**Enhanced Report Generation:**
- Intelligent parsing of SHAP analysis results
- Automated chart categorization and embedding
- Context-aware business interpretation of model behavior
- Actionable deployment recommendations and risk considerations

## Workflow Execution Control

### Strategic Decision Making
The Chief Strategist uses structured JSON output to control workflow progression:
- `next_step`: Target agent or END signal
- `feedback`: Detailed task instructions
- `validation_score`/`test_score`: Extracted performance metrics
- `performance_analysis`: Model evaluation summary
- `should_continue`: Optimization vs. completion decision

### Execution Limits
- Recursion limit: 75 iterations to prevent infinite loops
- Error escalation: Maximum 2 retry attempts before strategist intervention
- Performance thresholds: Configurable score targets for completion decisions