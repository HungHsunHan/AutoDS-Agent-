# Kaggle Agent - AI-Powered Data Science Workflow Automation

![Kaggle Agent Workflow](kaggle_agent_workflow_v2.png)

An intelligent, multi-agent data science automation system that leverages LangGraph and OpenAI GPT models to automate the complete Kaggle competition workflow from data analysis to model deployment.

## 🚀 Features

### Complete Data Science Pipeline Automation
- **Exploratory Data Analysis (EDA)**: Automated data profiling, visualization, and insight generation
- **Feature Engineering**: Intelligent preprocessing, feature creation, and data transformation
- **Model Architecture**: Advanced machine learning model selection and hyperparameter tuning
- **SHAP Interpretability**: Comprehensive model explainability with SHAP analysis
- **Report Generation**: Professional analysis reports with visualizations and insights

### Multi-Agent Architecture
- **Project Manager**: Problem analysis and high-level planning
- **Data Analyst**: Comprehensive EDA and data quality assessment
- **Feature Engineer**: Data preprocessing and feature engineering
- **Model Architect**: Model training, validation, and optimization
- **Report Generator**: Professional documentation and insights
- **Chief Strategist**: Workflow orchestration and decision making

### Advanced Error Handling
- Intelligent error detection and recovery
- Multi-attempt error correction with escalation
- Workflow optimization and adaptive planning

## 🏗️ Architecture

The system uses a **LangGraph** state machine with the following workflow:

```
Project Manager → Data Analyst → Feature Engineer → Model Architect → Report Generator
                        ↑                                    ↓
                    Chief Strategist ← Code Executor ← Triage Node
```

## 📁 Project Structure

```
kaggle_agent/
├── kaggle_agent.py          # Main agent orchestration logic
├── kaggle_agent.log         # System logs
├── final.csv               # Sample output file
├── CLAUDE.md               # Development notes
├── kaggle_agent_workflow_v2.png  # Architecture diagram
├── [WORKSPACE]_workspace/   # Generated workspaces for each project
│   ├── data/               # Original datasets
│   ├── after_preprocessing/ # Processed datasets
│   ├── image/              # Generated visualizations
│   ├── model/              # Trained models
│   ├── analysis_report.md  # Final analysis report
│   └── submission.csv      # Competition submission file
└── README.md               # This file
```

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- OpenAI API key or Azure OpenAI access
- Required Python packages (see requirements below)

### Setup

1. **Clone the repository:**
```bash
git clone <your-repo-url>
cd kaggle_agent
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Set up environment variables:**
Create a `.env` file in the project root:
```env
OPENAI_API_KEY=your_openai_api_key
# Or for Azure OpenAI:
AZURE_OPENAI_API_KEY=your_azure_openai_key
AZURE_OPENAI_ENDPOINT=your_azure_endpoint
```

### Required Packages
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
pip install langchain langchain-openai langsmith langgraph
pip install shap plotly python-dotenv
```

## 🎯 Usage

### Basic Usage

1. **Prepare your dataset**: Place your data files in a directory
2. **Configure the problem**: Modify the problem statement in `kaggle_agent.py`
3. **Run the agent**:

```python
python kaggle_agent.py
```

### Configuration Options

Edit the configuration section in `kaggle_agent.py`:

```python
USE_TANICS_DATASET = False  # Use built-in Titanic dataset for testing
problem = "Your competition problem description here"
data_path = "path/to/your/dataset.csv"
```

### Custom Dataset Integration

To use your own dataset:

```python
# In kaggle_agent.py, modify the setup_dataset function call:
data_directory = setup_dataset(
    file_path="path/to/your/data.csv",
    base_path="./your_workspace"
)
```

## 📊 Output

The system generates comprehensive outputs in the workspace directory:

### Generated Files
- **`analysis_report.md`**: Professional analysis report with SHAP interpretability
- **`submission.csv`**: Ready-to-submit competition file
- **Model files**: Trained models in pickle format
- **Visualizations**: EDA plots, feature importance, SHAP plots, confusion matrices

### Sample Analysis Features
- Data distribution analysis
- Feature correlation heatmaps
- SHAP summary and waterfall plots
- Model performance metrics
- Automated insights and recommendations

## 🧠 Agent Capabilities

### Data Analysis Agent
- Automated EDA with statistical summaries
- Data quality assessment
- Missing value analysis
- Feature distribution visualization
- Class balance analysis

### Feature Engineer Agent
- Intelligent preprocessing pipeline
- Missing value handling strategies
- Feature scaling and normalization
- Categorical encoding
- Feature creation and selection

### Model Architect Agent
- Automated model selection
- Hyperparameter optimization
- Cross-validation strategies
- Performance evaluation
- SHAP-based model interpretation

### Report Generator Agent
- Professional markdown reports
- Business insights extraction
- Model deployment recommendations
- Risk assessment and limitations
- Automated visualization embedding

## 📈 Performance Examples

Based on recent runs:
- **CSRC Dataset**: 99.89% validation accuracy, 99.43% test accuracy
- **Automated SHAP Analysis**: Complete feature interpretability
- **Professional Reports**: Business-ready documentation

## 🔧 Advanced Configuration

### Model Selection
The system automatically selects appropriate models based on problem type:
- Classification: Random Forest, XGBoost, Logistic Regression
- Regression: Random Forest Regressor, XGBoost Regressor, Linear Regression

### SHAP Integration
Automated SHAP analysis includes:
- Global feature importance
- Individual prediction explanations
- Feature interaction analysis
- Business insight generation

### Error Handling
- **Smart Retry**: Up to 2 retry attempts for each agent
- **Strategic Escalation**: Complex problems escalated to Chief Strategist
- **Adaptive Planning**: Dynamic workflow adjustment based on results

## 🚨 Troubleshooting

### Common Issues

1. **API Key Issues**:
   - Verify your OpenAI/Azure API keys in `.env`
   - Check API quota and permissions

2. **Memory Issues**:
   - Reduce dataset size for initial testing
   - Increase available system memory

3. **Package Dependencies**:
   - Ensure all required packages are installed
   - Use virtual environment for isolation

### Logging
Check `kaggle_agent.log` for detailed execution logs and debugging information.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Commit changes: `git commit -am 'Add feature'`
4. Push to branch: `git push origin feature-name`
5. Submit a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **LangGraph** for the multi-agent orchestration framework
- **OpenAI** for the underlying language models
- **SHAP** for model interpretability
- **Scikit-learn** for machine learning utilities

## 📞 Contact

For questions, issues, or contributions, please open an issue on GitHub or contact the maintainer.

---

**Note**: This is an experimental AI system. Always review and validate the generated code and analysis before using in production environments.
