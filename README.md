# ClinsightAI - Clinical Hospital Insight Analysis

## 🏥 Project Overview

**ClinsightAI** (Code_Blooded_ClinsightAI) is an advanced clinical intelligence platform designed to analyze hospital operations, patient data, and care quality metrics. This project leverages data-driven insights to identify systemic issues, recurring problems, and actionable improvement roadmaps for healthcare institutions.

The system processes hospital datasets to generate comprehensive reports at multiple analytical levels (review-level, theme-level) and creates strategic recommendations for operational improvements.

## 🎯 Key Features

- **Multi-Level Analysis**: Analyze hospital data at review, theme, and systemic levels
- **Quality Metric Identification**: Detect and track quality issues and their patterns
- **Systemic Problem Detection**: Identify recurring systemic issues in hospital operations
- **Action Roadmap Generation**: Create prioritized action plans for hospital improvements
- **Data Processing Pipeline**: Automated pipeline for transforming raw hospital data into actionable insights
- **JSON & CSV Output**: Generate outputs in multiple formats for further analysis and reporting

## 📁 Repository Structure
Code_Blooded_ClinsightAI/ 
├── app.py 
# Streamlit web application interface 
├── run_pipeline.py 
# Main data processing pipeline execution 
├── requirements.txt
# Python dependencies 
├── hospital.csv # Input: Raw hospital data ├── clinsight_output.json # Output: Structured insights in JSON format ├── review_level_outputs.csv # Output: Review-level analysis results ├── theme_level_outputs.csv # Output: Theme-level analysis results ├── task3_recurring_systemic.csv # Output: Recurring systemic issues analysis ├── task4_action_roadmap.csv # Output: Strategic action roadmap ├── pipeline/ # Pipeline modules directory └── README.md # Project documentation


## 📊 Data Files

### Input Data
- **hospital.csv**: Contains raw hospital operational data including patient reviews, quality metrics, and facility information

### Output Files
- **clinsight_output.json**: Structured JSON output with comprehensive insights
- **review_level_outputs.csv**: Granular analysis at individual review level
- **theme_level_outputs.csv**: Aggregated insights grouped by themes/categories
- **task3_recurring_systemic.csv**: Analysis of recurring and systemic problems
- **task4_action_roadmap.csv**: Prioritized action items and improvement recommendations

## 🚀 Getting Started

### Prerequisites
- Python 3.7+
- pip (Python package manager)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/lavender2412/Code_Blooded_ClinsightAI.git
cd Code_Blooded_ClinsightAI

pip install -r requirements.txt

Dependencies

The project requires the following Python packages:

For detailed dependencies, see requirements.txt.
💻 Usage

Running the Data Processing Pipeline

Execute the main pipeline to process hospital data and generate all outputs:

bash
python run_pipeline.py
This will:

Load hospital data from hospital.csv
Process and analyze the data through multiple analysis layers
Generate CSV outputs (review-level, theme-level, systemic, and roadmap)
Create structured JSON output (clinsight_output.json)
Running the Web Application

Launch the interactive Streamlit dashboard:

bash
streamlit run app.py
This provides a user-friendly interface to:

Explore hospital insights
View analysis results
Visualize key metrics
Download reports
📈 Analysis Outputs Explained

Review Level Analysis

Individual review-based metrics
Granular quality indicators
Detailed feedback and findings
Theme Level Analysis

Aggregated insights across similar issues
Category-based analysis
Pattern identification
Recurring Systemic Issues

Problems appearing across multiple reviews
Systemic root causes
Impact severity and frequency
Action Roadmap

Prioritized improvement initiatives
Implementation recommendations
Expected outcomes and metrics
🔄 Pipeline Architecture

The project uses a modular pipeline approach:

Data Ingestion: Load and validate hospital data
Data Cleaning: Standardize and prepare data for analysis
Analysis Engine: Run multi-level analytical processes
Insight Generation: Create actionable recommendations
Report Generation: Export results in multiple formats
📝 Project Workflow

Input: Hospital data (CSV format)
Processing: Run run_pipeline.py to execute analysis
Visualization: Use app.py for interactive exploration
Output: Access generated reports and CSV files
Action: Implement recommendations from action roadmap
🎓 Use Cases

Hospital quality improvement initiatives
Patient experience analysis
Operational efficiency assessment
Systemic problem identification
Strategic planning for healthcare institutions
📊 Technology Stack
Backend: Python 3.7+
Data Processing: Pandas, NumPy
Frontend: Streamlit
Data Formats: CSV, JSON
🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bug reports and feature requests.





