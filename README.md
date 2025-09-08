# Quantitative Healthcare Investment Platform

A machine learning-powered platform that applies quantitative finance techniques to optimize investment portfolios in cancer-related clinical trials, addressing the critical challenge of high-risk, high-cost drug development in healthcare.

## 🎯 Problem Statement

The pharmaceutical industry faces a critical challenge known as **Eroom's Law** - the inverse of Moore's Law - where the number of drugs produced per dollar spent has been decreasing exponentially. With only **5% of clinical trials reaching Phase 4 approval**, investing in clinical trials has become an extremely high-risk proposition, often requiring $200+ million investments with uncertain outcomes.

This platform addresses this challenge by applying quantitative investment strategies to make clinical trial investments more data-driven and risk-aware.

## 🚀 Solution Overview

Our platform combines **machine learning** and **modern portfolio theory** to:

1. **Predict Clinical Trial Success**: Use ML models to assess the probability of trial success based on comprehensive trial data
2. **Optimize Investment Portfolios**: Apply Markowitz Mean-Variance Optimization to construct optimal portfolios for different risk tolerances
3. **Interactive Decision Making**: Provide an intuitive web interface for real-time portfolio optimization and risk assessment

## 🏗️ Architecture

### Data Pipeline

- **Source**: Clinical trials data from ClinicalTrials.gov (60,000+ cancer-related trials)
- **Feature Engineering**:
  - Quantitative features (sponsors, enrollment, demographics)
  - Qualitative features via sentence embeddings using Hugging Face's SentenceTransformer
- **Target**: Binary classification (Phase 4 success vs. failure)

### Machine Learning Model

- **Algorithm**: Multi-layer Perceptron (MLP) Classifier
- **Features**: 1000+ engineered features including text embeddings
- **Training**: Partial fitting for large-scale data processing
- **Output**: Risk and return predictions for each clinical trial

### Portfolio Optimization

- **Method**: Markowitz Mean-Variance Optimization
- **Implementation**: PyPortfolioOpt with L2 regularization
- **Constraints**:
  - Risk tolerance: σ
  - Portfolio weights sum to 1
  - Non-negative weights
- **Output**: Efficient frontier and optimal portfolio weights

## 📊 Key Features

- **Risk-Return Prediction**: ML model predicts trial success probability and expected returns
- **Portfolio Optimization**: Real-time portfolio construction based on user risk tolerance
- **Interactive Visualization**:
  - Efficient frontier plots
  - Portfolio allocation pie charts
  - Risk-return scatter plots
- **Clinical Trial Integration**: Direct links to trial information and documentation
- **Responsive Design**: Modern web interface built with Dash and Plotly

## 🛠️ Technology Stack

### Backend

- **Python 3.11+**
- **Data Processing**: Pandas, NumPy
- **Machine Learning**: Scikit-learn, SentenceTransformer
- **Optimization**: PyPortfolioOpt, SciPy
- **Web Framework**: Dash, Flask

### Frontend

- **Interactive Visualizations**: Plotly
- **UI Components**: Dash HTML/Core Components
- **Styling**: Custom CSS with responsive design

### Deployment

- **Web Server**: Gunicorn
- **Containerization**: Ready for Docker deployment

## 📁 Project Structure

```
├── app.py                          # Main Dash web application
├── requirements.txt                # Python dependencies
├── data/
│   ├── preprocess.py              # Data preprocessing pipeline
│   ├── efficient_frontier.csv    # Risk-return frontier data
│   ├── predicted_out.csv         # ML predictions with trial info
│   ├── weights.csv               # Portfolio weights matrix
│   └── scatter.csv               # Visualization data
├── model/
│   ├── data_cleaning.py          # Feature engineering
│   ├── model.py                  # ML model training
│   ├── test_model.py             # Model evaluation
│   └── *.ipynb                   # Development notebooks
└── Portfolio/
    ├── PortfolioOptimizer.py     # Portfolio optimization engine
    └── app.py                    # Alternative portfolio interface
```

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- pip or conda

### Installation

1. **Clone the repository**

   ```bash
   git clone <repository-url>
   cd Quantitive-HealthCare-Investment
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**

   ```bash
   python app.py
   ```

4. **Access the web interface**
   - Open your browser to `http://localhost:8050`
   - Use the risk slider to adjust your risk tolerance
   - View optimal portfolio allocations and efficient frontier

## 📈 Usage

### Portfolio Optimization

1. **Adjust Risk Tolerance**: Use the slider to set your desired risk level
2. **View Portfolio Allocation**: The pie chart shows optimal weights for each trial
3. **Analyze Efficient Frontier**: The scatter plot displays risk-return trade-offs
4. **Review Trial Details**: The data table shows specific trial information and weights

### Key Metrics

- **Risk Level**: Standard deviation of portfolio returns
- **Expected Return**: Weighted average of predicted trial returns
- **Portfolio Weights**: Optimal allocation percentages for each clinical trial

## 🔬 Technical Details

### Machine Learning Pipeline

1. **Data Collection**: Query ClinicalTrials.gov for cancer-related trials
2. **Feature Engineering**:
   - Text embeddings for trial descriptions
   - One-hot encoding for categorical variables
   - Demographic and enrollment features
3. **Model Training**: MLP classifier with partial fitting for scalability
4. **Prediction**: Generate risk and return estimates for each trial

### Portfolio Optimization

The optimization problem is formulated as:

```
maximize: w^T * μ̂
subject to: w^T * Σ̂ * w ≤ σ
           Σw_i = 1
           w_i ≥ 0
```

Where:

- `w` = portfolio weights
- `μ̂` = expected returns
- `Σ̂` = covariance matrix
- `σ` = risk tolerance

## 📊 Results

The platform successfully:

- Processes 60,000+ clinical trials
- Achieves high-dimensional feature representation (1000+ features)
- Generates efficient frontiers for various risk tolerances
- Provides real-time portfolio optimization
- Enables data-driven investment decisions in healthcare

## 🤝 Contributing

We welcome contributions! Please feel free to:

- Report bugs or issues
- Suggest new features
- Submit pull requests
- Improve documentation

## 🙏 Acknowledgments

- ClinicalTrials.gov for providing comprehensive trial data
- Hugging Face for sentence transformer models
- PyPortfolioOpt for portfolio optimization tools
- The open-source community for various supporting libraries

---

**Note**: This platform is designed for research and educational purposes. Investment decisions should be made in consultation with qualified financial and medical professionals.
