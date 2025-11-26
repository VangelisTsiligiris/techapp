# Fintech AI/ML Advisor 🤖

An interactive educational tool designed for MSc students studying Fintech principles. This Streamlit app helps students understand AI and ML options and guides them in selecting the best approach for their fintech ideas.

## Features

### 🧠 AI Types
- **Generative AI**: Comprehensive coverage of LLMs, GANs, VAEs, and their fintech applications
- **Analytical AI**: Detailed exploration of ML-based analysis and prediction systems
- Side-by-side comparison with pros, cons, and use cases

### 📊 ML Algorithms Library
- **Supervised Learning**: Linear/Logistic Regression, Decision Trees, Random Forest, XGBoost, SVM, Neural Networks
- **Unsupervised Learning**: K-Means, Hierarchical Clustering, DBSCAN, PCA, Autoencoders
- **Reinforcement Learning**: Q-Learning, DQN, Policy Gradient Methods
- **Time Series & Specialized**: ARIMA, LSTM, Prophet, Isolation Forest

### 🎯 Interactive Decision Advisor
Answer questions about your project to receive:
- Personalized algorithm recommendations with scores
- Domain-specific insights
- Implementation guidance
- Next steps checklist

### 💼 Fintech Use Cases
Real-world applications across:
- Payments & Transactions
- Lending & Credit
- Investment & Trading
- Insurance
- RegTech & Compliance
- Customer Experience

### 📈 Algorithm Comparison
- Visual comparisons of complexity and interpretability
- Trade-off analysis charts
- Interactive exploration tools

## Installation

1. Clone or download this repository

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the App

### Local Development
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

### Streamlit Cloud Deployment

1. Push this code to a GitHub repository

2. Go to [Streamlit Cloud](https://share.streamlit.io/)

3. Click "New app" and connect your GitHub repository

4. Select the repository, branch, and `app.py` as the main file

5. Click "Deploy"

## Usage Guide

### For Students

1. **Start with Home**: Get an overview of the tool and learning path

2. **Explore AI Types**: Understand the fundamental difference between Generative and Analytical AI

3. **Browse Algorithms**: Learn about different ML algorithms, their strengths, and applications

4. **Use the Decision Advisor**: 
   - Answer 7 questions about your fintech idea
   - Receive tailored recommendations
   - Get domain-specific insights

5. **Study Use Cases**: See real-world applications in your area of interest

6. **Compare Algorithms**: Visualize trade-offs to make informed decisions

### For Instructors

- Use the **AI Types** section to introduce concepts
- Assign students to explore specific algorithm categories
- Have students use the **Decision Advisor** for their project proposals
- Use the **Fintech Use Cases** for case study discussions
- Reference the **Algorithm Comparison** for trade-off discussions

## Customization

### Adding New Algorithms

Edit the `ML_ALGORITHMS` dictionary in `app.py`:

```python
ML_ALGORITHMS = {
    "Category Name": {
        "Algorithm Name": {
            "type": "Classification/Regression/etc",
            "description": "Brief description",
            "fintech_use_cases": ["Use case 1", "Use case 2"],
            "pros": ["Pro 1", "Pro 2"],
            "cons": ["Con 1", "Con 2"],
            "data_requirements": "Requirements",
            "complexity": "Low/Medium/High",
            "interpretability": "Low/Medium/High"
        }
    }
}
```

### Adding New Fintech Domains

Edit the `FINTECH_DOMAINS` dictionary:

```python
FINTECH_DOMAINS = {
    "Domain Name": {
        "problems": ["Problem 1", "Problem 2"],
        "recommended_approaches": ["Approach 1", "Approach 2"]
    }
}
```

## Technical Details

- **Framework**: Streamlit
- **Visualizations**: Plotly
- **Data Handling**: Pandas
- **Python**: 3.8+

## License

This educational tool is provided for academic use. Feel free to modify and adapt for your teaching needs.

## Support

For questions or suggestions, please contact your course instructor or submit issues through your institution's channels.
