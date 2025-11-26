"""
Fintech AI/ML Advisor - Educational Tool for MSc Students
Helps students understand AI/ML options and select the best approach for their fintech ideas
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="Fintech AI/ML Advisor",
    page_icon="💹",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #0F4C75 0%, #3282B8 50%, #0F4C75 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 0.5rem;
        letter-spacing: -0.5px;
    }
    
    .sub-header {
        font-size: 1.1rem;
        color: #5a6c7d;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #1B262C;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #3282B8;
    }
    
    .info-box {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        border-left: 4px solid #3282B8;
        padding: 1.25rem;
        margin: 1rem 0;
        border-radius: 0 12px 12px 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }
    
    .success-box {
        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
        border-left: 4px solid #22c55e;
        padding: 1.25rem;
        margin: 1rem 0;
        border-radius: 0 12px 12px 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }
    
    .warning-box {
        background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%);
        border-left: 4px solid #f59e0b;
        padding: 1.25rem;
        margin: 1rem 0;
        border-radius: 0 12px 12px 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }
    
    .metric-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(0,0,0,0.1);
    }
    
    .metric-card h3 {
        color: #1B262C;
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    
    .metric-card p {
        color: #64748b;
        font-size: 0.9rem;
        margin: 0;
    }
    
    .metric-icon {
        font-size: 2rem;
        margin-bottom: 0.75rem;
    }
    
    .nav-item {
        padding: 0.75rem 1rem;
        border-radius: 8px;
        margin: 0.25rem 0;
        transition: background 0.2s ease;
    }
    
    .recommendation-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }
    
    .recommendation-card.top {
        border: 2px solid #3282B8;
        background: linear-gradient(135deg, #ffffff 0%, #f0f9ff 100%);
    }
    
    .score-badge {
        display: inline-block;
        background: linear-gradient(135deg, #3282B8 0%, #0F4C75 100%);
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    
    .step-number {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 28px;
        height: 28px;
        background: linear-gradient(135deg, #3282B8 0%, #0F4C75 100%);
        color: white;
        border-radius: 50%;
        font-weight: 600;
        font-size: 0.9rem;
        margin-right: 0.75rem;
    }
    
    .resource-link {
        display: block;
        padding: 0.75rem 1rem;
        background: #f8fafc;
        border-radius: 8px;
        margin: 0.5rem 0;
        text-decoration: none;
        color: #1B262C;
        border: 1px solid #e2e8f0;
        transition: all 0.2s ease;
    }
    
    .resource-link:hover {
        background: #f0f9ff;
        border-color: #3282B8;
        transform: translateX(4px);
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #f1f5f9;
        border-radius: 8px;
        padding: 8px 16px;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #3282B8 !important;
        color: white !important;
    }
    
    div[data-testid="stExpander"] {
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #1B262C 0%, #0F4C75 100%);
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# DATA DEFINITIONS
# ============================================================================

AI_TYPES = {
    "Generative AI": {
        "description": "AI systems that can create new content, including text, images, code, and synthetic data based on patterns learned from training data.",
        "key_characteristics": [
            "Creates new, original content",
            "Learns patterns and distributions from data",
            "Can generate human-like text, images, audio",
            "Uses deep learning architectures (Transformers, GANs, VAEs)"
        ],
        "fintech_applications": [
            "Automated report generation",
            "Chatbots and virtual assistants",
            "Synthetic data generation for model training",
            "Personalized financial advice generation",
            "Code generation for trading algorithms",
            "Document summarization and analysis"
        ],
        "technologies": ["GPT/LLMs", "GANs", "VAEs", "Diffusion Models", "Transformer architectures"],
        "pros": [
            "Automates content creation at scale",
            "Enhances customer experience",
            "Reduces manual effort in documentation",
            "Enables 24/7 customer support"
        ],
        "cons": [
            "Risk of hallucinations/inaccurate outputs",
            "Regulatory concerns in financial advice",
            "High computational costs",
            "Potential for misuse"
        ],
        "complexity": "High",
        "data_requirements": "Large amounts of training data",
        "interpretability": "Low (Black box)"
    },
    "Analytical AI": {
        "description": "AI systems designed to analyze data, identify patterns, make predictions, and support decision-making based on historical and real-time data.",
        "key_characteristics": [
            "Analyzes existing data to extract insights",
            "Makes predictions based on patterns",
            "Classifies and segments data",
            "Provides decision support"
        ],
        "fintech_applications": [
            "Credit scoring and risk assessment",
            "Fraud detection",
            "Algorithmic trading",
            "Customer segmentation",
            "Churn prediction",
            "Portfolio optimization",
            "Regulatory compliance monitoring"
        ],
        "technologies": ["Machine Learning", "Statistical Models", "Deep Learning", "Time Series Analysis"],
        "pros": [
            "More interpretable than generative AI",
            "Well-established regulatory frameworks",
            "Proven ROI in financial services",
            "Can provide explainable decisions"
        ],
        "cons": [
            "Requires quality historical data",
            "May not capture novel patterns",
            "Can perpetuate biases in data",
            "Needs regular retraining"
        ],
        "complexity": "Medium to High",
        "data_requirements": "Structured, labeled data typically required",
        "interpretability": "Medium to High"
    }
}

ML_ALGORITHMS = {
    "Supervised Learning": {
        "Linear Regression": {
            "type": "Regression",
            "description": "Models linear relationships between input features and continuous output.",
            "fintech_use_cases": ["Interest rate prediction", "Asset price forecasting", "Revenue prediction"],
            "pros": ["Simple and interpretable", "Fast training", "Works well for linear relationships"],
            "cons": ["Assumes linearity", "Sensitive to outliers", "May underfit complex data"],
            "data_requirements": "Continuous target variable, numerical features",
            "complexity": "Low",
            "interpretability": "High"
        },
        "Logistic Regression": {
            "type": "Classification",
            "description": "Models probability of binary outcomes using logistic function.",
            "fintech_use_cases": ["Loan default prediction", "Fraud detection (binary)", "Customer churn prediction"],
            "pros": ["Highly interpretable", "Provides probability scores", "Regulatory-friendly"],
            "cons": ["Assumes linear decision boundary", "May not capture complex patterns"],
            "data_requirements": "Binary or categorical target variable",
            "complexity": "Low",
            "interpretability": "High"
        },
        "Decision Trees": {
            "type": "Classification/Regression",
            "description": "Tree-structured model that makes decisions based on feature thresholds.",
            "fintech_use_cases": ["Credit approval decisions", "Risk categorization", "Customer segmentation"],
            "pros": ["Highly interpretable", "Handles non-linear relationships", "No feature scaling needed"],
            "cons": ["Prone to overfitting", "Can be unstable", "May not generalize well"],
            "data_requirements": "Can handle mixed data types",
            "complexity": "Low",
            "interpretability": "High"
        },
        "Random Forest": {
            "type": "Classification/Regression",
            "description": "Ensemble of decision trees that reduces overfitting through averaging.",
            "fintech_use_cases": ["Credit scoring", "Fraud detection", "Feature importance analysis"],
            "pros": ["Robust to overfitting", "Handles high-dimensional data", "Feature importance built-in"],
            "cons": ["Less interpretable than single tree", "Slower prediction", "Memory intensive"],
            "data_requirements": "Works with various data types",
            "complexity": "Medium",
            "interpretability": "Medium"
        },
        "Gradient Boosting (XGBoost/LightGBM)": {
            "type": "Classification/Regression",
            "description": "Sequential ensemble method that builds trees to correct previous errors.",
            "fintech_use_cases": ["Credit risk modeling", "Trading signal prediction", "Customer lifetime value"],
            "pros": ["State-of-the-art performance", "Handles missing values", "Feature importance"],
            "cons": ["Requires careful tuning", "Can overfit with too many trees", "Less interpretable"],
            "data_requirements": "Tabular data, can handle missing values",
            "complexity": "Medium",
            "interpretability": "Medium"
        },
        "Support Vector Machines (SVM)": {
            "type": "Classification/Regression",
            "description": "Finds optimal hyperplane to separate classes with maximum margin.",
            "fintech_use_cases": ["Fraud detection", "Credit default classification", "Market regime detection"],
            "pros": ["Effective in high dimensions", "Memory efficient", "Works well with clear margins"],
            "cons": ["Not suitable for large datasets", "Requires feature scaling", "Difficult to interpret"],
            "data_requirements": "Numerical features, requires scaling",
            "complexity": "Medium",
            "interpretability": "Low"
        },
        "Neural Networks": {
            "type": "Classification/Regression",
            "description": "Multi-layer networks that learn hierarchical representations of data.",
            "fintech_use_cases": ["Complex pattern recognition", "Image-based fraud detection", "NLP for document processing"],
            "pros": ["Captures complex patterns", "State-of-the-art for many tasks", "Automatic feature learning"],
            "cons": ["Requires large data", "Black box", "Computationally expensive"],
            "data_requirements": "Large amounts of data, GPU recommended",
            "complexity": "High",
            "interpretability": "Low"
        }
    },
    "Unsupervised Learning": {
        "K-Means Clustering": {
            "type": "Clustering",
            "description": "Partitions data into K clusters based on distance to centroids.",
            "fintech_use_cases": ["Customer segmentation", "Market regime identification", "Anomaly detection"],
            "pros": ["Simple and fast", "Scalable", "Easy to interpret clusters"],
            "cons": ["Must specify K", "Assumes spherical clusters", "Sensitive to initialization"],
            "data_requirements": "Numerical features",
            "complexity": "Low",
            "interpretability": "Medium"
        },
        "Hierarchical Clustering": {
            "type": "Clustering",
            "description": "Creates tree-like hierarchy of clusters through agglomerative or divisive approach.",
            "fintech_use_cases": ["Portfolio grouping", "Customer hierarchy", "Risk categorization"],
            "pros": ["No need to specify K", "Provides dendrogram", "Captures nested clusters"],
            "cons": ["Computationally expensive", "Not suitable for large datasets"],
            "data_requirements": "Numerical features",
            "complexity": "Medium",
            "interpretability": "High"
        },
        "DBSCAN": {
            "type": "Clustering",
            "description": "Density-based clustering that finds clusters of arbitrary shape.",
            "fintech_use_cases": ["Fraud detection", "Anomaly detection", "Geographic customer grouping"],
            "pros": ["Finds arbitrary shapes", "Identifies outliers", "No need to specify K"],
            "cons": ["Sensitive to parameters", "Struggles with varying densities"],
            "data_requirements": "Numerical features",
            "complexity": "Medium",
            "interpretability": "Medium"
        },
        "Principal Component Analysis (PCA)": {
            "type": "Dimensionality Reduction",
            "description": "Reduces dimensions by finding principal components that capture maximum variance.",
            "fintech_use_cases": ["Feature reduction", "Data visualization", "Noise reduction"],
            "pros": ["Reduces dimensionality", "Removes multicollinearity", "Speeds up training"],
            "cons": ["Loses interpretability", "Linear only", "Information loss"],
            "data_requirements": "Numerical features",
            "complexity": "Low",
            "interpretability": "Low"
        },
        "Autoencoders": {
            "type": "Dimensionality Reduction/Anomaly Detection",
            "description": "Neural networks that learn compressed representations and reconstruct data.",
            "fintech_use_cases": ["Fraud detection", "Feature extraction", "Anomaly detection"],
            "pros": ["Captures non-linear patterns", "Unsupervised", "Flexible architecture"],
            "cons": ["Requires tuning", "Black box", "Computationally expensive"],
            "data_requirements": "Large datasets preferred",
            "complexity": "High",
            "interpretability": "Low"
        }
    },
    "Reinforcement Learning": {
        "Q-Learning": {
            "type": "Model-Free RL",
            "description": "Learns optimal action-value function through trial and error.",
            "fintech_use_cases": ["Trading strategy optimization", "Portfolio rebalancing", "Dynamic pricing"],
            "pros": ["No model needed", "Can learn complex strategies", "Adapts to environment"],
            "cons": ["Requires many iterations", "May not converge", "Exploration-exploitation tradeoff"],
            "data_requirements": "Environment interaction, reward signals",
            "complexity": "High",
            "interpretability": "Low"
        },
        "Deep Q-Networks (DQN)": {
            "type": "Deep RL",
            "description": "Combines Q-learning with deep neural networks for complex state spaces.",
            "fintech_use_cases": ["Algorithmic trading", "Market making", "Order execution"],
            "pros": ["Handles complex states", "Learns from raw inputs", "Scalable"],
            "cons": ["Unstable training", "Requires extensive tuning", "Black box"],
            "data_requirements": "Large interaction data, simulation environments",
            "complexity": "Very High",
            "interpretability": "Very Low"
        },
        "Policy Gradient Methods": {
            "type": "Deep RL",
            "description": "Directly optimizes the policy function for continuous action spaces.",
            "fintech_use_cases": ["Continuous portfolio allocation", "Dynamic hedging", "Asset allocation"],
            "pros": ["Handles continuous actions", "More stable than value methods", "Direct optimization"],
            "cons": ["High variance", "Sample inefficient", "Complex implementation"],
            "data_requirements": "Continuous action space, simulation environment",
            "complexity": "Very High",
            "interpretability": "Very Low"
        }
    },
    "Time Series & Specialized": {
        "ARIMA/SARIMA": {
            "type": "Time Series",
            "description": "Autoregressive integrated moving average model for time series forecasting.",
            "fintech_use_cases": ["Stock price prediction", "Economic forecasting", "Demand forecasting"],
            "pros": ["Well-established", "Interpretable", "Handles trends and seasonality"],
            "cons": ["Assumes stationarity", "Limited to linear patterns", "Manual parameter selection"],
            "data_requirements": "Time series data",
            "complexity": "Medium",
            "interpretability": "High"
        },
        "LSTM/GRU": {
            "type": "Time Series (Deep Learning)",
            "description": "Recurrent neural networks designed to capture long-term dependencies.",
            "fintech_use_cases": ["Sequence prediction", "Price forecasting", "Transaction pattern analysis"],
            "pros": ["Captures long sequences", "Handles non-linear patterns", "Flexible"],
            "cons": ["Requires large data", "Expensive to train", "Black box"],
            "data_requirements": "Sequential data, large datasets",
            "complexity": "High",
            "interpretability": "Low"
        },
        "Prophet": {
            "type": "Time Series",
            "description": "Facebook's forecasting tool designed for business time series with seasonality.",
            "fintech_use_cases": ["Business metrics forecasting", "Revenue prediction", "Transaction volume"],
            "pros": ["Handles seasonality well", "Easy to use", "Robust to missing data"],
            "cons": ["May not capture complex patterns", "Limited customization"],
            "data_requirements": "Daily/weekly time series",
            "complexity": "Low",
            "interpretability": "Medium"
        },
        "Isolation Forest": {
            "type": "Anomaly Detection",
            "description": "Detects anomalies by isolating observations using random partitioning.",
            "fintech_use_cases": ["Fraud detection", "Outlier detection", "Unusual transaction flagging"],
            "pros": ["Fast and efficient", "Works well with high dimensions", "Unsupervised"],
            "cons": ["May miss complex anomalies", "Requires threshold tuning"],
            "data_requirements": "Numerical features",
            "complexity": "Low",
            "interpretability": "Medium"
        }
    }
}

FINTECH_DOMAINS = {
    "Payments & Transactions": {
        "problems": ["Fraud detection", "Transaction categorization", "Payment optimization"],
        "recommended_approaches": ["Gradient Boosting", "Neural Networks", "Isolation Forest"]
    },
    "Lending & Credit": {
        "problems": ["Credit scoring", "Default prediction", "Loan pricing"],
        "recommended_approaches": ["Logistic Regression", "XGBoost", "Random Forest"]
    },
    "Investment & Trading": {
        "problems": ["Price prediction", "Portfolio optimization", "Risk management"],
        "recommended_approaches": ["LSTM", "Reinforcement Learning", "ARIMA"]
    },
    "Insurance": {
        "problems": ["Risk assessment", "Claims prediction", "Pricing optimization"],
        "recommended_approaches": ["Gradient Boosting", "GLMs", "Survival Analysis"]
    },
    "RegTech & Compliance": {
        "problems": ["AML detection", "KYC automation", "Report generation"],
        "recommended_approaches": ["NLP/LLMs", "Graph Neural Networks", "Rule-based + ML hybrid"]
    },
    "Customer Experience": {
        "problems": ["Chatbots", "Personalization", "Churn prediction"],
        "recommended_approaches": ["LLMs", "Collaborative Filtering", "Classification models"]
    }
}

# Actual resource links
RESOURCES = {
    "Books": [
        {"title": "Machine Learning for Asset Managers", "author": "Marcos López de Prado", "url": "https://www.cambridge.org/core/books/machine-learning-for-asset-managers/6D9211305EA2E425D33A9F38D0AE3545"},
        {"title": "Advances in Financial Machine Learning", "author": "Marcos López de Prado", "url": "https://www.wiley.com/en-us/Advances+in+Financial+Machine+Learning-p-9781119482086"},
        {"title": "Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow", "author": "Aurélien Géron", "url": "https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/"},
        {"title": "Python for Finance", "author": "Yves Hilpisch", "url": "https://www.oreilly.com/library/view/python-for-finance/9781492024323/"},
        {"title": "Deep Learning", "author": "Ian Goodfellow et al.", "url": "https://www.deeplearningbook.org/"}
    ],
    "Online Courses": [
        {"title": "Machine Learning Specialization", "provider": "Stanford/Coursera", "url": "https://www.coursera.org/specializations/machine-learning-introduction"},
        {"title": "Deep Learning Specialization", "provider": "DeepLearning.AI/Coursera", "url": "https://www.coursera.org/specializations/deep-learning"},
        {"title": "Practical Deep Learning for Coders", "provider": "fast.ai", "url": "https://course.fast.ai/"},
        {"title": "AI for Trading", "provider": "Udacity", "url": "https://www.udacity.com/course/ai-for-trading--nd880"},
        {"title": "Financial Engineering and Risk Management", "provider": "Columbia/Coursera", "url": "https://www.coursera.org/specializations/financialengineering"}
    ],
    "Documentation & Tutorials": [
        {"title": "Scikit-learn User Guide", "description": "Comprehensive ML library documentation", "url": "https://scikit-learn.org/stable/user_guide.html"},
        {"title": "XGBoost Documentation", "description": "Gradient boosting framework", "url": "https://xgboost.readthedocs.io/en/stable/"},
        {"title": "TensorFlow Tutorials", "description": "Deep learning framework tutorials", "url": "https://www.tensorflow.org/tutorials"},
        {"title": "PyTorch Tutorials", "description": "Deep learning framework tutorials", "url": "https://pytorch.org/tutorials/"},
        {"title": "Hugging Face Course", "description": "NLP and Transformers", "url": "https://huggingface.co/learn/nlp-course"}
    ],
    "Research & Papers": [
        {"title": "arXiv Quantitative Finance", "description": "Latest research papers", "url": "https://arxiv.org/list/q-fin/recent"},
        {"title": "Journal of Financial Data Science", "description": "Academic journal", "url": "https://jfds.pm-research.com/"},
        {"title": "Papers With Code - Finance", "description": "ML papers with implementations", "url": "https://paperswithcode.com/area/finance"},
        {"title": "SSRN Financial Economics", "description": "Working papers", "url": "https://www.ssrn.com/index.cfm/en/fenetwk/"}
    ],
    "Tools & Libraries": [
        {"title": "scikit-learn", "description": "General ML algorithms", "url": "https://scikit-learn.org/"},
        {"title": "XGBoost", "description": "Gradient boosting", "url": "https://xgboost.ai/"},
        {"title": "LightGBM", "description": "Fast gradient boosting", "url": "https://lightgbm.readthedocs.io/"},
        {"title": "TensorFlow", "description": "Deep learning", "url": "https://www.tensorflow.org/"},
        {"title": "PyTorch", "description": "Deep learning", "url": "https://pytorch.org/"},
        {"title": "Prophet", "description": "Time series forecasting", "url": "https://facebook.github.io/prophet/"},
        {"title": "SHAP", "description": "Model explainability", "url": "https://shap.readthedocs.io/"},
        {"title": "statsmodels", "description": "Statistical models", "url": "https://www.statsmodels.org/"}
    ]
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def create_tradeoff_chart(algorithms):
    """Create an improved scatter plot with no overlapping labels."""
    trade_off_data = []
    
    for category, algos in algorithms.items():
        for name, details in algos.items():
            complexity_map = {"Low": 1, "Medium": 2, "Medium to High": 2.5, "High": 3, "Very High": 4}
            interpretability_map = {"Very Low": 1, "Low": 2, "Medium": 3, "High": 4}
            
            # Create shorter names for display
            short_names = {
                "Linear Regression": "Linear Reg.",
                "Logistic Regression": "Logistic Reg.",
                "Decision Trees": "Decision Tree",
                "Random Forest": "Random Forest",
                "Gradient Boosting (XGBoost/LightGBM)": "XGBoost/LGBM",
                "Support Vector Machines (SVM)": "SVM",
                "Neural Networks": "Neural Net",
                "K-Means Clustering": "K-Means",
                "Hierarchical Clustering": "Hier. Cluster",
                "DBSCAN": "DBSCAN",
                "Principal Component Analysis (PCA)": "PCA",
                "Autoencoders": "Autoencoder",
                "Q-Learning": "Q-Learning",
                "Deep Q-Networks (DQN)": "DQN",
                "Policy Gradient Methods": "Policy Grad.",
                "ARIMA/SARIMA": "ARIMA",
                "LSTM/GRU": "LSTM/GRU",
                "Prophet": "Prophet",
                "Isolation Forest": "Isolation For."
            }
            
            trade_off_data.append({
                "Algorithm": name,
                "Short_Name": short_names.get(name, name[:15]),
                "Category": category,
                "Complexity": complexity_map.get(details.get("complexity", "Medium"), 2),
                "Interpretability": interpretability_map.get(details.get("interpretability", "Medium"), 3),
                "Type": details.get("type", "")
            })
    
    df = pd.DataFrame(trade_off_data)
    
    # Create figure with Plotly Graph Objects
    fig = go.Figure()
    
    colors = {
        "Supervised Learning": "#3282B8",
        "Unsupervised Learning": "#22c55e", 
        "Reinforcement Learning": "#ef4444",
        "Time Series & Specialized": "#f59e0b"
    }
    
    # Add scatter points for each category
    for category in df["Category"].unique():
        cat_data = df[df["Category"] == category]
        fig.add_trace(go.Scatter(
            x=cat_data["Complexity"],
            y=cat_data["Interpretability"],
            mode='markers',
            name=category,
            marker=dict(
                size=16,
                color=colors.get(category, "#666"),
                line=dict(width=2, color='white'),
                opacity=0.9
            ),
            text=cat_data["Algorithm"],
            hovertemplate="<b>%{text}</b><br>" +
                          "Complexity: %{x:.1f}<br>" +
                          "Interpretability: %{y:.1f}<extra></extra>"
        ))
    
    # Smart label positioning to avoid overlaps
    # Group points by their approximate position
    label_positions = []
    for idx, row in df.iterrows():
        x, y = row["Complexity"], row["Interpretability"]
        
        # Determine text position based on location and nearby points
        # Check for conflicts
        x_anchor = "center"
        y_anchor = "bottom"
        x_offset = 0
        y_offset = 0.18
        
        # Adjust for edge cases
        if x >= 3.5:
            x_anchor = "right"
            x_offset = -0.1
        elif x <= 1.5:
            x_anchor = "left"
            x_offset = 0.1
        
        # Check for vertically stacked points
        nearby = [(lp["x"], lp["y"]) for lp in label_positions 
                  if abs(lp["x"] - x) < 0.4 and abs(lp["y"] - y) < 0.5]
        
        if nearby:
            y_offset = 0.3 + 0.15 * len(nearby)
        
        label_positions.append({
            "x": x,
            "y": y,
            "text": row["Short_Name"],
            "x_anchor": x_anchor,
            "y_anchor": y_anchor,
            "x_offset": x_offset,
            "y_offset": y_offset
        })
    
    # Add annotations with calculated positions
    annotations = []
    for lp in label_positions:
        annotations.append(dict(
            x=lp["x"] + lp["x_offset"],
            y=lp["y"] + lp["y_offset"],
            text=lp["text"],
            showarrow=False,
            font=dict(size=9, color="#374151"),
            xanchor=lp["x_anchor"],
            yanchor=lp["y_anchor"]
        ))
    
    # Add quadrant backgrounds
    fig.add_shape(type="rect", x0=0.5, y0=2.5, x1=2.25, y1=4.5,
                  fillcolor="rgba(34, 197, 94, 0.08)", line_width=0)
    fig.add_shape(type="rect", x0=2.25, y0=2.5, x1=4.5, y1=4.5,
                  fillcolor="rgba(245, 158, 11, 0.08)", line_width=0)
    fig.add_shape(type="rect", x0=0.5, y0=0.5, x1=2.25, y1=2.5,
                  fillcolor="rgba(50, 130, 184, 0.08)", line_width=0)
    fig.add_shape(type="rect", x0=2.25, y0=0.5, x1=4.5, y1=2.5,
                  fillcolor="rgba(239, 68, 68, 0.08)", line_width=0)
    
    # Add quadrant labels
    fig.add_annotation(x=1.35, y=4.35, text="◆ Simple & Explainable", 
                      showarrow=False, font=dict(size=11, color="#22c55e", weight="bold"), opacity=0.8)
    fig.add_annotation(x=3.35, y=4.35, text="◆ Complex but Explainable", 
                      showarrow=False, font=dict(size=11, color="#f59e0b", weight="bold"), opacity=0.8)
    fig.add_annotation(x=1.35, y=0.65, text="◆ Simple but Opaque", 
                      showarrow=False, font=dict(size=11, color="#3282B8", weight="bold"), opacity=0.8)
    fig.add_annotation(x=3.35, y=0.65, text="◆ Complex & Opaque", 
                      showarrow=False, font=dict(size=11, color="#ef4444", weight="bold"), opacity=0.8)
    
    fig.update_layout(
        title=dict(
            text="Algorithm Trade-offs: Complexity vs Interpretability",
            font=dict(size=18, color="#1B262C"),
            x=0.5
        ),
        xaxis=dict(
            title="Complexity",
            ticktext=["Low", "Medium", "High", "Very High"],
            tickvals=[1, 2, 3, 4],
            range=[0.5, 4.5],
            gridcolor="#e2e8f0",
            showgrid=True
        ),
        yaxis=dict(
            title="Interpretability", 
            ticktext=["Very Low", "Low", "Medium", "High"],
            tickvals=[1, 2, 3, 4],
            range=[0.5, 4.5],
            gridcolor="#e2e8f0",
            showgrid=True
        ),
        height=650,
        plot_bgcolor="white",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(255,255,255,0.8)"
        ),
        annotations=annotations,
        margin=dict(t=100, b=50)
    )
    
    return fig

def create_comparison_chart(algorithms, metric):
    """Create a comparison chart for algorithms based on a metric."""
    data = []
    for category, algos in algorithms.items():
        for name, details in algos.items():
            complexity_map = {"Low": 1, "Medium": 2, "Medium to High": 2.5, "High": 3, "Very High": 4}
            interpretability_map = {"Very Low": 1, "Low": 2, "Medium": 3, "High": 4}
            
            if metric == "complexity":
                value = complexity_map.get(details.get("complexity", "Medium"), 2)
            else:
                value = interpretability_map.get(details.get("interpretability", "Medium"), 3)
            
            # Shorter names
            short_name = name.split("(")[0].strip()
            if len(short_name) > 18:
                short_name = short_name[:15] + "..."
            
            data.append({
                "Algorithm": short_name,
                "Category": category,
                "Value": value,
                "Label": details.get(metric, "Medium")
            })
    
    df = pd.DataFrame(data)
    
    colors = {
        "Supervised Learning": "#3282B8",
        "Unsupervised Learning": "#22c55e", 
        "Reinforcement Learning": "#ef4444",
        "Time Series & Specialized": "#f59e0b"
    }
    
    fig = px.bar(df, x="Algorithm", y="Value", color="Category",
                 title=f"Algorithm Comparison by {metric.title()}",
                 hover_data=["Label"],
                 color_discrete_map=colors)
    fig.update_layout(
        xaxis_tickangle=-45, 
        height=450,
        plot_bgcolor="white",
        xaxis=dict(gridcolor="#e2e8f0"),
        yaxis=dict(gridcolor="#e2e8f0", title=metric.title())
    )
    return fig

def calculate_recommendation_scores(answers):
    """Calculate detailed scores for each algorithm based on questionnaire answers."""
    scores = {}
    explanations = {}
    
    # Initialize scores
    for category, algos in ML_ALGORITHMS.items():
        for name in algos.keys():
            scores[name] = 0
            explanations[name] = []
    
    # 1. Problem type scoring (highest weight)
    problem_type = answers.get("problem_type", "")
    if problem_type == "Binary Classification":
        for name in ["Logistic Regression", "Random Forest", "Gradient Boosting (XGBoost/LightGBM)", "Neural Networks"]:
            scores[name] = scores.get(name, 0) + 5
            explanations[name] = explanations.get(name, []) + ["Strong fit for binary classification"]
    elif problem_type == "Multi-class Classification":
        for name in ["Random Forest", "Gradient Boosting (XGBoost/LightGBM)", "Neural Networks"]:
            scores[name] = scores.get(name, 0) + 5
            explanations[name] = explanations.get(name, []) + ["Handles multi-class problems well"]
    elif problem_type == "Regression / Continuous Prediction":
        for name in ["Linear Regression", "Random Forest", "Gradient Boosting (XGBoost/LightGBM)", "Neural Networks"]:
            scores[name] = scores.get(name, 0) + 5
            explanations[name] = explanations.get(name, []) + ["Suitable for continuous predictions"]
    elif problem_type == "Clustering / Segmentation":
        for name in ["K-Means Clustering", "Hierarchical Clustering", "DBSCAN"]:
            scores[name] = scores.get(name, 0) + 5
            explanations[name] = explanations.get(name, []) + ["Designed for clustering tasks"]
    elif problem_type == "Anomaly / Fraud Detection":
        for name in ["Isolation Forest", "Autoencoders", "DBSCAN", "Gradient Boosting (XGBoost/LightGBM)"]:
            scores[name] = scores.get(name, 0) + 5
            explanations[name] = explanations.get(name, []) + ["Effective for anomaly detection"]
    elif problem_type == "Time Series Forecasting":
        for name in ["ARIMA/SARIMA", "LSTM/GRU", "Prophet"]:
            scores[name] = scores.get(name, 0) + 5
            explanations[name] = explanations.get(name, []) + ["Specialized for time series"]
    elif problem_type == "Sequential Decision Making":
        for name in ["Q-Learning", "Deep Q-Networks (DQN)", "Policy Gradient Methods"]:
            scores[name] = scores.get(name, 0) + 5
            explanations[name] = explanations.get(name, []) + ["Designed for sequential decisions"]
    elif problem_type == "Natural Language Processing":
        return [{"type": "Generative AI", "recommendation": "Large Language Models (LLMs)", 
                "reason": "NLP tasks are best handled by transformer-based models like GPT, BERT, or similar LLMs"}]
    elif problem_type == "Content / Report Generation":
        return [{"type": "Generative AI", "recommendation": "Generative AI (LLMs, GPT)", 
                "reason": "Content generation requires generative AI capabilities"}]
    
    # 2. Data size scoring
    data_size = answers.get("data_size", "")
    if data_size == "Very Small (<500 samples)":
        for name in ["Linear Regression", "Logistic Regression", "Decision Trees"]:
            scores[name] = scores.get(name, 0) + 3
            explanations[name] = explanations.get(name, []) + ["Works well with limited data"]
        for name in ["Neural Networks", "LSTM/GRU", "Deep Q-Networks (DQN)", "Autoencoders"]:
            scores[name] = scores.get(name, 0) - 3
            explanations[name] = explanations.get(name, []) + ["Requires more data than available"]
    elif data_size == "Small (500-5,000 samples)":
        for name in ["Logistic Regression", "Decision Trees", "Random Forest"]:
            scores[name] = scores.get(name, 0) + 2
        for name in ["Neural Networks", "LSTM/GRU"]:
            scores[name] = scores.get(name, 0) - 2
    elif data_size == "Large (50,000-500,000 samples)":
        for name in ["Gradient Boosting (XGBoost/LightGBM)", "Neural Networks"]:
            scores[name] = scores.get(name, 0) + 2
    elif data_size == "Very Large (>500,000 samples)":
        for name in ["Neural Networks", "LSTM/GRU", "Gradient Boosting (XGBoost/LightGBM)"]:
            scores[name] = scores.get(name, 0) + 3
            explanations[name] = explanations.get(name, []) + ["Scales well with large data"]
    
    # 3. Interpretability scoring
    interpretability = answers.get("interpretability", "")
    if interpretability == "Critical - Must explain every decision (regulatory requirement)":
        for name in ["Linear Regression", "Logistic Regression", "Decision Trees"]:
            scores[name] = scores.get(name, 0) + 4
            explanations[name] = explanations.get(name, []) + ["Highly interpretable for regulatory needs"]
        for name in ["Neural Networks", "Autoencoders", "Deep Q-Networks (DQN)", "LSTM/GRU"]:
            scores[name] = scores.get(name, 0) - 4
            explanations[name] = explanations.get(name, []) + ["May not meet interpretability requirements"]
    elif interpretability == "Important - Need to understand key drivers":
        for name in ["Random Forest", "Gradient Boosting (XGBoost/LightGBM)"]:
            scores[name] = scores.get(name, 0) + 2
            explanations[name] = explanations.get(name, []) + ["Provides feature importance insights"]
    
    # 4. Latency requirements
    latency = answers.get("latency", "")
    if latency == "Real-time (<10ms)":
        for name in ["Linear Regression", "Logistic Regression", "Decision Trees"]:
            scores[name] = scores.get(name, 0) + 2
            explanations[name] = explanations.get(name, []) + ["Fast inference for real-time use"]
        for name in ["Neural Networks", "LSTM/GRU"]:
            scores[name] = scores.get(name, 0) - 1
    elif latency == "Batch processing is fine":
        for name in ["Neural Networks", "Gradient Boosting (XGBoost/LightGBM)"]:
            scores[name] = scores.get(name, 0) + 1
    
    # 5. Data quality
    data_quality = answers.get("data_quality", "")
    if data_quality == "Poor - Lots of missing values and noise":
        for name in ["Gradient Boosting (XGBoost/LightGBM)", "Random Forest"]:
            scores[name] = scores.get(name, 0) + 2
            explanations[name] = explanations.get(name, []) + ["Handles missing values well"]
        for name in ["Linear Regression", "Support Vector Machines (SVM)"]:
            scores[name] = scores.get(name, 0) - 1
    
    # 6. Feature engineering capability
    feature_eng = answers.get("feature_engineering", "")
    if feature_eng == "Minimal - Prefer automatic feature learning":
        for name in ["Neural Networks", "Autoencoders", "LSTM/GRU"]:
            scores[name] = scores.get(name, 0) + 2
            explanations[name] = explanations.get(name, []) + ["Automatic feature learning"]
    elif feature_eng == "Extensive - Will invest in feature engineering":
        for name in ["Gradient Boosting (XGBoost/LightGBM)", "Random Forest"]:
            scores[name] = scores.get(name, 0) + 1
    
    # 7. Team expertise
    expertise = answers.get("expertise", "")
    if expertise == "Beginner - New to ML":
        for name in ["Linear Regression", "Logistic Regression", "Decision Trees", "K-Means Clustering"]:
            scores[name] = scores.get(name, 0) + 2
            explanations[name] = explanations.get(name, []) + ["Good starting point for beginners"]
        for name in ["Neural Networks", "Deep Q-Networks (DQN)", "Policy Gradient Methods"]:
            scores[name] = scores.get(name, 0) - 2
    elif expertise == "Expert - Deep ML experience":
        for name in ["Neural Networks", "LSTM/GRU", "Deep Q-Networks (DQN)"]:
            scores[name] = scores.get(name, 0) + 1
    
    # 8. Labeled data availability
    labeled_data = answers.get("labeled_data", "")
    if labeled_data == "No labeled data available":
        for name in ["K-Means Clustering", "Hierarchical Clustering", "DBSCAN", "Autoencoders", "Isolation Forest", "Principal Component Analysis (PCA)"]:
            scores[name] = scores.get(name, 0) + 4
            explanations[name] = explanations.get(name, []) + ["Works without labeled data"]
        for name in ["Logistic Regression", "Random Forest", "Gradient Boosting (XGBoost/LightGBM)", "Neural Networks"]:
            scores[name] = scores.get(name, 0) - 3
    elif labeled_data == "Limited labels (can label some data)":
        for name in ["Random Forest", "Gradient Boosting (XGBoost/LightGBM)"]:
            scores[name] = scores.get(name, 0) + 1
    
    # Sort and return top recommendations
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    recommendations = []
    for alg_name, score in sorted_scores[:5]:
        if score > 0:
            for category, algos in ML_ALGORITHMS.items():
                if alg_name in algos:
                    details = algos[alg_name]
                    recommendations.append({
                        "algorithm": alg_name,
                        "category": category,
                        "type": details["type"],
                        "score": score,
                        "use_cases": details["fintech_use_cases"],
                        "pros": details["pros"],
                        "cons": details["cons"],
                        "complexity": details["complexity"],
                        "interpretability": details["interpretability"],
                        "explanations": explanations.get(alg_name, [])
                    })
                    break
    
    return recommendations

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    # Sidebar navigation
    st.sidebar.markdown("""
    <div style="text-align: center; padding: 1rem 0;">
        <h2 style="color: #1B262C; margin: 0;">FinAI Advisor</h2>
        <p style="color: #64748b; font-size: 0.85rem;">ML Decision Support Tool</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.sidebar.markdown("---")
    
    page = st.sidebar.radio(
        "Navigate to:",
        ["Home", "AI Types", "ML Algorithms", "Decision Advisor", 
         "Fintech Use Cases", "Algorithm Comparison", "Resources"],
        format_func=lambda x: f"{'◆' if x == 'Decision Advisor' else '○'} {x}"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    <div style="font-size: 0.8rem; color: #64748b; padding: 0.5rem;">
        <strong>Quick Links</strong><br>
        • Use <strong>Decision Advisor</strong> for personalized recommendations<br>
        • Explore <strong>ML Algorithms</strong> to learn details<br>
        • Check <strong>Resources</strong> for further learning
    </div>
    """, unsafe_allow_html=True)
    
    # ========================================================================
    # HOME PAGE
    # ========================================================================
    if page == "Home":
        st.markdown('<p class="main-header">Fintech AI/ML Advisor</p>', unsafe_allow_html=True)
        st.markdown('<p class="sub-header">Your intelligent guide to selecting the right AI/ML approach for fintech innovation</p>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-icon">◈</div>
                <h3>AI Types</h3>
                <p>Generative vs Analytical AI frameworks</p>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-icon">◇</div>
                <h3>20+ Algorithms</h3>
                <p>Comprehensive ML algorithm library</p>
            </div>
            """, unsafe_allow_html=True)
            
        with col3:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-icon">◎</div>
                <h3>Smart Advisor</h3>
                <p>Interactive decision support questionnaire</p>
            </div>
            """, unsafe_allow_html=True)
            
        with col4:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-icon">◉</div>
                <h3>Use Cases</h3>
                <p>Real-world fintech applications</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.markdown('<p class="section-header">How to Use This Tool</p>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="info-box">
                <span class="step-number">1</span><strong>Understand the Landscape</strong><br>
                Start with the <strong>AI Types</strong> section to understand Generative vs Analytical AI and their fintech applications.
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="info-box">
                <span class="step-number">2</span><strong>Explore Algorithms</strong><br>
                Browse <strong>ML Algorithms</strong> to learn about different approaches, their strengths, weaknesses, and use cases.
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="success-box">
                <span class="step-number">3</span><strong>Get Recommendations</strong><br>
                Use the <strong>Decision Advisor</strong> questionnaire to receive tailored algorithm recommendations for your specific project.
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="info-box">
                <span class="step-number">4</span><strong>Deep Dive</strong><br>
                Explore <strong>Use Cases</strong> and <strong>Resources</strong> for implementation guidance and further learning.
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown('<p class="section-header">Key Considerations for Fintech AI/ML</p>', unsafe_allow_html=True)
        
        considerations = pd.DataFrame({
            "Factor": ["Regulatory Compliance", "Model Interpretability", "Data Privacy", "Model Governance", "Scalability"],
            "Description": [
                "GDPR, CCPA, fair lending laws may restrict certain approaches",
                "Many decisions require explainable models for audit trails",
                "Financial data requires strict controls and anonymization",
                "Models need validation, monitoring, and retraining protocols",
                "Solutions must handle varying loads and growth"
            ],
            "Impact on Approach": [
                "May favor interpretable models over black-box",
                "Consider SHAP/LIME for complex models",
                "Explore federated learning, differential privacy",
                "Build MLOps pipelines from the start",
                "Design for cloud-native deployment"
            ]
        })
        
        st.dataframe(considerations, use_container_width=True, hide_index=True)
    
    # ========================================================================
    # AI TYPES PAGE
    # ========================================================================
    elif page == "AI Types":
        st.markdown('<p class="main-header">Types of Artificial Intelligence</p>', unsafe_allow_html=True)
        st.markdown('<p class="sub-header">Understanding Generative vs Analytical AI in Financial Services</p>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Comparison table
        st.markdown('<p class="section-header">Side-by-Side Comparison</p>', unsafe_allow_html=True)
        
        comparison_df = pd.DataFrame({
            "Aspect": ["Primary Function", "Output Type", "Key Technologies", "Data Needs", 
                      "Interpretability", "Regulatory Fit", "Common Fintech Uses"],
            "Generative AI": [
                "Creates new content",
                "Text, images, code, synthetic data",
                "LLMs, GANs, VAEs, Diffusion Models",
                "Large unstructured datasets",
                "Low (black box)",
                "Challenging - outputs need verification",
                "Chatbots, reports, synthetic data"
            ],
            "Analytical AI": [
                "Analyzes and predicts",
                "Predictions, classifications, scores",
                "ML algorithms, statistical models",
                "Structured, labeled data",
                "Medium to High",
                "Well-established frameworks",
                "Credit scoring, fraud detection"
            ]
        })
        
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        
        # Tabs for detailed info
        tab1, tab2 = st.tabs(["Generative AI", "Analytical AI"])
        
        with tab1:
            gen_ai = AI_TYPES["Generative AI"]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Key Characteristics")
                for char in gen_ai["key_characteristics"]:
                    st.markdown(f"• {char}")
                
                st.markdown("#### Core Technologies")
                for tech in gen_ai["technologies"]:
                    st.markdown(f"• {tech}")
            
            with col2:
                st.markdown("#### Fintech Applications")
                for app in gen_ai["fintech_applications"]:
                    st.markdown(f"• {app}")
            
            st.markdown("---")
            
            col3, col4 = st.columns(2)
            
            with col3:
                st.markdown("#### Advantages")
                for pro in gen_ai["pros"]:
                    st.markdown(f"""
                    <div class="success-box" style="padding: 0.75rem; margin: 0.25rem 0;">
                        ✓ {pro}
                    </div>
                    """, unsafe_allow_html=True)
            
            with col4:
                st.markdown("#### Challenges")
                for con in gen_ai["cons"]:
                    st.markdown(f"""
                    <div class="warning-box" style="padding: 0.75rem; margin: 0.25rem 0;">
                        ⚠ {con}
                    </div>
                    """, unsafe_allow_html=True)
        
        with tab2:
            ana_ai = AI_TYPES["Analytical AI"]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Key Characteristics")
                for char in ana_ai["key_characteristics"]:
                    st.markdown(f"• {char}")
                
                st.markdown("#### Core Technologies")
                for tech in ana_ai["technologies"]:
                    st.markdown(f"• {tech}")
            
            with col2:
                st.markdown("#### Fintech Applications")
                for app in ana_ai["fintech_applications"]:
                    st.markdown(f"• {app}")
            
            st.markdown("---")
            
            col3, col4 = st.columns(2)
            
            with col3:
                st.markdown("#### Advantages")
                for pro in ana_ai["pros"]:
                    st.markdown(f"""
                    <div class="success-box" style="padding: 0.75rem; margin: 0.25rem 0;">
                        ✓ {pro}
                    </div>
                    """, unsafe_allow_html=True)
            
            with col4:
                st.markdown("#### Challenges")
                for con in ana_ai["cons"]:
                    st.markdown(f"""
                    <div class="warning-box" style="padding: 0.75rem; margin: 0.25rem 0;">
                        ⚠ {con}
                    </div>
                    """, unsafe_allow_html=True)
    
    # ========================================================================
    # ML ALGORITHMS PAGE
    # ========================================================================
    elif page == "ML Algorithms":
        st.markdown('<p class="main-header">Machine Learning Algorithms</p>', unsafe_allow_html=True)
        st.markdown('<p class="sub-header">Comprehensive guide to ML algorithms for fintech applications</p>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        category = st.selectbox(
            "Select Algorithm Category:",
            list(ML_ALGORITHMS.keys()),
            format_func=lambda x: f"● {x}"
        )
        
        st.markdown(f'<p class="section-header">{category}</p>', unsafe_allow_html=True)
        
        algorithms = ML_ALGORITHMS[category]
        
        for algo_name, algo_details in algorithms.items():
            with st.expander(f"▸ {algo_name} — {algo_details['type']}", expanded=False):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown(f"**Description:** {algo_details['description']}")
                    
                    st.markdown("**Fintech Use Cases:**")
                    for use_case in algo_details['fintech_use_cases']:
                        st.markdown(f"• {use_case}")
                    
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.markdown("**Advantages:**")
                        for pro in algo_details['pros']:
                            st.markdown(f"✓ {pro}")
                    with col_b:
                        st.markdown("**Limitations:**")
                        for con in algo_details['cons']:
                            st.markdown(f"⚠ {con}")
                
                with col2:
                    st.markdown("**Quick Reference:**")
                    st.markdown(f"""
                    <div class="info-box">
                        <strong>Complexity:</strong> {algo_details['complexity']}<br>
                        <strong>Interpretability:</strong> {algo_details['interpretability']}<br>
                        <strong>Data:</strong> {algo_details['data_requirements']}
                    </div>
                    """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown('<p class="section-header">Algorithm Selection Quick Guide</p>', unsafe_allow_html=True)
        
        selection_guide = pd.DataFrame({
            "If you need...": [
                "High interpretability",
                "Non-linear pattern detection",
                "Limited training data",
                "Maximum predictive power",
                "Anomaly detection",
                "Time series forecasting",
                "Customer segmentation"
            ],
            "Recommended": [
                "Logistic Regression, Decision Trees",
                "Random Forest, XGBoost, Neural Networks",
                "Logistic Regression, Decision Trees",
                "Gradient Boosting, Neural Networks",
                "Isolation Forest, Autoencoders",
                "ARIMA, LSTM, Prophet",
                "K-Means, Hierarchical Clustering"
            ],
            "Avoid": [
                "Neural Networks, Deep Ensembles",
                "Simple Linear Models",
                "Deep Learning methods",
                "Simple Linear Models (if data is complex)",
                "Standard classifiers without adaptation",
                "Cross-sectional models",
                "Supervised methods (without labels)"
            ]
        })
        
        st.dataframe(selection_guide, use_container_width=True, hide_index=True)
    
    # ========================================================================
    # DECISION ADVISOR PAGE
    # ========================================================================
    elif page == "Decision Advisor":
        st.markdown('<p class="main-header">AI/ML Decision Advisor</p>', unsafe_allow_html=True)
        st.markdown('<p class="sub-header">Answer the questionnaire below to receive personalized algorithm recommendations</p>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        if 'advisor_submitted' not in st.session_state:
            st.session_state.advisor_submitted = False
        
        with st.form("advisor_form"):
            st.markdown('<p class="section-header">Project Assessment Questionnaire</p>', unsafe_allow_html=True)
            
            # Question 1: Problem Type
            st.markdown("#### 1. What type of problem are you trying to solve?")
            problem_type = st.selectbox(
                "Select the primary problem type:",
                [
                    "Binary Classification",
                    "Multi-class Classification", 
                    "Regression / Continuous Prediction",
                    "Clustering / Segmentation",
                    "Anomaly / Fraud Detection",
                    "Time Series Forecasting",
                    "Sequential Decision Making",
                    "Natural Language Processing",
                    "Content / Report Generation"
                ],
                label_visibility="collapsed",
                help="Classification: Yes/No or category prediction | Regression: Numeric prediction | Clustering: Group discovery"
            )
            
            st.markdown("---")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Question 2: Data Size
                st.markdown("#### 2. How much data do you have?")
                data_size = st.selectbox(
                    "Data size:",
                    [
                        "Very Small (<500 samples)",
                        "Small (500-5,000 samples)",
                        "Medium (5,000-50,000 samples)",
                        "Large (50,000-500,000 samples)",
                        "Very Large (>500,000 samples)"
                    ],
                    label_visibility="collapsed"
                )
                
                # Question 3: Labeled Data
                st.markdown("#### 3. Do you have labeled/target data?")
                labeled_data = st.selectbox(
                    "Labeled data availability:",
                    [
                        "Yes - Fully labeled dataset",
                        "Partial - Some data is labeled",
                        "Limited labels (can label some data)",
                        "No labeled data available"
                    ],
                    label_visibility="collapsed"
                )
                
                # Question 4: Data Quality
                st.markdown("#### 4. What is your data quality?")
                data_quality = st.selectbox(
                    "Data quality:",
                    [
                        "Excellent - Clean, complete data",
                        "Good - Minor issues, mostly complete",
                        "Fair - Some missing values and inconsistencies",
                        "Poor - Lots of missing values and noise"
                    ],
                    label_visibility="collapsed"
                )
                
                # Question 5: Feature Engineering
                st.markdown("#### 5. Feature engineering capacity?")
                feature_engineering = st.selectbox(
                    "Feature engineering:",
                    [
                        "Extensive - Will invest in feature engineering",
                        "Moderate - Some feature engineering possible",
                        "Minimal - Prefer automatic feature learning"
                    ],
                    label_visibility="collapsed"
                )
            
            with col2:
                # Question 6: Interpretability
                st.markdown("#### 6. How important is model interpretability?")
                interpretability = st.selectbox(
                    "Interpretability requirement:",
                    [
                        "Critical - Must explain every decision (regulatory requirement)",
                        "Important - Need to understand key drivers",
                        "Moderate - Some explanation helpful",
                        "Low - Accuracy is the priority"
                    ],
                    label_visibility="collapsed"
                )
                
                # Question 7: Latency
                st.markdown("#### 7. What are your latency requirements?")
                latency = st.selectbox(
                    "Latency needs:",
                    [
                        "Real-time (<10ms)",
                        "Near real-time (<1 second)",
                        "Fast (<10 seconds)",
                        "Batch processing is fine"
                    ],
                    label_visibility="collapsed"
                )
                
                # Question 8: Team Expertise
                st.markdown("#### 8. What is your team's ML expertise?")
                expertise = st.selectbox(
                    "Team expertise:",
                    [
                        "Beginner - New to ML",
                        "Intermediate - Some ML experience",
                        "Advanced - Strong ML background",
                        "Expert - Deep ML experience"
                    ],
                    label_visibility="collapsed"
                )
                
                # Question 9: Domain
                st.markdown("#### 9. What fintech domain is this for?")
                domain = st.selectbox(
                    "Fintech domain:",
                    list(FINTECH_DOMAINS.keys()),
                    label_visibility="collapsed"
                )
            
            st.markdown("---")
            
            # Question 10: Project Description
            st.markdown("#### 10. Describe your project briefly (optional)")
            project_description = st.text_area(
                "Project description:",
                placeholder="E.g., 'We want to predict which loan applicants will default based on application data and credit history to automate initial credit decisions...'",
                height=100,
                label_visibility="collapsed"
            )
            
            submitted = st.form_submit_button("▶ Generate Recommendations", use_container_width=True)
        
        if submitted:
            st.session_state.advisor_submitted = True
            st.session_state.answers = {
                "problem_type": problem_type,
                "data_size": data_size,
                "labeled_data": labeled_data,
                "data_quality": data_quality,
                "feature_engineering": feature_engineering,
                "interpretability": interpretability,
                "latency": latency,
                "expertise": expertise,
                "domain": domain,
                "project_description": project_description
            }
        
        if st.session_state.advisor_submitted:
            answers = st.session_state.answers
            
            st.markdown("---")
            st.markdown('<p class="section-header">Your Personalized Recommendations</p>', unsafe_allow_html=True)
            
            recommendations = calculate_recommendation_scores(answers)
            
            if recommendations:
                # Generative AI recommendation
                gen_ai_recs = [r for r in recommendations if r.get("type") == "Generative AI"]
                ml_recs = [r for r in recommendations if r.get("algorithm")]
                
                if gen_ai_recs:
                    st.markdown("#### Generative AI Recommendation")
                    for rec in gen_ai_recs:
                        st.markdown(f"""
                        <div class="success-box">
                            <h4 style="margin-top: 0;">▸ {rec['recommendation']}</h4>
                            <p>{rec['reason']}</p>
                            <p><strong>Consider:</strong> GPT-4, Claude, Llama, or BERT-based models for NLP tasks; 
                            GANs or Diffusion Models for synthetic data generation.</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                if ml_recs:
                    st.markdown("#### Top Algorithm Recommendations")
                    
                    for i, rec in enumerate(ml_recs[:3], 1):
                        rank_label = ["Top Choice", "Strong Alternative", "Also Consider"][i-1]
                        
                        with st.expander(f"{'▸' if i == 1 else '▹'} #{i}: {rec['algorithm']} — {rank_label} (Score: {rec['score']})", expanded=(i==1)):
                            col1, col2 = st.columns([2, 1])
                            
                            with col1:
                                st.markdown(f"**Category:** {rec['category']}")
                                st.markdown(f"**Type:** {rec['type']}")
                                
                                if rec.get('explanations'):
                                    st.markdown("**Why this recommendation:**")
                                    for exp in rec['explanations'][:3]:
                                        st.markdown(f"• {exp}")
                                
                                st.markdown("**Fintech Use Cases:**")
                                for use_case in rec['use_cases'][:3]:
                                    st.markdown(f"• {use_case}")
                                
                                col_a, col_b = st.columns(2)
                                with col_a:
                                    st.markdown("**Pros:**")
                                    for pro in rec['pros'][:3]:
                                        st.markdown(f"✓ {pro}")
                                with col_b:
                                    st.markdown("**Cons:**")
                                    for con in rec['cons'][:2]:
                                        st.markdown(f"⚠ {con}")
                            
                            with col2:
                                st.markdown(f"""
                                <div class="info-box">
                                    <strong>Complexity:</strong><br>{rec['complexity']}<br><br>
                                    <strong>Interpretability:</strong><br>{rec['interpretability']}
                                </div>
                                """, unsafe_allow_html=True)
                
                # Domain insights
                st.markdown("---")
                st.markdown(f'<p class="section-header">{answers["domain"]} Domain Insights</p>', unsafe_allow_html=True)
                
                domain_info = FINTECH_DOMAINS[answers['domain']]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Common Problems:**")
                    for problem in domain_info['problems']:
                        st.markdown(f"• {problem}")
                
                with col2:
                    st.markdown("**Recommended Approaches:**")
                    for approach in domain_info['recommended_approaches']:
                        st.markdown(f"• {approach}")
                
                # Next steps
                st.markdown("---")
                st.markdown('<p class="section-header">Recommended Next Steps</p>', unsafe_allow_html=True)
                
                steps = [
                    ("Data Assessment", "Evaluate data quality, completeness, and potential biases"),
                    ("Baseline Model", "Start with a simple interpretable model to establish baseline performance"),
                    ("Iterative Improvement", "Gradually increase complexity only if needed"),
                    ("Validation", "Implement proper cross-validation and out-of-time testing"),
                    ("Compliance Review", "Ensure approach meets regulatory requirements for your domain"),
                    ("MLOps Planning", "Design monitoring, retraining, and deployment pipelines")
                ]
                
                for i, (step_title, step_desc) in enumerate(steps, 1):
                    st.markdown(f"""
                    <div class="info-box" style="padding: 0.75rem;">
                        <span class="step-number">{i}</span><strong>{step_title}</strong><br>
                        <span style="color: #64748b;">{step_desc}</span>
                    </div>
                    """, unsafe_allow_html=True)
    
    # ========================================================================
    # FINTECH USE CASES PAGE
    # ========================================================================
    elif page == "Fintech Use Cases":
        st.markdown('<p class="main-header">Fintech Use Cases</p>', unsafe_allow_html=True)
        st.markdown('<p class="sub-header">Real-world AI/ML applications across financial services domains</p>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        domain = st.selectbox("Select Domain:", list(FINTECH_DOMAINS.keys()))
        
        st.markdown(f'<p class="section-header">{domain}</p>', unsafe_allow_html=True)
        
        domain_info = FINTECH_DOMAINS[domain]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Key Problems")
            for problem in domain_info['problems']:
                st.markdown(f"""
                <div class="info-box" style="padding: 0.75rem;">
                    ◇ {problem}
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("#### Recommended Approaches")
            for approach in domain_info['recommended_approaches']:
                st.markdown(f"""
                <div class="success-box" style="padding: 0.75rem;">
                    ◆ {approach}
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Detailed use cases
        use_cases = {
            "Payments & Transactions": [
                {
                    "title": "Real-time Fraud Detection",
                    "description": "Detect fraudulent transactions in real-time using behavioral patterns, device fingerprints, and transaction history.",
                    "algorithms": ["XGBoost", "Neural Networks", "Isolation Forest"],
                    "challenges": ["Low latency requirements", "Extreme class imbalance", "Evolving fraud patterns"],
                    "metrics": ["Fraud detection rate", "False positive rate", "Processing latency"]
                },
                {
                    "title": "Transaction Categorization",
                    "description": "Automatically categorize transactions for PFM apps and accounting systems using merchant data and descriptions.",
                    "algorithms": ["BERT/NLP models", "Random Forest", "Gradient Boosting"],
                    "challenges": ["Merchant name variations", "Multi-language support", "New categories"],
                    "metrics": ["Classification accuracy", "Coverage rate", "User correction rate"]
                }
            ],
            "Lending & Credit": [
                {
                    "title": "Credit Scoring",
                    "description": "Assess creditworthiness using traditional and alternative data to predict default probability.",
                    "algorithms": ["Logistic Regression", "XGBoost", "Neural Networks + SHAP"],
                    "challenges": ["Regulatory compliance", "Fair lending", "Interpretability"],
                    "metrics": ["AUC-ROC", "Gini coefficient", "KS statistic"]
                },
                {
                    "title": "Early Warning Systems",
                    "description": "Identify borrowers at risk of default before they miss payments using behavioral signals.",
                    "algorithms": ["Survival Analysis", "LSTM", "Gradient Boosting"],
                    "challenges": ["Lead time requirements", "Signal noise", "Action timing"],
                    "metrics": ["Early detection rate", "False alarm rate", "Time to default"]
                }
            ],
            "Investment & Trading": [
                {
                    "title": "Algorithmic Trading",
                    "description": "Develop automated trading strategies using pattern recognition and market signal analysis.",
                    "algorithms": ["LSTM", "Reinforcement Learning", "Ensemble methods"],
                    "challenges": ["Market regime changes", "Transaction costs", "Overfitting"],
                    "metrics": ["Sharpe ratio", "Maximum drawdown", "Alpha generation"]
                },
                {
                    "title": "Portfolio Optimization",
                    "description": "Construct and rebalance portfolios using ML-enhanced optimization for risk-adjusted returns.",
                    "algorithms": ["Mean-Variance + ML", "Deep Learning", "RL for allocation"],
                    "challenges": ["Estimation error", "Transaction costs", "Correlation instability"],
                    "metrics": ["Risk-adjusted returns", "Tracking error", "Turnover"]
                }
            ],
            "Insurance": [
                {
                    "title": "Claims Prediction",
                    "description": "Predict claim likelihood and severity for pricing and reserving using policyholder data.",
                    "algorithms": ["GLMs", "XGBoost", "Survival Analysis"],
                    "challenges": ["Long-tail distributions", "Regulatory requirements", "Data sparsity"],
                    "metrics": ["Loss ratio", "Reserve accuracy", "Prediction interval"]
                },
                {
                    "title": "Underwriting Automation",
                    "description": "Automate risk assessment and policy pricing using application data and external sources.",
                    "algorithms": ["Decision Trees", "Gradient Boosting", "Rules + ML hybrid"],
                    "challenges": ["Explainability needs", "Edge cases", "Regulatory approval"],
                    "metrics": ["STP rate", "Loss ratio by segment", "Processing time"]
                }
            ],
            "RegTech & Compliance": [
                {
                    "title": "Anti-Money Laundering",
                    "description": "Detect suspicious patterns and potential money laundering using transaction network analysis.",
                    "algorithms": ["Graph Neural Networks", "Anomaly Detection", "Network Analysis"],
                    "challenges": ["High false positive rates", "Evolving typologies", "Regulatory scrutiny"],
                    "metrics": ["SAR quality", "Alert-to-case ratio", "Detection rate"]
                },
                {
                    "title": "KYC Automation",
                    "description": "Automate identity verification and document processing using OCR and NLP.",
                    "algorithms": ["OCR + NLP", "Face Recognition", "Document Classification"],
                    "challenges": ["Document variations", "Fraud attempts", "Global compliance"],
                    "metrics": ["Automation rate", "Accuracy", "Processing time"]
                }
            ],
            "Customer Experience": [
                {
                    "title": "Intelligent Virtual Assistants",
                    "description": "Deploy conversational AI for customer service, transactions, and financial guidance.",
                    "algorithms": ["LLMs (GPT, Claude)", "Intent Classification", "RAG"],
                    "challenges": ["Accuracy for financial queries", "Escalation handling", "Compliance"],
                    "metrics": ["Containment rate", "CSAT", "Resolution time"]
                },
                {
                    "title": "Churn Prediction",
                    "description": "Identify customers likely to leave and trigger retention actions using behavioral signals.",
                    "algorithms": ["XGBoost", "Survival Analysis", "Neural Networks"],
                    "challenges": ["Definition of churn", "Action timing", "Causal inference"],
                    "metrics": ["Churn prediction accuracy", "Retention rate lift", "ROI"]
                }
            ]
        }
        
        if domain in use_cases:
            st.markdown("#### Detailed Use Cases")
            
            for use_case in use_cases[domain]:
                with st.expander(f"▸ {use_case['title']}", expanded=True):
                    st.markdown(f"**{use_case['description']}**")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown("**Algorithms:**")
                        for algo in use_case['algorithms']:
                            st.markdown(f"• {algo}")
                    
                    with col2:
                        st.markdown("**Challenges:**")
                        for challenge in use_case['challenges']:
                            st.markdown(f"• {challenge}")
                    
                    with col3:
                        st.markdown("**Success Metrics:**")
                        for metric in use_case['metrics']:
                            st.markdown(f"• {metric}")
    
    # ========================================================================
    # ALGORITHM COMPARISON PAGE
    # ========================================================================
    elif page == "Algorithm Comparison":
        st.markdown('<p class="main-header">Algorithm Comparison</p>', unsafe_allow_html=True)
        st.markdown('<p class="sub-header">Visual comparison of algorithms across key dimensions</p>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Main trade-off chart
        st.markdown('<p class="section-header">Complexity vs Interpretability Trade-off</p>', unsafe_allow_html=True)
        
        fig_tradeoff = create_tradeoff_chart(ML_ALGORITHMS)
        st.plotly_chart(fig_tradeoff, use_container_width=True)
        
        st.markdown("""
        <div class="info-box">
            <strong>Reading the Chart:</strong> Algorithms in the top-left quadrant (simple & explainable) are ideal for 
            regulated applications. Bottom-right algorithms (complex & opaque) offer higher performance but require 
            explainability tools like SHAP for compliance.
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Individual comparisons
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<p class="section-header">Complexity Comparison</p>', unsafe_allow_html=True)
            fig_complexity = create_comparison_chart(ML_ALGORITHMS, "complexity")
            st.plotly_chart(fig_complexity, use_container_width=True)
        
        with col2:
            st.markdown('<p class="section-header">Interpretability Comparison</p>', unsafe_allow_html=True)
            fig_interpret = create_comparison_chart(ML_ALGORITHMS, "interpretability")
            st.plotly_chart(fig_interpret, use_container_width=True)
    
    # ========================================================================
    # RESOURCES PAGE
    # ========================================================================
    elif page == "Resources":
        st.markdown('<p class="main-header">Learning Resources</p>', unsafe_allow_html=True)
        st.markdown('<p class="sub-header">Curated resources for deepening your AI/ML knowledge in fintech</p>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Books
        st.markdown('<p class="section-header">Books</p>', unsafe_allow_html=True)
        
        for book in RESOURCES["Books"]:
            st.markdown(f"""
            <a href="{book['url']}" target="_blank" class="resource-link">
                <strong>{book['title']}</strong><br>
                <span style="color: #64748b;">by {book['author']}</span>
            </a>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Online Courses
        st.markdown('<p class="section-header">Online Courses</p>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        courses = RESOURCES["Online Courses"]
        
        for i, course in enumerate(courses):
            with col1 if i % 2 == 0 else col2:
                st.markdown(f"""
                <a href="{course['url']}" target="_blank" class="resource-link">
                    <strong>{course['title']}</strong><br>
                    <span style="color: #64748b;">{course['provider']}</span>
                </a>
                """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Documentation
        st.markdown('<p class="section-header">Documentation & Tutorials</p>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        docs = RESOURCES["Documentation & Tutorials"]
        
        for i, doc in enumerate(docs):
            with col1 if i % 2 == 0 else col2:
                st.markdown(f"""
                <a href="{doc['url']}" target="_blank" class="resource-link">
                    <strong>{doc['title']}</strong><br>
                    <span style="color: #64748b;">{doc['description']}</span>
                </a>
                """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Research
        st.markdown('<p class="section-header">Research & Papers</p>', unsafe_allow_html=True)
        
        for paper in RESOURCES["Research & Papers"]:
            st.markdown(f"""
            <a href="{paper['url']}" target="_blank" class="resource-link">
                <strong>{paper['title']}</strong><br>
                <span style="color: #64748b;">{paper['description']}</span>
            </a>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Tools
        st.markdown('<p class="section-header">Tools & Libraries</p>', unsafe_allow_html=True)
        
        cols = st.columns(4)
        tools = RESOURCES["Tools & Libraries"]
        
        for i, tool in enumerate(tools):
            with cols[i % 4]:
                st.markdown(f"""
                <a href="{tool['url']}" target="_blank" class="resource-link" style="text-align: center;">
                    <strong>{tool['title']}</strong><br>
                    <span style="color: #64748b; font-size: 0.8rem;">{tool['description']}</span>
                </a>
                """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Checklist
        st.markdown('<p class="section-header">ML Project Checklist</p>', unsafe_allow_html=True)
        
        checklist_items = [
            ("Problem Definition", "Define business problem, success metrics, and constraints"),
            ("Data Assessment", "Evaluate quality, completeness, biases, and privacy requirements"),
            ("EDA & Feature Engineering", "Understand distributions, create meaningful features"),
            ("Model Selection", "Choose algorithms based on requirements and constraints"),
            ("Validation Strategy", "Design proper cross-validation and holdout testing"),
            ("Hyperparameter Tuning", "Systematic optimization with appropriate search methods"),
            ("Interpretability", "Ensure decisions can be explained (SHAP, LIME)"),
            ("Bias & Fairness", "Check for discriminatory patterns across protected groups"),
            ("Documentation", "Document methodology, assumptions, limitations"),
            ("Deployment", "Plan production deployment, API design, scaling"),
            ("Monitoring", "Implement performance tracking, drift detection, alerts"),
            ("Governance", "Establish retraining triggers, approval workflows, audit trails")
        ]
        
        col1, col2 = st.columns(2)
        
        for i, (item, desc) in enumerate(checklist_items):
            with col1 if i % 2 == 0 else col2:
                st.markdown(f"""
                <div class="info-box" style="padding: 0.75rem;">
                    <strong>☐ {item}</strong><br>
                    <span style="color: #64748b; font-size: 0.9rem;">{desc}</span>
                </div>
                """, unsafe_allow_html=True)

# ============================================================================
# RUN APPLICATION
# ============================================================================

if __name__ == "__main__":
    main()