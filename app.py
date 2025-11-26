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
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E3A5F;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .info-box {
        background-color: #f0f7ff;
        border-left: 5px solid #1E88E5;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 8px 8px 0;
    }
    .success-box {
        background-color: #e8f5e9;
        border-left: 5px solid #4CAF50;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 8px 8px 0;
    }
    .warning-box {
        background-color: #fff3e0;
        border-left: 5px solid #FF9800;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 8px 8px 0;
    }
    .algorithm-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .metric-card {
        background: white;
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
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
            "complexity": "Low to Medium",
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
        "Gradient Boosting (XGBoost, LightGBM)": {
            "type": "Classification/Regression",
            "description": "Sequential ensemble method that builds trees to correct previous errors.",
            "fintech_use_cases": ["Credit risk modeling", "Trading signal prediction", "Customer lifetime value"],
            "pros": ["State-of-the-art performance", "Handles missing values", "Feature importance"],
            "cons": ["Requires careful tuning", "Can overfit with too many trees", "Less interpretable"],
            "data_requirements": "Tabular data, can handle missing values",
            "complexity": "Medium to High",
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
        "Neural Networks (Deep Learning)": {
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

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

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
            
            data.append({
                "Algorithm": name,
                "Category": category,
                "Value": value,
                "Label": details.get(metric, "Medium")
            })
    
    df = pd.DataFrame(data)
    fig = px.bar(df, x="Algorithm", y="Value", color="Category",
                 title=f"Algorithm Comparison by {metric.title()}",
                 hover_data=["Label"])
    fig.update_layout(xaxis_tickangle=-45, height=500)
    return fig

def get_recommendation(answers):
    """Generate algorithm recommendations based on user answers."""
    recommendations = []
    scores = {}
    
    # Initialize scores for all algorithms
    for category, algos in ML_ALGORITHMS.items():
        for name in algos.keys():
            scores[name] = 0
    
    # Problem type scoring
    if answers.get("problem_type") == "Classification":
        for name in ["Logistic Regression", "Random Forest", "Gradient Boosting (XGBoost, LightGBM)", 
                     "Neural Networks (Deep Learning)", "SVM"]:
            scores[name] = scores.get(name, 0) + 3
    elif answers.get("problem_type") == "Regression":
        for name in ["Linear Regression", "Random Forest", "Gradient Boosting (XGBoost, LightGBM)",
                     "Neural Networks (Deep Learning)"]:
            scores[name] = scores.get(name, 0) + 3
    elif answers.get("problem_type") == "Clustering":
        for name in ["K-Means Clustering", "Hierarchical Clustering", "DBSCAN"]:
            scores[name] = scores.get(name, 0) + 3
    elif answers.get("problem_type") == "Anomaly Detection":
        for name in ["Isolation Forest", "Autoencoders", "DBSCAN"]:
            scores[name] = scores.get(name, 0) + 3
    elif answers.get("problem_type") == "Time Series Forecasting":
        for name in ["ARIMA/SARIMA", "LSTM/GRU", "Prophet"]:
            scores[name] = scores.get(name, 0) + 3
    elif answers.get("problem_type") == "Content Generation":
        recommendations.append({
            "type": "Generative AI",
            "recommendation": "Large Language Models (GPT-style)",
            "reason": "Content generation requires generative AI capabilities"
        })
    elif answers.get("problem_type") == "Sequential Decision Making":
        for name in ["Q-Learning", "Deep Q-Networks (DQN)", "Policy Gradient Methods"]:
            scores[name] = scores.get(name, 0) + 3
    
    # Data size scoring
    if answers.get("data_size") == "Small (<1,000 samples)":
        for name in ["Linear Regression", "Logistic Regression", "Decision Trees"]:
            scores[name] = scores.get(name, 0) + 2
        for name in ["Neural Networks (Deep Learning)", "LSTM/GRU", "Deep Q-Networks (DQN)"]:
            scores[name] = scores.get(name, 0) - 2
    elif answers.get("data_size") == "Large (>100,000 samples)":
        for name in ["Neural Networks (Deep Learning)", "Gradient Boosting (XGBoost, LightGBM)"]:
            scores[name] = scores.get(name, 0) + 2
    
    # Interpretability scoring
    if answers.get("interpretability") == "Critical (Regulatory requirement)":
        for name in ["Linear Regression", "Logistic Regression", "Decision Trees"]:
            scores[name] = scores.get(name, 0) + 3
        for name in ["Neural Networks (Deep Learning)", "Autoencoders", "Deep Q-Networks (DQN)"]:
            scores[name] = scores.get(name, 0) - 3
    elif answers.get("interpretability") == "Important":
        for name in ["Random Forest", "Gradient Boosting (XGBoost, LightGBM)"]:
            scores[name] = scores.get(name, 0) + 1
    
    # Real-time scoring
    if answers.get("real_time") == "Yes":
        for name in ["Linear Regression", "Logistic Regression", "Decision Trees"]:
            scores[name] = scores.get(name, 0) + 1
        for name in ["Neural Networks (Deep Learning)"]:
            scores[name] = scores.get(name, 0) - 1
    
    # Sort and get top recommendations
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_algorithms = [alg for alg, score in sorted_scores[:5] if score > 0]
    
    for alg_name in top_algorithms:
        # Find algorithm details
        for category, algos in ML_ALGORITHMS.items():
            if alg_name in algos:
                details = algos[alg_name]
                recommendations.append({
                    "algorithm": alg_name,
                    "category": category,
                    "type": details["type"],
                    "score": scores[alg_name],
                    "use_cases": details["fintech_use_cases"],
                    "pros": details["pros"],
                    "cons": details["cons"],
                    "complexity": details["complexity"],
                    "interpretability": details["interpretability"]
                })
                break
    
    return recommendations

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    # Sidebar navigation
    st.sidebar.title("🤖 Navigation")
    page = st.sidebar.radio(
        "Select a section:",
        ["🏠 Home", "🧠 AI Types", "📊 ML Algorithms", "🎯 Decision Advisor", 
         "💼 Fintech Use Cases", "📈 Algorithm Comparison", "📚 Resources"]
    )
    
    # ========================================================================
    # HOME PAGE
    # ========================================================================
    if page == "🏠 Home":
        st.markdown('<p class="main-header">🤖 Fintech AI/ML Advisor</p>', unsafe_allow_html=True)
        st.markdown('<p class="sub-header">Your guide to selecting the right AI/ML approach for fintech innovation</p>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
                <h3>🧠 AI Types</h3>
                <p>Understand Generative vs Analytical AI</p>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown("""
            <div class="metric-card">
                <h3>📊 20+ Algorithms</h3>
                <p>Comprehensive ML algorithm library</p>
            </div>
            """, unsafe_allow_html=True)
            
        with col3:
            st.markdown("""
            <div class="metric-card">
                <h3>🎯 Smart Advisor</h3>
                <p>Interactive decision support tool</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.markdown("### 🎓 How to Use This Tool")
        
        st.markdown("""
        <div class="info-box">
        <strong>Step 1: Understand the Landscape</strong><br>
        Start with the <strong>AI Types</strong> section to understand the fundamental difference between 
        Generative and Analytical AI, and when to use each in fintech.
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
        <strong>Step 2: Explore Algorithms</strong><br>
        Browse the <strong>ML Algorithms</strong> section to learn about different algorithms, 
        their strengths, weaknesses, and fintech applications.
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
        <strong>Step 3: Get Personalized Recommendations</strong><br>
        Use the <strong>Decision Advisor</strong> to answer questions about your specific problem 
        and receive tailored algorithm recommendations.
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
        <strong>Step 4: Explore Use Cases</strong><br>
        Check the <strong>Fintech Use Cases</strong> section to see real-world applications 
        and best practices in different fintech domains.
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### 🔑 Key Considerations for Fintech AI/ML Projects")
        
        considerations = pd.DataFrame({
            "Factor": ["Regulatory Compliance", "Interpretability", "Data Privacy", "Model Validation", "Scalability"],
            "Description": [
                "Financial regulations (GDPR, CCPA, fair lending) may restrict certain AI approaches",
                "Many financial decisions require explainable models for audit and compliance",
                "Customer financial data requires strict privacy controls and anonymization",
                "Models must be rigorously validated and monitored for drift",
                "Solutions must handle varying loads and grow with business needs"
            ],
            "Impact on AI Choice": [
                "May favor simpler, interpretable models over black-box approaches",
                "Consider SHAP, LIME for complex models; prefer inherently interpretable models",
                "Consider federated learning, differential privacy techniques",
                "Build comprehensive monitoring and retraining pipelines",
                "Consider cloud-native, microservices architectures"
            ]
        })
        
        st.table(considerations)
    
    # ========================================================================
    # AI TYPES PAGE
    # ========================================================================
    elif page == "🧠 AI Types":
        st.markdown('<p class="main-header">🧠 Types of Artificial Intelligence</p>', unsafe_allow_html=True)
        st.markdown('<p class="sub-header">Understanding Generative vs Analytical AI in Fintech</p>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Overview
        st.markdown("### 📌 Overview")
        st.markdown("""
        In fintech, AI applications generally fall into two broad categories: **Generative AI** and **Analytical AI**. 
        Understanding the distinction is crucial for selecting the right approach for your specific problem.
        """)
        
        # Comparison table
        comparison_df = pd.DataFrame({
            "Aspect": ["Primary Function", "Output", "Key Technologies", "Data Requirements", 
                      "Interpretability", "Regulatory Fit", "Typical Use Cases"],
            "Generative AI": [
                "Creates new content",
                "Text, images, code, synthetic data",
                "LLMs, GANs, VAEs, Diffusion Models",
                "Large unstructured datasets",
                "Low (black box)",
                "Challenging - outputs need verification",
                "Chatbots, report generation, synthetic data"
            ],
            "Analytical AI": [
                "Analyzes and predicts",
                "Predictions, classifications, insights",
                "ML algorithms, statistical models",
                "Structured, labeled data",
                "Medium to High",
                "Well-established frameworks",
                "Credit scoring, fraud detection, forecasting"
            ]
        })
        
        st.table(comparison_df)
        
        st.markdown("---")
        
        # Detailed sections for each AI type
        tab1, tab2 = st.tabs(["🎨 Generative AI", "📊 Analytical AI"])
        
        with tab1:
            gen_ai = AI_TYPES["Generative AI"]
            st.markdown(f"### {gen_ai['description']}")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 🔑 Key Characteristics")
                for char in gen_ai["key_characteristics"]:
                    st.markdown(f"- {char}")
                
                st.markdown("#### 🛠️ Technologies")
                for tech in gen_ai["technologies"]:
                    st.markdown(f"- {tech}")
            
            with col2:
                st.markdown("#### 💼 Fintech Applications")
                for app in gen_ai["fintech_applications"]:
                    st.markdown(f"- {app}")
            
            st.markdown("---")
            
            col3, col4 = st.columns(2)
            
            with col3:
                st.markdown("#### ✅ Advantages")
                for pro in gen_ai["pros"]:
                    st.success(pro)
            
            with col4:
                st.markdown("#### ⚠️ Challenges")
                for con in gen_ai["cons"]:
                    st.warning(con)
            
            st.markdown("---")
            st.markdown("#### 🎓 When to Use Generative AI in Fintech")
            st.markdown("""
            <div class="success-box">
            <strong>Best suited for:</strong>
            <ul>
                <li>Customer service automation (chatbots, virtual assistants)</li>
                <li>Document summarization and analysis</li>
                <li>Generating synthetic data for model training</li>
                <li>Personalized content generation (reports, insights)</li>
                <li>Code generation for internal tools</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="warning-box">
            <strong>Exercise caution for:</strong>
            <ul>
                <li>Direct financial advice generation (regulatory concerns)</li>
                <li>Decision-making without human oversight</li>
                <li>Applications requiring auditability</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with tab2:
            ana_ai = AI_TYPES["Analytical AI"]
            st.markdown(f"### {ana_ai['description']}")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 🔑 Key Characteristics")
                for char in ana_ai["key_characteristics"]:
                    st.markdown(f"- {char}")
                
                st.markdown("#### 🛠️ Technologies")
                for tech in ana_ai["technologies"]:
                    st.markdown(f"- {tech}")
            
            with col2:
                st.markdown("#### 💼 Fintech Applications")
                for app in ana_ai["fintech_applications"]:
                    st.markdown(f"- {app}")
            
            st.markdown("---")
            
            col3, col4 = st.columns(2)
            
            with col3:
                st.markdown("#### ✅ Advantages")
                for pro in ana_ai["pros"]:
                    st.success(pro)
            
            with col4:
                st.markdown("#### ⚠️ Challenges")
                for con in ana_ai["cons"]:
                    st.warning(con)
            
            st.markdown("---")
            st.markdown("#### 🎓 When to Use Analytical AI in Fintech")
            st.markdown("""
            <div class="success-box">
            <strong>Best suited for:</strong>
            <ul>
                <li>Credit scoring and risk assessment</li>
                <li>Fraud detection and prevention</li>
                <li>Algorithmic trading and portfolio optimization</li>
                <li>Customer segmentation and churn prediction</li>
                <li>Regulatory compliance and reporting</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # Decision framework
        st.markdown("---")
        st.markdown("### 🎯 Decision Framework: Generative vs Analytical")
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=[1, 2, 3, 4, 5],
            y=[5, 4, 3, 2, 1],
            mode='markers+text',
            name='Generative AI',
            text=['Chatbots', 'Content Gen', 'Synthetic Data', 'Doc Analysis', 'Code Gen'],
            textposition='top center',
            marker=dict(size=20, color='#667eea')
        ))
        
        fig.add_trace(go.Scatter(
            x=[1, 2, 3, 4, 5],
            y=[1, 2, 3, 4, 5],
            mode='markers+text',
            name='Analytical AI',
            text=['Credit Score', 'Fraud Detect', 'Trading', 'Risk Mgmt', 'Compliance'],
            textposition='bottom center',
            marker=dict(size=20, color='#764ba2')
        ))
        
        fig.update_layout(
            title="Use Case Mapping",
            xaxis_title="Interpretability Requirement →",
            yaxis_title="← Creativity/Generation Need",
            height=400,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # ========================================================================
    # ML ALGORITHMS PAGE
    # ========================================================================
    elif page == "📊 ML Algorithms":
        st.markdown('<p class="main-header">📊 Machine Learning Algorithms</p>', unsafe_allow_html=True)
        st.markdown('<p class="sub-header">Comprehensive guide to ML algorithms for fintech applications</p>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Category selector
        category = st.selectbox(
            "Select Algorithm Category:",
            list(ML_ALGORITHMS.keys())
        )
        
        st.markdown(f"### {category}")
        
        # Display algorithms in the selected category
        algorithms = ML_ALGORITHMS[category]
        
        for algo_name, algo_details in algorithms.items():
            with st.expander(f"📌 {algo_name} ({algo_details['type']})", expanded=False):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown(f"**Description:** {algo_details['description']}")
                    
                    st.markdown("**Fintech Use Cases:**")
                    for use_case in algo_details['fintech_use_cases']:
                        st.markdown(f"- {use_case}")
                    
                    st.markdown("**Pros:**")
                    for pro in algo_details['pros']:
                        st.markdown(f"- ✅ {pro}")
                    
                    st.markdown("**Cons:**")
                    for con in algo_details['cons']:
                        st.markdown(f"- ⚠️ {con}")
                
                with col2:
                    st.markdown("**Quick Facts:**")
                    st.info(f"📊 **Complexity:** {algo_details['complexity']}")
                    st.info(f"🔍 **Interpretability:** {algo_details['interpretability']}")
                    st.info(f"📁 **Data:** {algo_details['data_requirements']}")
        
        st.markdown("---")
        st.markdown("### 📊 Algorithm Selection Guide")
        
        selection_guide = pd.DataFrame({
            "If you need...": [
                "High interpretability",
                "Handle non-linear patterns",
                "Work with limited data",
                "Best predictive performance",
                "Detect anomalies",
                "Forecast time series",
                "Segment customers",
                "Generate content"
            ],
            "Consider...": [
                "Logistic Regression, Decision Trees, Linear Regression",
                "Random Forest, XGBoost, Neural Networks",
                "Logistic Regression, Decision Trees, SVM",
                "Gradient Boosting, Neural Networks",
                "Isolation Forest, Autoencoders, DBSCAN",
                "ARIMA, LSTM, Prophet",
                "K-Means, Hierarchical Clustering",
                "LLMs, GANs (Generative AI)"
            ],
            "Avoid...": [
                "Deep Neural Networks, Complex Ensembles",
                "Simple Linear Models",
                "Deep Learning, Complex Neural Networks",
                "Simple Linear Models (unless data is linear)",
                "Standard Classification Models",
                "Cross-sectional Models",
                "Supervised Learning (without labels)",
                "Traditional ML (for generation tasks)"
            ]
        })
        
        st.table(selection_guide)
    
    # ========================================================================
    # DECISION ADVISOR PAGE
    # ========================================================================
    elif page == "🎯 Decision Advisor":
        st.markdown('<p class="main-header">🎯 AI/ML Decision Advisor</p>', unsafe_allow_html=True)
        st.markdown('<p class="sub-header">Answer a few questions to get personalized algorithm recommendations</p>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Initialize session state for form
        if 'advisor_submitted' not in st.session_state:
            st.session_state.advisor_submitted = False
        
        with st.form("advisor_form"):
            st.markdown("### 📋 Tell us about your fintech project")
            
            col1, col2 = st.columns(2)
            
            with col1:
                problem_type = st.selectbox(
                    "1️⃣ What type of problem are you solving?",
                    ["Classification", "Regression", "Clustering", "Anomaly Detection", 
                     "Time Series Forecasting", "Content Generation", "Sequential Decision Making"],
                    help="Classification: Categorize into groups | Regression: Predict a number | Clustering: Find natural groups"
                )
                
                data_size = st.selectbox(
                    "2️⃣ How much labeled data do you have?",
                    ["Small (<1,000 samples)", "Medium (1,000 - 100,000 samples)", 
                     "Large (>100,000 samples)", "No labeled data"],
                    help="The amount of data significantly impacts algorithm choice"
                )
                
                interpretability = st.selectbox(
                    "3️⃣ How important is model interpretability?",
                    ["Critical (Regulatory requirement)", "Important", "Nice to have", "Not important"],
                    help="Regulated industries often require explainable models"
                )
            
            with col2:
                real_time = st.selectbox(
                    "4️⃣ Do you need real-time predictions?",
                    ["Yes", "No", "Near real-time (seconds)"],
                    help="Real-time needs may limit model complexity"
                )
                
                domain = st.selectbox(
                    "5️⃣ What fintech domain is this for?",
                    list(FINTECH_DOMAINS.keys()),
                    help="Different domains have different best practices"
                )
                
                team_expertise = st.selectbox(
                    "6️⃣ What is your team's ML expertise level?",
                    ["Beginner", "Intermediate", "Advanced", "Expert"],
                    help="Some algorithms require more expertise to implement correctly"
                )
            
            st.markdown("---")
            
            project_description = st.text_area(
                "7️⃣ Briefly describe your project (optional):",
                placeholder="E.g., 'We want to predict which loan applicants will default based on their application data and credit history...'",
                height=100
            )
            
            submitted = st.form_submit_button("🚀 Get Recommendations", use_container_width=True)
        
        if submitted:
            st.session_state.advisor_submitted = True
            st.session_state.answers = {
                "problem_type": problem_type,
                "data_size": data_size,
                "interpretability": interpretability,
                "real_time": real_time,
                "domain": domain,
                "team_expertise": team_expertise,
                "project_description": project_description
            }
        
        if st.session_state.advisor_submitted:
            answers = st.session_state.answers
            
            st.markdown("---")
            st.markdown("## 🎯 Your Personalized Recommendations")
            
            recommendations = get_recommendation(answers)
            
            if recommendations:
                # Check if it's a generative AI recommendation
                gen_ai_recs = [r for r in recommendations if r.get("type") == "Generative AI"]
                ml_recs = [r for r in recommendations if r.get("algorithm")]
                
                if gen_ai_recs:
                    st.markdown("### 🎨 Generative AI Recommendation")
                    for rec in gen_ai_recs:
                        st.markdown(f"""
                        <div class="success-box">
                        <h4>💡 {rec['recommendation']}</h4>
                        <p>{rec['reason']}</p>
                        <p><strong>Consider:</strong> GPT-4, Claude, or similar LLMs for text generation; 
                        GANs or Diffusion Models for image/data generation.</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                if ml_recs:
                    st.markdown("### 📊 Top Algorithm Recommendations")
                    
                    for i, rec in enumerate(ml_recs[:3], 1):
                        score_color = "🥇" if i == 1 else "🥈" if i == 2 else "🥉"
                        
                        with st.expander(f"{score_color} #{i}: {rec['algorithm']} (Score: {rec['score']})", expanded=(i==1)):
                            col1, col2 = st.columns([2, 1])
                            
                            with col1:
                                st.markdown(f"**Category:** {rec['category']}")
                                st.markdown(f"**Type:** {rec['type']}")
                                
                                st.markdown("**Why this algorithm?**")
                                for use_case in rec['use_cases'][:3]:
                                    st.markdown(f"- {use_case}")
                                
                                st.markdown("**Pros:**")
                                for pro in rec['pros'][:3]:
                                    st.markdown(f"- ✅ {pro}")
                                
                                st.markdown("**Cons:**")
                                for con in rec['cons'][:2]:
                                    st.markdown(f"- ⚠️ {con}")
                            
                            with col2:
                                st.metric("Complexity", rec['complexity'])
                                st.metric("Interpretability", rec['interpretability'])
                
                # Domain-specific advice
                st.markdown("---")
                st.markdown(f"### 💼 {answers['domain']} Domain Insights")
                
                domain_info = FINTECH_DOMAINS[answers['domain']]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Common Problems in this Domain:**")
                    for problem in domain_info['problems']:
                        st.markdown(f"- {problem}")
                
                with col2:
                    st.markdown("**Recommended Approaches:**")
                    for approach in domain_info['recommended_approaches']:
                        st.markdown(f"- {approach}")
                
                # Next steps
                st.markdown("---")
                st.markdown("### 📝 Recommended Next Steps")
                
                steps = [
                    "**Data Assessment:** Evaluate your data quality, completeness, and potential biases",
                    "**Proof of Concept:** Start with a simple model to establish a baseline",
                    "**Iterate:** Gradually increase complexity if needed",
                    "**Validation:** Implement proper cross-validation and out-of-time testing",
                    "**Compliance Check:** Ensure your approach meets regulatory requirements",
                    "**Monitoring:** Plan for model monitoring and retraining"
                ]
                
                for i, step in enumerate(steps, 1):
                    st.markdown(f"{i}. {step}")
            
            else:
                st.warning("Please answer more questions to get recommendations.")
    
    # ========================================================================
    # FINTECH USE CASES PAGE
    # ========================================================================
    elif page == "💼 Fintech Use Cases":
        st.markdown('<p class="main-header">💼 Fintech Use Cases</p>', unsafe_allow_html=True)
        st.markdown('<p class="sub-header">Real-world AI/ML applications across fintech domains</p>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        domain = st.selectbox("Select a Fintech Domain:", list(FINTECH_DOMAINS.keys()))
        
        st.markdown(f"## {domain}")
        
        domain_info = FINTECH_DOMAINS[domain]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🎯 Key Problems")
            for problem in domain_info['problems']:
                st.info(problem)
        
        with col2:
            st.markdown("### 🛠️ Recommended Approaches")
            for approach in domain_info['recommended_approaches']:
                st.success(approach)
        
        st.markdown("---")
        
        # Detailed use case cards
        st.markdown("### 📚 Detailed Use Cases")
        
        use_cases = {
            "Payments & Transactions": [
                {
                    "title": "Real-time Fraud Detection",
                    "description": "Detect fraudulent transactions as they occur using ML models that analyze transaction patterns, device fingerprints, and behavioral biometrics.",
                    "algorithms": ["XGBoost", "Neural Networks", "Isolation Forest"],
                    "challenges": ["Low latency requirements", "Class imbalance", "Evolving fraud patterns"],
                    "success_metrics": ["Fraud detection rate", "False positive rate", "Processing time"]
                },
                {
                    "title": "Transaction Categorization",
                    "description": "Automatically categorize transactions for personal finance management apps and accounting software.",
                    "algorithms": ["NLP + Classification", "BERT-based models", "Random Forest"],
                    "challenges": ["Merchant name variations", "Multi-language support", "New merchant categories"],
                    "success_metrics": ["Classification accuracy", "Coverage rate", "User correction rate"]
                }
            ],
            "Lending & Credit": [
                {
                    "title": "Credit Scoring",
                    "description": "Assess creditworthiness using traditional and alternative data sources to predict loan default probability.",
                    "algorithms": ["Logistic Regression", "XGBoost", "Neural Networks with explainability"],
                    "challenges": ["Regulatory compliance", "Fair lending requirements", "Model interpretability"],
                    "success_metrics": ["AUC-ROC", "Gini coefficient", "Approval rate at target default rate"]
                },
                {
                    "title": "Dynamic Loan Pricing",
                    "description": "Optimize interest rates based on risk assessment, market conditions, and competitive positioning.",
                    "algorithms": ["Gradient Boosting", "Reinforcement Learning", "Optimization algorithms"],
                    "challenges": ["Balance risk and competitiveness", "Regulatory pricing constraints", "Market dynamics"],
                    "success_metrics": ["Portfolio yield", "Default rate", "Market share"]
                }
            ],
            "Investment & Trading": [
                {
                    "title": "Algorithmic Trading",
                    "description": "Develop automated trading strategies using ML to identify patterns and execute trades.",
                    "algorithms": ["LSTM", "Reinforcement Learning", "Ensemble methods"],
                    "challenges": ["Market regime changes", "Transaction costs", "Overfitting to historical data"],
                    "success_metrics": ["Sharpe ratio", "Maximum drawdown", "Alpha generation"]
                },
                {
                    "title": "Portfolio Optimization",
                    "description": "Construct and rebalance portfolios to maximize returns for given risk levels using ML-enhanced optimization.",
                    "algorithms": ["Mean-Variance Optimization", "Deep Learning", "RL for dynamic allocation"],
                    "challenges": ["Estimation error", "Transaction costs", "Changing correlations"],
                    "success_metrics": ["Risk-adjusted returns", "Tracking error", "Turnover"]
                }
            ],
            "Insurance": [
                {
                    "title": "Claims Prediction",
                    "description": "Predict insurance claims likelihood and severity for pricing and reserving purposes.",
                    "algorithms": ["GLMs", "XGBoost", "Survival Analysis"],
                    "challenges": ["Long-tail distributions", "Regulatory requirements", "Data quality"],
                    "success_metrics": ["Loss ratio", "Combined ratio", "Prediction accuracy"]
                },
                {
                    "title": "Underwriting Automation",
                    "description": "Automate underwriting decisions using ML models that assess risk from application data.",
                    "algorithms": ["Decision Trees", "Gradient Boosting", "Rule engines + ML hybrid"],
                    "challenges": ["Explainability requirements", "Edge cases", "Regulatory approval"],
                    "success_metrics": ["Straight-through processing rate", "Loss ratio by segment", "Processing time"]
                }
            ],
            "RegTech & Compliance": [
                {
                    "title": "Anti-Money Laundering (AML)",
                    "description": "Detect suspicious transaction patterns and potential money laundering activities using ML.",
                    "algorithms": ["Graph Neural Networks", "Anomaly Detection", "Network Analysis"],
                    "challenges": ["High false positive rates", "Evolving typologies", "Regulatory expectations"],
                    "success_metrics": ["SAR quality", "Alert-to-case ratio", "Detection rate"]
                },
                {
                    "title": "KYC Automation",
                    "description": "Automate Know Your Customer processes using document processing and verification ML.",
                    "algorithms": ["OCR + NLP", "Face Recognition", "Document Classification"],
                    "challenges": ["Document variations", "Fraud attempts", "Global compliance"],
                    "success_metrics": ["Automation rate", "Accuracy", "Processing time"]
                }
            ],
            "Customer Experience": [
                {
                    "title": "Intelligent Chatbots",
                    "description": "Deploy conversational AI to handle customer inquiries, transactions, and financial advice.",
                    "algorithms": ["LLMs (GPT, Claude)", "Intent Classification", "Retrieval-Augmented Generation"],
                    "challenges": ["Accuracy for financial queries", "Escalation handling", "Regulatory compliance"],
                    "success_metrics": ["Containment rate", "CSAT", "Query resolution time"]
                },
                {
                    "title": "Personalized Recommendations",
                    "description": "Provide personalized product recommendations based on customer behavior and financial profile.",
                    "algorithms": ["Collaborative Filtering", "Content-based Filtering", "Deep Learning Recommenders"],
                    "challenges": ["Cold start problem", "Privacy concerns", "Regulatory constraints"],
                    "success_metrics": ["Conversion rate", "Cross-sell ratio", "Customer engagement"]
                }
            ]
        }
        
        if domain in use_cases:
            for use_case in use_cases[domain]:
                with st.expander(f"📌 {use_case['title']}", expanded=True):
                    st.markdown(f"**{use_case['description']}**")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown("**Recommended Algorithms:**")
                        for algo in use_case['algorithms']:
                            st.markdown(f"- {algo}")
                    
                    with col2:
                        st.markdown("**Key Challenges:**")
                        for challenge in use_case['challenges']:
                            st.markdown(f"- ⚠️ {challenge}")
                    
                    with col3:
                        st.markdown("**Success Metrics:**")
                        for metric in use_case['success_metrics']:
                            st.markdown(f"- 📊 {metric}")
    
    # ========================================================================
    # ALGORITHM COMPARISON PAGE
    # ========================================================================
    elif page == "📈 Algorithm Comparison":
        st.markdown('<p class="main-header">📈 Algorithm Comparison</p>', unsafe_allow_html=True)
        st.markdown('<p class="sub-header">Compare algorithms across different dimensions</p>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Complexity comparison
        st.markdown("### 📊 Complexity Comparison")
        fig_complexity = create_comparison_chart(ML_ALGORITHMS, "complexity")
        st.plotly_chart(fig_complexity, use_container_width=True)
        
        # Interpretability comparison
        st.markdown("### 🔍 Interpretability Comparison")
        fig_interpret = create_comparison_chart(ML_ALGORITHMS, "interpretability")
        st.plotly_chart(fig_interpret, use_container_width=True)
        
        # Trade-off visualization
        st.markdown("### ⚖️ Complexity vs Interpretability Trade-off")
        
        trade_off_data = []
        for category, algos in ML_ALGORITHMS.items():
            for name, details in algos.items():
                complexity_map = {"Low": 1, "Medium": 2, "Medium to High": 2.5, "High": 3, "Very High": 4}
                interpretability_map = {"Very Low": 1, "Low": 2, "Medium": 3, "High": 4}
                
                trade_off_data.append({
                    "Algorithm": name,
                    "Category": category,
                    "Complexity": complexity_map.get(details.get("complexity", "Medium"), 2),
                    "Interpretability": interpretability_map.get(details.get("interpretability", "Medium"), 3)
                })
        
        df_trade_off = pd.DataFrame(trade_off_data)
        
        fig_trade = px.scatter(
            df_trade_off,
            x="Complexity",
            y="Interpretability",
            color="Category",
            text="Algorithm",
            title="Algorithm Trade-offs: Complexity vs Interpretability",
            labels={"Complexity": "Complexity (1=Low, 4=Very High)", 
                   "Interpretability": "Interpretability (1=Very Low, 4=High)"}
        )
        fig_trade.update_traces(textposition='top center')
        fig_trade.update_layout(height=600)
        st.plotly_chart(fig_trade, use_container_width=True)
        
        st.markdown("""
        <div class="info-box">
        <strong>💡 Key Insight:</strong> There's often a trade-off between model complexity and interpretability. 
        For regulated fintech applications, you may need to sacrifice some predictive performance for explainability.
        </div>
        """, unsafe_allow_html=True)
    
    # ========================================================================
    # RESOURCES PAGE
    # ========================================================================
    elif page == "📚 Resources":
        st.markdown('<p class="main-header">📚 Learning Resources</p>', unsafe_allow_html=True)
        st.markdown('<p class="sub-header">Further reading and tools for fintech AI/ML</p>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.markdown("### 📖 Recommended Reading")
        
        resources = {
            "Books": [
                "Machine Learning for Asset Managers - Marcos López de Prado",
                "Advances in Financial Machine Learning - Marcos López de Prado",
                "Deep Learning for Finance - Jannes Klaas",
                "Python for Finance - Yves Hilpisch",
                "Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow - Aurélien Géron"
            ],
            "Research Papers": [
                "XGBoost: A Scalable Tree Boosting System",
                "Attention Is All You Need (Transformers)",
                "Deep Learning in Finance: A Literature Review",
                "Explainable AI for Credit Scoring"
            ],
            "Online Courses": [
                "Coursera: Machine Learning Specialization",
                "Fast.ai: Practical Deep Learning",
                "Udacity: AI for Trading Nanodegree",
                "edX: Fintech courses from various universities"
            ]
        }
        
        for category, items in resources.items():
            st.markdown(f"#### {category}")
            for item in items:
                st.markdown(f"- {item}")
        
        st.markdown("---")
        
        st.markdown("### 🛠️ Useful Python Libraries")
        
        libraries = pd.DataFrame({
            "Category": ["General ML", "General ML", "Deep Learning", "Deep Learning", 
                        "Time Series", "Explainability", "Visualization", "Data Processing"],
            "Library": ["scikit-learn", "XGBoost/LightGBM", "TensorFlow", "PyTorch",
                       "Prophet/statsmodels", "SHAP/LIME", "Plotly/Matplotlib", "Pandas/NumPy"],
            "Use Case": [
                "Traditional ML algorithms, preprocessing, evaluation",
                "Gradient boosting, tabular data",
                "Deep learning, production deployment",
                "Deep learning, research",
                "Time series forecasting",
                "Model interpretability",
                "Data visualization",
                "Data manipulation and analysis"
            ]
        })
        
        st.table(libraries)
        
        st.markdown("---")
        
        st.markdown("### 📋 ML Project Checklist")
        
        checklist = [
            "☐ **Problem Definition:** Clearly define the business problem and success metrics",
            "☐ **Data Assessment:** Evaluate data quality, completeness, and potential biases",
            "☐ **Exploratory Data Analysis:** Understand distributions, correlations, and patterns",
            "☐ **Feature Engineering:** Create meaningful features from raw data",
            "☐ **Model Selection:** Choose appropriate algorithms based on requirements",
            "☐ **Validation Strategy:** Implement proper cross-validation and out-of-time testing",
            "☐ **Hyperparameter Tuning:** Optimize model parameters systematically",
            "☐ **Model Interpretation:** Ensure the model's decisions can be explained",
            "☐ **Bias and Fairness:** Check for discriminatory patterns in predictions",
            "☐ **Documentation:** Document methodology, assumptions, and limitations",
            "☐ **Deployment Planning:** Plan for production deployment and monitoring",
            "☐ **Monitoring Setup:** Implement model performance and drift monitoring"
        ]
        
        for item in checklist:
            st.markdown(item)

# ============================================================================
# RUN APPLICATION
# ============================================================================

if __name__ == "__main__":
    main()
