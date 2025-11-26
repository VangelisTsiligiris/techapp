"""
Fintech AI/ML Portal - Educational Tool Students
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="Fintech AI/ML Portal",
    page_icon="💹",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium Custom CSS with Modern Design System
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
    
    :root {
        --primary-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        --secondary-gradient: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        --accent-gradient: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        --success-gradient: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        --warning-gradient: linear-gradient(135deg, #F2994A 0%, #F2C94C 100%);
        --dark-gradient: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        
        --primary: #667eea;
        --primary-dark: #5a67d8;
        --secondary: #764ba2;
        --accent: #4facfe;
        --success: #38ef7d;
        --warning: #F2994A;
        --danger: #f5576c;
        
        --text-primary: #1a1a2e;
        --text-secondary: #4a5568;
        --text-muted: #718096;
        --bg-light: #f7fafc;
        --bg-card: #ffffff;
        --border-light: #e2e8f0;
    }
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    }
    
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] span,
    [data-testid="stSidebar"] label {
        color: #e2e8f0 !important;
    }
    
    [data-testid="stSidebar"] hr {
        border-color: rgba(255,255,255,0.1);
    }
    
    /* Custom Radio Buttons for Navigation */
    [data-testid="stSidebar"] .stRadio > div {
        gap: 4px;
    }
    
    [data-testid="stSidebar"] .stRadio > div > label {
        background: rgba(255,255,255,0.05);
        border-radius: 12px;
        padding: 12px 16px;
        margin: 2px 0;
        transition: all 0.3s ease;
        border: 1px solid transparent;
    }
    
    [data-testid="stSidebar"] .stRadio > div > label:hover {
        background: rgba(255,255,255,0.1);
        border-color: rgba(102, 126, 234, 0.5);
        transform: translateX(4px);
    }
    
    [data-testid="stSidebar"] .stRadio > div > label[data-checked="true"] {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.3) 0%, rgba(118, 75, 162, 0.3) 100%);
        border-color: #667eea;
    }
    
    /* Hero Section */
    .hero-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        border-radius: 24px;
        padding: 3rem;
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
    }
    
    .hero-container::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -50%;
        width: 100%;
        height: 100%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
        animation: pulse 4s ease-in-out infinite;
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); opacity: 0.5; }
        50% { transform: scale(1.1); opacity: 0.8; }
    }
    
    .hero-title {
        font-size: 3.2rem;
        font-weight: 800;
        color: white !important;
        text-align: center;
        margin-bottom: 0.5rem;
        text-shadow: 0 4px 20px rgba(0,0,0,0.2);
        position: relative;
        z-index: 1;
    }
    
    .hero-subtitle {
        font-size: 1.25rem;
        color: rgba(255,255,255,0.9);
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 400;
        position: relative;
        z-index: 1;
    }
    
    /* Stats Cards */
    .stats-container {
        display: flex;
        justify-content: center;
        gap: 2rem;
        flex-wrap: wrap;
        position: relative;
        z-index: 1;
    }
    
    .stat-card {
        background: rgba(255,255,255,0.15);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        padding: 1.5rem 2rem;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.2);
        min-width: 140px;
    }
    
    .stat-number {
        font-size: 2.5rem;
        font-weight: 800;
        color: white;
        line-height: 1;
    }
    
    .stat-label {
        font-size: 0.9rem;
        color: rgba(255,255,255,0.8);
        margin-top: 0.5rem;
        font-weight: 500;
    }
    
    /* Section Headers */
    .section-header {
        font-size: 1.75rem;
        font-weight: 700;
        color: var(--text-primary);
        margin: 2rem 0 1.5rem 0;
        padding-bottom: 0.75rem;
        border-bottom: 3px solid;
        border-image: var(--primary-gradient) 1;
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }
    
    .section-icon {
        width: 36px;
        height: 36px;
        background: var(--primary-gradient);
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-weight: 700;
    }
    
    /* Cards */
    .feature-card {
        background: white;
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 10px 40px rgba(0,0,0,0.08);
        transition: all 0.3s ease;
        border: 1px solid var(--border-light);
        height: 100%;
    }
    
    .feature-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 20px 60px rgba(102, 126, 234, 0.15);
        border-color: var(--primary);
    }
    
    .feature-icon {
        width: 64px;
        height: 64px;
        border-radius: 16px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.75rem;
        margin-bottom: 1.25rem;
    }
    
    .feature-icon.purple { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
    .feature-icon.blue { background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); }
    .feature-icon.green { background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); }
    .feature-icon.orange { background: linear-gradient(135deg, #F2994A 0%, #F2C94C 100%); }
    .feature-icon.pink { background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); }
    
    .feature-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: var(--text-primary);
        margin-bottom: 0.75rem;
    }
    
    .feature-desc {
        color: var(--text-secondary);
        font-size: 0.95rem;
        line-height: 1.6;
    }
    
    /* Info Boxes */
    .info-box {
        background: linear-gradient(135deg, #f8fafc 0%, #edf2f7 100%);
        border-left: 4px solid var(--primary);
        padding: 1.25rem 1.5rem;
        margin: 1rem 0;
        border-radius: 0 16px 16px 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
    }
    
    .success-box {
        background: linear-gradient(135deg, #f0fff4 0%, #c6f6d5 100%);
        border-left: 4px solid #38ef7d;
        padding: 1.25rem 1.5rem;
        margin: 1rem 0;
        border-radius: 0 16px 16px 0;
        box-shadow: 0 4px 15px rgba(56, 239, 125, 0.1);
    }
    
    .warning-box {
        background: linear-gradient(135deg, #fffaf0 0%, #feebc8 100%);
        border-left: 4px solid #F2994A;
        padding: 1.25rem 1.5rem;
        margin: 1rem 0;
        border-radius: 0 16px 16px 0;
        box-shadow: 0 4px 15px rgba(242, 153, 74, 0.1);
    }
    
    .danger-box {
        background: linear-gradient(135deg, #fff5f5 0%, #fed7d7 100%);
        border-left: 4px solid #f5576c;
        padding: 1.25rem 1.5rem;
        margin: 1rem 0;
        border-radius: 0 16px 16px 0;
        box-shadow: 0 4px 15px rgba(245, 87, 108, 0.1);
    }
    
    /* Algorithm Cards */
    .algo-card {
        background: white;
        border-radius: 16px;
        padding: 1.5rem;
        margin: 0.75rem 0;
        box-shadow: 0 4px 20px rgba(0,0,0,0.06);
        border: 1px solid var(--border-light);
        transition: all 0.3s ease;
    }
    
    .algo-card:hover {
        box-shadow: 0 8px 30px rgba(102, 126, 234, 0.12);
        border-color: var(--primary);
    }
    
    /* Step Indicators */
    .step-container {
        display: flex;
        align-items: flex-start;
        gap: 1rem;
        padding: 1.25rem;
        background: white;
        border-radius: 16px;
        margin: 0.75rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        border: 1px solid var(--border-light);
        transition: all 0.3s ease;
    }
    
    .step-container:hover {
        transform: translateX(8px);
        border-color: var(--primary);
    }
    
    .step-number {
        min-width: 40px;
        height: 40px;
        background: var(--primary-gradient);
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-weight: 700;
        font-size: 1.1rem;
    }
    
    .step-content h4 {
        margin: 0 0 0.25rem 0;
        color: var(--text-primary);
        font-weight: 600;
    }
    
    .step-content p {
        margin: 0;
        color: var(--text-secondary);
        font-size: 0.9rem;
    }
    
    /* Resource Links */
    .resource-card {
        display: flex;
        align-items: center;
        gap: 1rem;
        padding: 1rem 1.25rem;
        background: white;
        border-radius: 12px;
        margin: 0.5rem 0;
        text-decoration: none;
        color: var(--text-primary);
        border: 1px solid var(--border-light);
        transition: all 0.3s ease;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }
    
    .resource-card:hover {
        background: linear-gradient(135deg, #f8fafc 0%, #edf2f7 100%);
        border-color: var(--primary);
        transform: translateX(8px);
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.1);
    }
    
    .resource-icon {
        width: 44px;
        height: 44px;
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.25rem;
        flex-shrink: 0;
    }
    
    .resource-icon.lib { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
    .resource-icon.course { background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); }
    .resource-icon.doc { background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); }
    .resource-icon.book { background: linear-gradient(135deg, #F2994A 0%, #F2C94C 100%); }
    .resource-icon.paper { background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); }
    
    .resource-info {
        flex-grow: 1;
    }
    
    .resource-title {
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: 0.25rem;
    }
    
    .resource-meta {
        font-size: 0.85rem;
        color: var(--text-muted);
    }
    
    .resource-badge {
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
    }
    
    .badge-free {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        color: #155724;
    }
    
    .badge-paid {
        background: linear-gradient(135deg, #e2e3e5 0%, #d6d8db 100%);
        color: #383d41;
    }
    
    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: var(--bg-light);
        padding: 8px;
        border-radius: 16px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 12px;
        padding: 12px 24px;
        font-weight: 600;
        color: var(--text-secondary);
    }
    
    .stTabs [aria-selected="true"] {
        background: var(--primary-gradient) !important;
        color: white !important;
    }
    
    /* Expander Styling */
    div[data-testid="stExpander"] {
        border: 1px solid var(--border-light);
        border-radius: 16px;
        margin: 0.75rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.04);
        overflow: hidden;
    }
    
    div[data-testid="stExpander"] summary {
        font-weight: 600;
    }
    
    /* Form Styling */
    .stSelectbox > div > div {
        border-radius: 12px;
    }
    
    .stTextArea > div > div > textarea {
        border-radius: 12px;
    }
    
    /* Recommendation Cards */
    .rec-card {
        background: white;
        border-radius: 20px;
        padding: 1.75rem;
        margin: 1rem 0;
        box-shadow: 0 10px 40px rgba(0,0,0,0.08);
        border: 2px solid transparent;
        transition: all 0.3s ease;
    }
    
    .rec-card.top-pick {
        border-color: var(--primary);
        background: linear-gradient(135deg, #ffffff 0%, #f0f4ff 100%);
    }
    
    .rec-card.top-pick::before {
        content: '★ TOP RECOMMENDATION';
        display: block;
        background: var(--primary-gradient);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        font-size: 0.75rem;
        font-weight: 700;
        margin-bottom: 1rem;
        display: inline-block;
    }
    
    .rec-score {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        background: var(--primary-gradient);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 25px;
        font-weight: 700;
        font-size: 0.9rem;
    }
    
    /* Table Styling */
    .dataframe {
        border-radius: 12px !important;
        overflow: hidden;
    }
    
    /* Comparison Table */
    .comparison-table {
        width: 100%;
        border-collapse: separate;
        border-spacing: 0;
        border-radius: 16px;
        overflow: hidden;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
    }
    
    .comparison-table th {
        background: var(--primary-gradient);
        color: white;
        padding: 1rem;
        font-weight: 600;
        text-align: left;
    }
    
    .comparison-table td {
        padding: 1rem;
        border-bottom: 1px solid var(--border-light);
        background: white;
    }
    
    .comparison-table tr:last-child td {
        border-bottom: none;
    }
    
    .comparison-table tr:hover td {
        background: var(--bg-light);
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# COMPREHENSIVE DATA DEFINITIONS
# ============================================================================

AI_TYPES = {
    "Generative AI": {
        "description": """Generative AI represents a revolutionary class of artificial intelligence systems capable of creating 
        entirely new content—text, images, code, music, and synthetic data—by learning complex patterns and distributions 
        from massive training datasets. Unlike traditional AI that analyzes and classifies, generative AI produces novel 
        outputs that didn't exist before, making it transformative for creative and content-heavy applications.""",
        
        "detailed_explanation": """
        **How It Works:**
        Generative AI models learn the underlying probability distribution of training data. When prompted, they sample 
        from this learned distribution to create new, realistic outputs. Modern generative AI primarily uses:
        
        • **Transformer Architecture**: The backbone of large language models (LLMs) like GPT-4, Claude, and Llama. 
          Uses self-attention mechanisms to understand context and relationships across entire sequences.
        
        • **Generative Adversarial Networks (GANs)**: Two neural networks compete—a generator creates fake data while 
          a discriminator tries to distinguish real from fake. This adversarial process produces highly realistic outputs.
        
        • **Variational Autoencoders (VAEs)**: Encode data into a compressed latent space, then decode to generate 
          new samples. Useful for controlled generation and interpolation.
        
        • **Diffusion Models**: Gradually add noise to data, then learn to reverse the process. Powers state-of-the-art 
          image generation (DALL-E 3, Midjourney, Stable Diffusion).
        
        **Training Requirements:**
        Generative AI requires massive computational resources and data. GPT-4 was trained on hundreds of billions of 
        tokens using thousands of GPUs. This scale enables emergent capabilities but also creates significant barriers 
        to entry for custom model development.
        """,
        
        "key_characteristics": [
            "Creates original, novel content from learned patterns",
            "Understands and generates human-like language",
            "Can perform zero-shot and few-shot learning",
            "Exhibits emergent capabilities at scale",
            "Handles unstructured data (text, images, audio)"
        ],
        
        "fintech_applications": [
            ("Intelligent Virtual Assistants", "24/7 customer support chatbots that understand complex financial queries, explain products, and guide users through processes"),
            ("Automated Report Generation", "Transform raw data into narrative financial reports, earnings summaries, and market analyses"),
            ("Synthetic Data Generation", "Create realistic but privacy-preserving datasets for model training, testing, and regulatory sandboxes"),
            ("Document Intelligence", "Extract, summarize, and analyze financial documents, contracts, and regulatory filings"),
            ("Code Generation", "Assist developers in writing trading algorithms, data pipelines, and analytical tools"),
            ("Personalized Communications", "Generate tailored financial advice, marketing content, and customer notifications")
        ],
        
        "technologies": [
            ("Large Language Models", "GPT-4, Claude, Llama, PaLM"),
            ("Image Generation", "DALL-E, Midjourney, Stable Diffusion"),
            ("GANs", "StyleGAN, CycleGAN for synthetic data"),
            ("VAEs", "For controlled generation and anomaly detection"),
            ("RAG Systems", "Retrieval-Augmented Generation for accurate responses")
        ],
        
        "pros": [
            "Dramatically reduces content creation time and cost",
            "Enables 24/7 intelligent customer interactions",
            "Can process and synthesize vast amounts of unstructured data",
            "Adapts to new tasks with minimal additional training",
            "Unlocks new product possibilities (AI advisors, auto-documentation)"
        ],
        
        "cons": [
            "Hallucination risk—may generate plausible but incorrect information",
            "Regulatory uncertainty for financial advice applications",
            "High computational costs for training and inference",
            "Potential for misuse (deepfakes, fraud, manipulation)",
            "Difficult to audit and explain decision-making process",
            "Data privacy concerns with training on sensitive information"
        ],
        
        "regulatory_considerations": """
        Financial regulators are still developing frameworks for generative AI. Key concerns include:
        • **Accuracy**: Hallucinations in financial advice could constitute regulatory violations
        • **Explainability**: MiFID II and similar regulations require explainable recommendations
        • **Data Privacy**: GDPR implications of training on customer data
        • **Model Risk**: SR 11-7 guidance applies to AI models in banking decisions
        """,
        
        "complexity": "High",
        "data_requirements": "Massive datasets (billions of tokens/images)",
        "interpretability": "Very Low - Black box"
    },
    
    "Analytical AI": {
        "description": """Analytical AI encompasses the broad category of machine learning and statistical systems designed 
        to analyze existing data, identify patterns, make predictions, and support decision-making. This is the workhorse 
        of fintech—powering credit scoring, fraud detection, algorithmic trading, and risk management with proven, 
        often interpretable methods.""",
        
        "detailed_explanation": """
        **How It Works:**
        Analytical AI learns mathematical relationships between input features and target outcomes from historical data. 
        The approach varies by algorithm family:
        
        • **Supervised Learning**: Given labeled examples (input-output pairs), the model learns a mapping function. 
          Used for classification (fraud/not fraud) and regression (predict price).
        
        • **Unsupervised Learning**: Finds hidden patterns in unlabeled data. Used for clustering (customer segments), 
          dimensionality reduction, and anomaly detection.
        
        • **Reinforcement Learning**: Agent learns optimal actions through trial and error in an environment. 
          Used for trading strategies and dynamic pricing.
        
        • **Time Series Analysis**: Specialized methods for sequential, temporal data. Captures trends, seasonality, 
          and autocorrelation for forecasting.
        
        **Model Selection Principles:**
        The "No Free Lunch" theorem states no single algorithm is best for all problems. Selection depends on:
        - Data size and quality
        - Interpretability requirements  
        - Computational constraints
        - Problem complexity
        - Regulatory environment
        """,
        
        "key_characteristics": [
            "Extracts insights and patterns from structured data",
            "Makes quantitative predictions with confidence estimates",
            "Classifies entities into discrete categories",
            "Identifies anomalies and outliers",
            "Segments populations into meaningful groups",
            "Optimizes decisions under uncertainty"
        ],
        
        "fintech_applications": [
            ("Credit Risk Scoring", "Assess probability of default using application data, bureau scores, and alternative data"),
            ("Fraud Detection", "Real-time identification of suspicious transactions using behavioral patterns"),
            ("Algorithmic Trading", "Automated execution strategies, signal generation, and portfolio optimization"),
            ("Customer Lifetime Value", "Predict long-term profitability for acquisition and retention decisions"),
            ("Churn Prediction", "Identify at-risk customers before they leave for proactive retention"),
            ("Regulatory Compliance", "AML transaction monitoring, sanctions screening, and suspicious activity detection"),
            ("Price Optimization", "Dynamic pricing for loans, insurance premiums, and financial products")
        ],
        
        "technologies": [
            ("Gradient Boosting", "XGBoost, LightGBM, CatBoost"),
            ("Deep Learning", "Neural networks, CNNs, RNNs"),
            ("Ensemble Methods", "Random Forest, Stacking"),
            ("Statistical Models", "GLMs, GAMs, Survival Analysis"),
            ("Time Series", "ARIMA, Prophet, State Space Models")
        ],
        
        "pros": [
            "Mature, well-understood technology with decades of research",
            "Many algorithms offer high interpretability for regulatory compliance",
            "Proven ROI across thousands of financial services implementations",
            "Robust validation frameworks and governance practices exist",
            "Can provide calibrated probability estimates for risk management",
            "Lower computational requirements than generative AI"
        ],
        
        "cons": [
            "Requires high-quality labeled training data",
            "May perpetuate or amplify biases present in historical data",
            "Performance degrades when data distribution shifts",
            "Complex feature engineering often required",
            "Models require regular retraining and monitoring",
            "May miss novel patterns not represented in training data"
        ],
        
        "regulatory_considerations": """
        Analytical AI in finance operates under established regulatory frameworks:
        • **Fair Lending**: ECOA, Fair Housing Act require non-discriminatory models
        • **Model Risk Management**: SR 11-7, OCC 2011-12 define validation standards
        • **Explainability**: GDPR Article 22 provides right to explanation
        • **Documentation**: Models must have comprehensive documentation and audit trails
        """,
        
        "complexity": "Low to High (varies by algorithm)",
        "data_requirements": "Structured, labeled data (thousands to millions of records)",
        "interpretability": "Low to High (varies by algorithm)"
    }
}

ML_ALGORITHMS = {
    "Supervised Learning": {
        "Linear Regression": {
            "type": "Regression",
            "description": "The foundational regression technique that models the relationship between features and a continuous target as a linear combination. Despite its simplicity, it remains highly valuable for its interpretability and serves as an excellent baseline.",
            "detailed_explanation": """
            **Mathematical Foundation:**
            Linear regression finds coefficients (β) that minimize the sum of squared residuals: y = β₀ + β₁x₁ + β₂x₂ + ... + ε
            
            **Key Assumptions:**
            • Linearity: Relationship between X and Y is linear
            • Independence: Observations are independent
            • Homoscedasticity: Constant variance of residuals
            • Normality: Residuals are normally distributed
            
            **Variants:**
            • Ridge Regression (L2): Adds penalty λΣβ² to prevent overfitting
            • Lasso (L1): Adds penalty λΣ|β| for feature selection
            • Elastic Net: Combines L1 and L2 penalties
            
            **When to Use:**
            Best when you need interpretability, have a roughly linear relationship, and want to understand feature importance through coefficients.
            """,
            "fintech_use_cases": [
                "Interest rate prediction based on market factors",
                "Revenue forecasting from historical trends",
                "Asset price modeling for valuation",
                "Cost estimation for operational planning"
            ],
            "pros": ["Highly interpretable coefficients", "Fast training and inference", "Closed-form solution exists", "Works well for linear relationships", "Easy to regularize"],
            "cons": ["Assumes linearity", "Sensitive to outliers", "May underfit complex patterns", "Requires feature scaling for regularized versions"],
            "data_requirements": "Continuous target, numerical features, typically 10+ samples per feature",
            "complexity": "Low",
            "interpretability": "High"
        },
        
        "Logistic Regression": {
            "type": "Classification",
            "description": "The gold standard for binary classification in regulated industries. Uses the logistic function to model probability of class membership, providing both predictions and well-calibrated probability scores essential for risk-based decisions.",
            "detailed_explanation": """
            **Mathematical Foundation:**
            Models log-odds as linear: log(p/(1-p)) = β₀ + β₁x₁ + ... 
            Probability: p = 1/(1 + e^(-z)) where z is the linear combination
            
            **Why It's Preferred in Finance:**
            • Coefficients are interpretable as log-odds ratios
            • Outputs are true probabilities (useful for expected loss calculations)
            • Established regulatory acceptance and validation methods
            • Feature effects can be directly explained to auditors
            
            **Extensions:**
            • Multinomial: For multi-class problems
            • Ordinal: For ordered categories (credit ratings)
            • Regularized: L1/L2 for high-dimensional data
            
            **Scorecard Development:**
            Logistic regression is the basis for traditional credit scorecards. Coefficients are transformed into "points" for each characteristic, summed to produce a score.
            """,
            "fintech_use_cases": [
                "Credit default prediction (PD models)",
                "Fraud classification (fraud/legitimate)",
                "Loan approval decisions",
                "Customer churn prediction",
                "Marketing response modeling"
            ],
            "pros": ["Highly interpretable for regulatory compliance", "Outputs calibrated probabilities", "Fast and scalable", "Well-established validation framework", "Works with sparse data"],
            "cons": ["Assumes linear log-odds", "May underfit complex patterns", "Requires feature engineering for non-linearities", "Sensitive to multicollinearity"],
            "data_requirements": "Binary or categorical target, numerical features, balanced classes preferred",
            "complexity": "Low",
            "interpretability": "High"
        },
        
        "Decision Trees": {
            "type": "Classification/Regression",
            "description": "Intuitive tree-based models that make sequential binary decisions to partition data. The resulting tree structure directly mirrors human decision-making, making it exceptionally easy to explain and audit.",
            "detailed_explanation": """
            **How Trees Split:**
            At each node, the algorithm finds the feature and threshold that best separates classes (classification) or reduces variance (regression). Common criteria:
            • Gini Impurity: Measures probability of incorrect classification
            • Entropy/Information Gain: Measures reduction in uncertainty
            • MSE: For regression tasks
            
            **Tree Structure:**
            • Root Node: First decision, most important feature
            • Internal Nodes: Subsequent decisions
            • Leaf Nodes: Final predictions (class or value)
            
            **Advantages for Compliance:**
            The decision path can be extracted and presented as simple if-then rules, making it easy to explain why a specific decision was made—crucial for adverse action notices in lending.
            
            **Controlling Complexity:**
            • max_depth: Limits tree depth
            • min_samples_split: Minimum samples to split
            • min_samples_leaf: Minimum samples in leaves
            • Pruning: Remove branches that don't improve validation performance
            """,
            "fintech_use_cases": [
                "Credit policy rules extraction",
                "Simple fraud rules",
                "Risk tier assignment",
                "Underwriting decision trees",
                "Customer segmentation"
            ],
            "pros": ["Extremely interpretable", "Handles non-linear relationships", "No feature scaling needed", "Handles mixed data types", "Fast prediction"],
            "cons": ["Prone to overfitting", "Unstable (small data changes cause different trees)", "Greedy algorithm may miss global optimum", "Axis-aligned splits only"],
            "data_requirements": "Any target type, handles categorical and numerical features natively",
            "complexity": "Low",
            "interpretability": "High"
        },
        
        "Random Forest": {
            "type": "Classification/Regression",
            "description": "An ensemble of decision trees that achieves robust predictions through averaging. Each tree is trained on a bootstrap sample with random feature subsets, reducing overfitting and variance while maintaining reasonable interpretability.",
            "detailed_explanation": """
            **Ensemble Mechanism:**
            1. Create B bootstrap samples (sampling with replacement)
            2. For each sample, grow a decision tree
            3. At each split, consider only m random features (typically √p for classification, p/3 for regression)
            4. Aggregate predictions: majority vote (classification) or average (regression)
            
            **Why It Works:**
            • Bagging reduces variance through averaging
            • Random feature selection decorrelates trees
            • Result: Lower generalization error than single tree
            
            **Feature Importance:**
            • Mean Decrease Impurity: Average reduction in Gini/MSE across all splits
            • Permutation Importance: Drop in accuracy when feature is shuffled
            • SHAP values: Game-theoretic feature attribution
            
            **Out-of-Bag (OOB) Evaluation:**
            Each tree doesn't see ~37% of data (OOB samples). These can be used for validation without a separate test set—useful for small datasets.
            """,
            "fintech_use_cases": [
                "Credit scoring with feature importance",
                "Fraud detection ensemble",
                "Transaction classification",
                "Risk factor identification",
                "Missing value imputation"
            ],
            "pros": ["Robust to overfitting", "Handles high-dimensional data", "Built-in feature importance", "No feature scaling needed", "Parallelizable training"],
            "cons": ["Less interpretable than single tree", "Slower inference than linear models", "Memory intensive for large forests", "Can overfit noisy data with many trees"],
            "data_requirements": "Works with various data types, benefits from larger datasets",
            "complexity": "Medium",
            "interpretability": "Medium"
        },
        
        "Gradient Boosting (XGBoost/LightGBM)": {
            "type": "Classification/Regression",
            "description": "State-of-the-art ensemble method that sequentially builds trees to correct errors of previous trees. Dominates Kaggle competitions and is the go-to algorithm for tabular data in production financial systems.",
            "detailed_explanation": """
            **Boosting Mechanism:**
            Unlike bagging (parallel), boosting is sequential:
            1. Fit initial model (often just mean/mode)
            2. Calculate residuals (errors)
            3. Fit new tree to predict residuals
            4. Add new tree with learning rate: F(x) = F(x) + η·h(x)
            5. Repeat until stopping criterion
            
            **XGBoost Innovations:**
            • Regularized objective (L1/L2 on leaf weights)
            • Second-order gradient approximation
            • Sparsity-aware split finding
            • Cache-optimized algorithms
            
            **LightGBM Innovations:**
            • Gradient-based One-Side Sampling (GOSS)
            • Exclusive Feature Bundling (EFB)
            • Leaf-wise growth (vs. level-wise)
            • Much faster training on large data
            
            **Hyperparameter Tuning:**
            Key parameters to tune:
            • learning_rate: Usually 0.01-0.3
            • max_depth: 3-10 typically
            • n_estimators: 100-1000+
            • subsample: 0.5-1.0
            • colsample_bytree: 0.5-1.0
            """,
            "fintech_use_cases": [
                "Credit risk models (PD, LGD, EAD)",
                "Fraud detection systems",
                "Customer lifetime value",
                "Propensity models",
                "Trading signal prediction"
            ],
            "pros": ["State-of-the-art accuracy on tabular data", "Handles missing values natively", "Built-in regularization", "Feature importance built-in", "Fast and scalable"],
            "cons": ["Requires careful hyperparameter tuning", "Can overfit with too many rounds", "Less interpretable than linear models", "Sequential training (less parallelizable)"],
            "data_requirements": "Tabular data, handles missing values, benefits from large datasets",
            "complexity": "Medium",
            "interpretability": "Medium"
        },
        
        "Support Vector Machines (SVM)": {
            "type": "Classification/Regression",
            "description": "Finds the optimal hyperplane that maximizes the margin between classes. Effective in high-dimensional spaces and with clear class separation, though less popular in modern finance due to scalability limitations.",
            "detailed_explanation": """
            **Mathematical Foundation:**
            SVM finds the hyperplane w·x + b = 0 that maximizes the margin (distance to nearest points of each class). These nearest points are "support vectors."
            
            **Kernel Trick:**
            For non-linear boundaries, SVM uses kernels to implicitly map data to higher dimensions:
            • Linear: K(x,y) = x·y
            • Polynomial: K(x,y) = (x·y + c)^d
            • RBF (Gaussian): K(x,y) = exp(-γ||x-y||²)
            
            **Soft Margin:**
            Parameter C controls trade-off between margin width and misclassification. High C = narrow margin, fewer errors on training data.
            
            **Limitations:**
            Training complexity is O(n²) to O(n³), making it impractical for large datasets (>100k samples). For big data, consider:
            • LinearSVC for linear kernels
            • SGDClassifier with hinge loss
            • Approximation methods (Nyström)
            """,
            "fintech_use_cases": [
                "Small-sample classification problems",
                "Text classification (sentiment analysis)",
                "Image-based fraud detection",
                "Market regime detection"
            ],
            "pros": ["Effective in high dimensions", "Memory efficient (only stores support vectors)", "Versatile through kernel selection", "Strong theoretical foundations"],
            "cons": ["Not suitable for large datasets", "Requires feature scaling", "Difficult to interpret", "Poor probability estimates", "Sensitive to kernel choice"],
            "data_requirements": "Numerical features, requires scaling, best for datasets <50k samples",
            "complexity": "Medium",
            "interpretability": "Low"
        },
        
        "Neural Networks": {
            "type": "Classification/Regression",
            "description": "Multi-layer networks of interconnected nodes that learn hierarchical representations. Capable of capturing extremely complex patterns, neural networks power modern deep learning but require large data and careful tuning.",
            "detailed_explanation": """
            **Architecture:**
            • Input Layer: One node per feature
            • Hidden Layers: Learn intermediate representations
            • Output Layer: Predictions (sigmoid for binary, softmax for multi-class, linear for regression)
            
            **Activation Functions:**
            • ReLU: f(x) = max(0, x) - Most common, avoids vanishing gradients
            • Sigmoid: f(x) = 1/(1+e^-x) - Output layer for binary
            • Tanh: f(x) = (e^x - e^-x)/(e^x + e^-x) - Centered around 0
            
            **Training:**
            • Forward pass: Calculate predictions
            • Loss calculation: Compare to true values
            • Backpropagation: Calculate gradients
            • Optimization: Update weights (Adam, SGD with momentum)
            
            **Regularization:**
            • Dropout: Randomly zero activations during training
            • Batch Normalization: Normalize layer inputs
            • Early Stopping: Stop when validation loss increases
            • L2 Regularization: Penalty on weight magnitudes
            
            **Architectures for Finance:**
            • MLP: Standard for tabular data
            • CNN: For image-based inputs (check images, signatures)
            • RNN/LSTM: For sequential data (transactions, time series)
            • Transformers: For NLP tasks (document analysis)
            """,
            "fintech_use_cases": [
                "Complex pattern recognition in high-dimensional data",
                "Image-based document processing (OCR, check reading)",
                "Sequential transaction analysis",
                "Natural language processing for documents",
                "Multi-task learning across related problems"
            ],
            "pros": ["Captures highly complex patterns", "Automatic feature learning", "State-of-the-art for many tasks", "Flexible architecture design", "Transfer learning possible"],
            "cons": ["Requires large training data", "Computationally expensive", "Black box - difficult to interpret", "Many hyperparameters to tune", "Prone to overfitting without regularization"],
            "data_requirements": "Large datasets (10k+ samples minimum), numerical features, GPU recommended",
            "complexity": "High",
            "interpretability": "Low"
        }
    },
    
    "Unsupervised Learning": {
        "K-Means Clustering": {
            "type": "Clustering",
            "description": "Partitions data into K distinct clusters by iteratively assigning points to nearest centroids and updating centroid positions. Simple, fast, and intuitive for customer segmentation and similar grouping tasks.",
            "detailed_explanation": """
            **Algorithm:**
            1. Initialize K centroids (random or k-means++)
            2. Assign each point to nearest centroid
            3. Recalculate centroids as cluster means
            4. Repeat until convergence
            
            **Choosing K:**
            • Elbow Method: Plot inertia vs K, look for "elbow"
            • Silhouette Score: Measures cluster cohesion/separation
            • Gap Statistic: Compare to null reference distribution
            • Domain Knowledge: Business requirements may dictate K
            
            **Limitations:**
            • Assumes spherical clusters of similar size
            • Sensitive to initialization (use k-means++)
            • Sensitive to outliers (consider k-medoids)
            • Must specify K in advance
            
            **Variants:**
            • Mini-batch K-Means: For large datasets
            • K-Medoids: Uses actual points as centers (robust to outliers)
            • Bisecting K-Means: Hierarchical approach
            """,
            "fintech_use_cases": [
                "Customer segmentation by behavior",
                "Transaction pattern grouping",
                "Market regime identification",
                "Portfolio clustering",
                "Risk tier definition"
            ],
            "pros": ["Simple and intuitive", "Scales to large datasets", "Fast convergence", "Easy to interpret clusters"],
            "cons": ["Must specify K", "Assumes spherical clusters", "Sensitive to initialization", "Sensitive to outliers"],
            "data_requirements": "Numerical features, typically scaled",
            "complexity": "Low",
            "interpretability": "Medium"
        },
        
        "Hierarchical Clustering": {
            "type": "Clustering",
            "description": "Builds a tree-like hierarchy of clusters either bottom-up (agglomerative) or top-down (divisive). The dendrogram visualization helps understand data structure and choose appropriate cluster numbers.",
            "detailed_explanation": """
            **Agglomerative (Bottom-up):**
            1. Start with each point as its own cluster
            2. Merge two closest clusters
            3. Repeat until one cluster remains
            
            **Linkage Methods:**
            • Single: Distance between closest points
            • Complete: Distance between furthest points
            • Average: Average pairwise distance
            • Ward: Minimizes within-cluster variance
            
            **Dendrogram:**
            Visual representation of merge hierarchy. Height indicates distance at merge. Cut at different heights to get different numbers of clusters.
            
            **When to Use:**
            • Don't know number of clusters
            • Want to understand hierarchical structure
            • Dataset is small (<10k points)
            • Need deterministic results
            """,
            "fintech_use_cases": [
                "Portfolio hierarchy construction",
                "Customer relationship networks",
                "Risk category taxonomies",
                "Market structure analysis"
            ],
            "pros": ["No need to specify K", "Provides dendrogram visualization", "Captures nested cluster structure", "Deterministic results"],
            "cons": ["Computationally expensive O(n³)", "Not suitable for large datasets", "Sensitive to noise", "Cannot undo merge decisions"],
            "data_requirements": "Numerical features, scaled, typically <10k samples",
            "complexity": "Medium",
            "interpretability": "High"
        },
        
        "DBSCAN": {
            "type": "Clustering",
            "description": "Density-based algorithm that finds clusters as dense regions separated by sparse areas. Unlike K-means, it can discover clusters of arbitrary shape and automatically identifies outliers.",
            "detailed_explanation": """
            **Algorithm:**
            Points are classified as:
            • Core: Has ≥ min_samples within ε radius
            • Border: Within ε of a core point but not core itself
            • Noise: Neither core nor border (outliers!)
            
            Clusters form by connecting core points within ε of each other.
            
            **Parameters:**
            • ε (eps): Radius for neighborhood search
            • min_samples: Minimum points for core status
            
            **Parameter Selection:**
            • k-distance plot: Plot distance to k-th nearest neighbor, look for elbow
            • Domain knowledge: What constitutes "close" in your application?
            
            **Advantages:**
            • Finds arbitrarily shaped clusters
            • Automatically identifies outliers
            • No need to specify cluster count
            • Robust to outliers
            """,
            "fintech_use_cases": [
                "Fraud detection (fraudsters as outliers)",
                "Unusual transaction patterns",
                "Geographic customer clustering",
                "Anomaly detection in market data"
            ],
            "pros": ["Finds arbitrary shapes", "Automatic outlier detection", "No need to specify K", "Robust to outliers"],
            "cons": ["Sensitive to ε and min_samples", "Struggles with varying densities", "Not suitable for high dimensions", "Doesn't assign outliers to clusters"],
            "data_requirements": "Numerical features, scaled, works best in low dimensions",
            "complexity": "Medium",
            "interpretability": "Medium"
        },
        
        "Principal Component Analysis (PCA)": {
            "type": "Dimensionality Reduction",
            "description": "Transforms high-dimensional data into orthogonal principal components that capture maximum variance. Essential for visualization, noise reduction, and handling multicollinearity before modeling.",
            "detailed_explanation": """
            **Mathematical Foundation:**
            PCA finds orthogonal directions (principal components) that maximize variance:
            1. Center data (subtract mean)
            2. Compute covariance matrix
            3. Find eigenvectors/eigenvalues
            4. Project data onto top-k eigenvectors
            
            **Variance Explained:**
            Each component explains a portion of total variance. Cumulative plot helps choose number of components (e.g., keep 95% variance).
            
            **Applications:**
            • Visualization: Project to 2-3 dimensions
            • Noise reduction: Drop low-variance components
            • Multicollinearity: PCs are uncorrelated
            • Feature engineering: Use PCs as model inputs
            
            **Limitations:**
            • Linear transformations only
            • Assumes variance = importance
            • Loses interpretability of original features
            • Sensitive to scaling
            """,
            "fintech_use_cases": [
                "Portfolio risk decomposition",
                "High-dimensional data visualization",
                "Multicollinearity reduction",
                "Feature preprocessing",
                "Yield curve modeling"
            ],
            "pros": ["Reduces dimensionality", "Removes multicollinearity", "Speeds up training", "Noise reduction", "Visualization"],
            "cons": ["Loses interpretability", "Linear only", "Assumes variance = importance", "Sensitive to scaling"],
            "data_requirements": "Numerical features, must be scaled",
            "complexity": "Low",
            "interpretability": "Low"
        },
        
        "Autoencoders": {
            "type": "Dimensionality Reduction/Anomaly Detection",
            "description": "Neural networks that learn compressed representations by training to reconstruct their input. The bottleneck layer captures essential data features, useful for anomaly detection when reconstruction fails.",
            "detailed_explanation": """
            **Architecture:**
            • Encoder: Compresses input to latent space
            • Bottleneck: Low-dimensional representation
            • Decoder: Reconstructs input from latent space
            
            Loss = ||x - decoder(encoder(x))||²
            
            **For Anomaly Detection:**
            Train on normal data only. Anomalies will have high reconstruction error because the model hasn't learned their patterns.
            
            **Variants:**
            • Variational (VAE): Learns probability distribution, enables generation
            • Denoising: Trained to reconstruct from corrupted input
            • Sparse: Encourages sparse latent representations
            • Contractive: Robust to small input changes
            
            **Advantages over PCA:**
            • Captures non-linear relationships
            • More flexible architecture
            • Can generate new samples (VAE)
            """,
            "fintech_use_cases": [
                "Fraud detection via reconstruction error",
                "Non-linear dimensionality reduction",
                "Feature extraction for downstream models",
                "Data denoising",
                "Anomaly detection in transactions"
            ],
            "pros": ["Captures non-linear patterns", "Flexible architecture", "Unsupervised learning", "Anomaly detection capability", "Feature extraction"],
            "cons": ["Requires tuning architecture", "Black box", "Computationally expensive", "Needs substantial data", "Training can be unstable"],
            "data_requirements": "Large datasets preferred, numerical features",
            "complexity": "High",
            "interpretability": "Low"
        }
    },
    
    "Reinforcement Learning": {
        "Q-Learning": {
            "type": "Model-Free RL",
            "description": "Foundational RL algorithm that learns action values (Q-values) through trial and error. The agent learns which actions maximize cumulative reward in each state without needing a model of the environment.",
            "detailed_explanation": """
            **Q-Value:**
            Q(s,a) represents expected cumulative reward from taking action a in state s and following optimal policy thereafter.
            
            **Update Rule:**
            Q(s,a) ← Q(s,a) + α[r + γ·max_a'(Q(s',a')) - Q(s,a)]
            
            Where:
            • α: Learning rate
            • γ: Discount factor (how much to value future rewards)
            • r: Immediate reward
            • s': Next state
            
            **Exploration vs Exploitation:**
            • ε-greedy: Random action with probability ε
            • Softmax: Probability proportional to Q-values
            • UCB: Upper confidence bound exploration
            
            **Limitations:**
            • Requires discrete state/action spaces
            • Doesn't scale to large problems
            • Needs many episodes to converge
            """,
            "fintech_use_cases": [
                "Simple trading strategy optimization",
                "Order routing decisions",
                "Dynamic pricing with discrete actions",
                "Portfolio rebalancing rules"
            ],
            "pros": ["Simple to implement", "Guaranteed convergence under conditions", "Model-free (no environment model needed)", "Foundation for advanced methods"],
            "cons": ["Requires discrete spaces", "Doesn't scale", "Slow convergence", "Exploration-exploitation tradeoff"],
            "data_requirements": "Environment interaction, reward signals, discrete state/action",
            "complexity": "High",
            "interpretability": "Low"
        },
        
        "Deep Q-Networks (DQN)": {
            "type": "Deep RL",
            "description": "Combines Q-learning with deep neural networks to handle complex, high-dimensional state spaces. The breakthrough that enabled RL to master Atari games and inspired modern deep RL applications.",
            "detailed_explanation": """
            **Innovation:**
            Replace Q-table with neural network that approximates Q(s,a) for any state s.
            
            **Key Techniques:**
            • Experience Replay: Store transitions in buffer, sample randomly for training (breaks correlation)
            • Target Network: Separate network for Q-targets, updated periodically (stabilizes training)
            • Frame Stacking: Use multiple recent frames as state (captures dynamics)
            
            **Architecture:**
            For trading: State could include price history, positions, indicators
            Network outputs Q-value for each possible action
            
            **Challenges:**
            • Overestimation of Q-values (Double DQN addresses this)
            • Sample inefficiency
            • Hyperparameter sensitivity
            • Requires massive interaction data
            """,
            "fintech_use_cases": [
                "Algorithmic trading strategies",
                "Market making with complex state",
                "Optimal execution (VWAP, TWAP)",
                "Multi-asset portfolio management"
            ],
            "pros": ["Handles complex states", "Learns from raw inputs", "Powerful representation learning", "Foundation for advanced methods"],
            "cons": ["Unstable training", "Sample inefficient", "Many hyperparameters", "Black box", "Requires simulation environment"],
            "data_requirements": "Large interaction data, simulation environment essential",
            "complexity": "Very High",
            "interpretability": "Very Low"
        },
        
        "Policy Gradient Methods": {
            "type": "Deep RL",
            "description": "Directly optimize the policy function (action probabilities given state) rather than learning value functions. Essential for continuous action spaces like portfolio weights or position sizing.",
            "detailed_explanation": """
            **Approach:**
            Parameterize policy π_θ(a|s) and optimize parameters θ to maximize expected return.
            
            **REINFORCE Algorithm:**
            ∇J(θ) = E[∇log π_θ(a|s) · G_t]
            
            Update policy in direction that increases probability of actions that led to high returns.
            
            **Actor-Critic:**
            • Actor: Policy network (what action to take)
            • Critic: Value network (how good is current state)
            Critic provides lower-variance baseline for actor updates.
            
            **Variants:**
            • A2C/A3C: Advantage Actor-Critic with async training
            • PPO: Proximal Policy Optimization (stable, popular)
            • SAC: Soft Actor-Critic (entropy regularization)
            • DDPG: Deep Deterministic Policy Gradient (continuous)
            
            **For Finance:**
            Action space can be continuous portfolio weights that sum to 1. Not possible with value-based methods.
            """,
            "fintech_use_cases": [
                "Continuous portfolio allocation",
                "Dynamic hedging strategies",
                "Optimal execution with continuous sizing",
                "Multi-period asset allocation"
            ],
            "pros": ["Handles continuous actions", "Direct policy optimization", "Can learn stochastic policies", "Better for high-dimensional action spaces"],
            "cons": ["High variance gradients", "Sample inefficient", "Complex implementation", "Sensitive to hyperparameters"],
            "data_requirements": "Continuous action space, simulation environment",
            "complexity": "Very High",
            "interpretability": "Very Low"
        }
    },
    
    "Time Series & Specialized": {
        "ARIMA/SARIMA": {
            "type": "Time Series",
            "description": "Classical statistical models for time series forecasting that capture autoregressive patterns, trends, and seasonality. Well-understood, interpretable, and still competitive for many financial forecasting tasks.",
            "detailed_explanation": """
            **Components:**
            • AR(p): Autoregressive - depends on p previous values
            • I(d): Integrated - differencing d times for stationarity
            • MA(q): Moving Average - depends on q previous errors
            
            ARIMA(p,d,q): y_t = c + Σφ_i·y_{t-i} + Σθ_i·ε_{t-i} + ε_t
            
            **SARIMA:**
            Adds seasonal components: ARIMA(p,d,q)(P,D,Q)_m
            • P,D,Q: Seasonal AR, differencing, MA orders
            • m: Seasonal period (12 for monthly, 4 for quarterly)
            
            **Model Selection:**
            • ACF/PACF plots to identify p, q
            • AIC/BIC for model comparison
            • Auto-ARIMA for automated selection
            
            **Assumptions:**
            • Stationarity (or differencing achieves it)
            • Linear relationships
            • Constant variance (or use GARCH for variance)
            """,
            "fintech_use_cases": [
                "Economic indicator forecasting",
                "Revenue prediction",
                "Interest rate forecasting",
                "Volatility baseline models",
                "Cash flow forecasting"
            ],
            "pros": ["Well-understood theory", "Interpretable parameters", "Handles trends/seasonality", "Confidence intervals", "Works with limited data"],
            "cons": ["Assumes stationarity", "Linear patterns only", "Manual parameter selection", "Sensitive to outliers", "Doesn't capture complex dynamics"],
            "data_requirements": "Time series data, ideally 50+ observations",
            "complexity": "Medium",
            "interpretability": "High"
        },
        
        "LSTM/GRU": {
            "type": "Time Series (Deep Learning)",
            "description": "Recurrent neural networks with gating mechanisms that capture long-term dependencies in sequential data. Powerful for complex patterns but require substantial data and careful training.",
            "detailed_explanation": """
            **LSTM Architecture:**
            Three gates control information flow:
            • Forget Gate: What to discard from cell state
            • Input Gate: What new information to add
            • Output Gate: What to output from cell state
            
            This addresses vanishing gradient problem of vanilla RNNs, allowing learning over long sequences.
            
            **GRU (Simpler):**
            Two gates: Reset and Update
            Fewer parameters, often similar performance
            
            **For Time Series:**
            • Many-to-one: Sequence → single prediction
            • Many-to-many: Sequence → sequence (multi-step forecast)
            • Encoder-Decoder: For variable length input/output
            
            **Best Practices:**
            • Use dropout for regularization
            • Try bidirectional for some problems
            • Consider attention mechanisms
            • Normalize/scale input sequences
            """,
            "fintech_use_cases": [
                "Price movement prediction",
                "Transaction sequence analysis",
                "Customer behavior sequences",
                "Multi-step forecasting",
                "Event detection in time series"
            ],
            "pros": ["Captures long-term dependencies", "Handles non-linear patterns", "Works with variable length sequences", "State-of-the-art for many tasks"],
            "cons": ["Requires large data", "Expensive to train", "Black box", "Difficult to tune", "May overfit"],
            "data_requirements": "Sequential data, large datasets (1000s of sequences)",
            "complexity": "High",
            "interpretability": "Low"
        },
        
        "Prophet": {
            "type": "Time Series",
            "description": "Facebook's accessible forecasting tool designed for business time series with daily observations and strong seasonal patterns. Handles holidays, missing data, and outliers gracefully.",
            "detailed_explanation": """
            **Model:**
            y(t) = g(t) + s(t) + h(t) + ε_t
            
            • g(t): Trend - piecewise linear or logistic growth
            • s(t): Seasonality - Fourier series for yearly, weekly
            • h(t): Holiday effects - indicator variables
            • ε_t: Error term
            
            **Advantages:**
            • Intuitive parameters (no need to understand ARIMA orders)
            • Automatic changepoint detection
            • Built-in holiday handling
            • Robust to missing data and outliers
            • Uncertainty intervals included
            
            **When to Use:**
            • Daily/weekly business data
            • Strong seasonal patterns
            • Multiple seasonalities
            • Holidays/events impact data
            • Analyst adjustments needed
            
            **Limitations:**
            • Less flexible than custom models
            • Assumes additive components
            • May not capture complex dynamics
            """,
            "fintech_use_cases": [
                "Daily transaction volume forecasting",
                "Revenue prediction",
                "Website traffic forecasting",
                "Customer acquisition forecasting",
                "Operational metric prediction"
            ],
            "pros": ["Easy to use", "Handles seasonality automatically", "Robust to missing data", "Holiday effects", "Uncertainty quantification"],
            "cons": ["Less customizable", "May miss complex patterns", "Not for high-frequency data", "Assumes additive structure"],
            "data_requirements": "Daily or weekly time series, ideally 2+ years for seasonality",
            "complexity": "Low",
            "interpretability": "Medium"
        },
        
        "Isolation Forest": {
            "type": "Anomaly Detection",
            "description": "Efficient unsupervised algorithm that detects anomalies by measuring how easily points can be isolated through random partitioning. Anomalies are isolated quickly; normal points require many splits.",
            "detailed_explanation": """
            **Intuition:**
            Anomalies are "few and different" - they're easier to separate from the rest of the data.
            
            **Algorithm:**
            1. Build trees by randomly selecting feature and split value
            2. Anomalies are isolated in fewer splits (shorter path length)
            3. Anomaly score = average path length across trees
            
            **Advantages:**
            • No need to define "normal" - learns from data structure
            • Linear time complexity: O(n log n)
            • Works well in high dimensions
            • Memory efficient
            
            **Parameters:**
            • n_estimators: Number of trees (100 typical)
            • contamination: Expected proportion of anomalies
            • max_samples: Subsample size for each tree
            
            **Output:**
            Scores from -1 (most anomalous) to 1 (most normal)
            Threshold at contamination percentile
            """,
            "fintech_use_cases": [
                "Fraud detection in transactions",
                "Unusual trading pattern detection",
                "Outlier detection in applications",
                "Network intrusion detection",
                "Market manipulation detection"
            ],
            "pros": ["Fast and scalable", "No labeled data needed", "Works in high dimensions", "Memory efficient", "Few parameters"],
            "cons": ["May miss local anomalies", "Assumes anomalies are global outliers", "Threshold selection subjective", "Random nature requires ensembling"],
            "data_requirements": "Numerical features, typically <1% anomalies",
            "complexity": "Low",
            "interpretability": "Medium"
        }
    }
}

FINTECH_DOMAINS = {
    "Payments & Transactions": {
        "description": "Real-time processing of financial transactions including fraud prevention, categorization, and optimization.",
        "problems": ["Real-time fraud detection", "Transaction categorization", "Payment routing optimization", "Dispute prediction"],
        "recommended_approaches": ["Gradient Boosting (XGBoost)", "Neural Networks", "Isolation Forest", "Rule + ML Hybrid"],
        "key_challenges": ["Sub-100ms latency requirements", "Extreme class imbalance (0.1% fraud)", "24/7 availability", "Adversarial attackers adapting"],
        "success_metrics": ["Fraud detection rate", "False positive rate", "Authorization approval rate", "Processing latency"]
    },
    "Lending & Credit": {
        "description": "Credit decisioning, risk assessment, and loan lifecycle management using traditional and alternative data.",
        "problems": ["Credit scoring", "Default prediction", "Loan pricing", "Early warning systems", "Collections optimization"],
        "recommended_approaches": ["Logistic Regression", "XGBoost/LightGBM", "Random Forest", "Survival Analysis"],
        "key_challenges": ["Regulatory compliance (fair lending)", "Model interpretability requirements", "Through-the-cycle stability", "Alternative data integration"],
        "success_metrics": ["AUC-ROC / Gini", "KS statistic", "PSI (stability)", "Discrimination at cutoff", "Expected vs actual default rate"]
    },
    "Investment & Trading": {
        "description": "Algorithmic trading, portfolio management, and market analysis using quantitative methods.",
        "problems": ["Alpha generation", "Portfolio optimization", "Risk management", "Execution optimization", "Market regime detection"],
        "recommended_approaches": ["LSTM/GRU", "Reinforcement Learning", "Ensemble Methods", "Factor Models"],
        "key_challenges": ["Low signal-to-noise ratio", "Non-stationarity", "Transaction costs", "Overfitting to historical data", "Black swan events"],
        "success_metrics": ["Sharpe ratio", "Maximum drawdown", "Information ratio", "Turnover", "Slippage"]
    },
    "Insurance": {
        "description": "Risk selection, pricing, claims management, and fraud detection across insurance products.",
        "problems": ["Risk assessment", "Premium pricing", "Claims prediction", "Fraud detection", "Reserving"],
        "recommended_approaches": ["GLMs", "Gradient Boosting", "Survival Analysis", "Two-part Models"],
        "key_challenges": ["Long-tail distributions", "Rare events", "Regulatory pricing constraints", "Adverse selection"],
        "success_metrics": ["Loss ratio", "Combined ratio", "Reserve accuracy", "Underwriting profit"]
    },
    "RegTech & Compliance": {
        "description": "Regulatory technology for anti-money laundering, KYC, and compliance monitoring.",
        "problems": ["AML transaction monitoring", "KYC automation", "Sanctions screening", "Regulatory reporting", "Surveillance"],
        "recommended_approaches": ["Graph Neural Networks", "NLP/LLMs", "Anomaly Detection", "Rule + ML Hybrid"],
        "key_challenges": ["High false positive rates", "Evolving regulations", "Explainability for regulators", "Audit trail requirements"],
        "success_metrics": ["Alert quality (SAR conversion)", "False positive rate", "Processing time", "Coverage"]
    },
    "Customer Experience": {
        "description": "Personalization, engagement, and retention through intelligent customer interactions.",
        "problems": ["Chatbots/Virtual assistants", "Personalization", "Churn prediction", "Next-best-action", "Sentiment analysis"],
        "recommended_approaches": ["LLMs (GPT, Claude)", "Collaborative Filtering", "Classification Models", "NLP"],
        "key_challenges": ["Data privacy concerns", "Real-time personalization", "Omnichannel consistency", "Measuring impact"],
        "success_metrics": ["Containment rate", "NPS/CSAT", "Churn rate", "Engagement metrics", "Conversion rate"]
    }
}

# Resources - Reorganized with FREE and LIBRARIES first
RESOURCES = {
    "Python Libraries (Free)": [
        {"title": "scikit-learn", "description": "Comprehensive ML library - classification, regression, clustering", "url": "https://scikit-learn.org/", "type": "lib", "free": True},
        {"title": "XGBoost", "description": "Gradient boosting framework - state-of-the-art for tabular data", "url": "https://xgboost.ai/", "type": "lib", "free": True},
        {"title": "LightGBM", "description": "Microsoft's fast gradient boosting - handles large datasets", "url": "https://lightgbm.readthedocs.io/", "type": "lib", "free": True},
        {"title": "CatBoost", "description": "Yandex gradient boosting - excellent for categorical features", "url": "https://catboost.ai/", "type": "lib", "free": True},
        {"title": "TensorFlow", "description": "Google's deep learning framework - production ready", "url": "https://www.tensorflow.org/", "type": "lib", "free": True},
        {"title": "PyTorch", "description": "Facebook's deep learning framework - research favorite", "url": "https://pytorch.org/", "type": "lib", "free": True},
        {"title": "statsmodels", "description": "Statistical modeling - ARIMA, GLM, hypothesis testing", "url": "https://www.statsmodels.org/", "type": "lib", "free": True},
        {"title": "Prophet", "description": "Facebook's time series forecasting - handles seasonality", "url": "https://facebook.github.io/prophet/", "type": "lib", "free": True},
        {"title": "SHAP", "description": "Model explainability - Shapley value explanations", "url": "https://shap.readthedocs.io/", "type": "lib", "free": True},
        {"title": "pandas", "description": "Data manipulation and analysis", "url": "https://pandas.pydata.org/", "type": "lib", "free": True}
    ],
    "Free Online Courses": [
        {"title": "Machine Learning Specialization", "description": "Stanford/DeepLearning.AI - Andrew Ng's comprehensive ML course", "url": "https://www.coursera.org/specializations/machine-learning-introduction", "type": "course", "free": True, "note": "Audit free"},
        {"title": "Practical Deep Learning for Coders", "description": "fast.ai - Top-down practical approach to deep learning", "url": "https://course.fast.ai/", "type": "course", "free": True},
        {"title": "CS229: Machine Learning", "description": "Stanford - Full course materials and lecture videos", "url": "https://cs229.stanford.edu/", "type": "course", "free": True},
        {"title": "Deep Learning Specialization", "description": "DeepLearning.AI - Neural networks from basics to advanced", "url": "https://www.coursera.org/specializations/deep-learning", "type": "course", "free": True, "note": "Audit free"},
        {"title": "Hugging Face NLP Course", "description": "Transformers and NLP - hands-on with state-of-the-art models", "url": "https://huggingface.co/learn/nlp-course", "type": "course", "free": True},
        {"title": "Full Stack Deep Learning", "description": "MLOps and production ML - Berkeley course", "url": "https://fullstackdeeplearning.com/", "type": "course", "free": True},
        {"title": "Financial Engineering and Risk Management", "description": "Columbia - Derivatives, portfolio theory, risk", "url": "https://www.coursera.org/specializations/financialengineering", "type": "course", "free": True, "note": "Audit free"}
    ],
    "Documentation & Tutorials (Free)": [
        {"title": "Scikit-learn User Guide", "description": "Comprehensive ML documentation with examples", "url": "https://scikit-learn.org/stable/user_guide.html", "type": "doc", "free": True},
        {"title": "XGBoost Tutorials", "description": "Official tutorials for gradient boosting", "url": "https://xgboost.readthedocs.io/en/stable/tutorials/index.html", "type": "doc", "free": True},
        {"title": "TensorFlow Tutorials", "description": "Deep learning tutorials from basics to advanced", "url": "https://www.tensorflow.org/tutorials", "type": "doc", "free": True},
        {"title": "PyTorch Tutorials", "description": "Learn deep learning with PyTorch", "url": "https://pytorch.org/tutorials/", "type": "doc", "free": True},
        {"title": "Kaggle Learn", "description": "Micro-courses on ML, Python, SQL, and more", "url": "https://www.kaggle.com/learn", "type": "doc", "free": True},
        {"title": "Google ML Crash Course", "description": "Quick introduction to ML with TensorFlow", "url": "https://developers.google.com/machine-learning/crash-course", "type": "doc", "free": True}
    ],
    "Research & Papers (Free)": [
        {"title": "arXiv Quantitative Finance", "description": "Latest research papers in quantitative finance", "url": "https://arxiv.org/list/q-fin/recent", "type": "paper", "free": True},
        {"title": "Papers With Code", "description": "ML papers with implementations - searchable by task", "url": "https://paperswithcode.com/", "type": "paper", "free": True},
        {"title": "SSRN Financial Economics", "description": "Working papers in finance and economics", "url": "https://www.ssrn.com/index.cfm/en/fenetwk/", "type": "paper", "free": True},
        {"title": "Google Scholar", "description": "Search academic papers across all fields", "url": "https://scholar.google.com/", "type": "paper", "free": True},
        {"title": "Distill.pub", "description": "Clear explanations of ML concepts with interactive visualizations", "url": "https://distill.pub/", "type": "paper", "free": True}
    ],
    "Books (Paid)": [
        {"title": "Advances in Financial Machine Learning", "author": "Marcos López de Prado", "description": "Essential for quant finance ML - backtesting, feature engineering", "url": "https://www.wiley.com/en-us/Advances+in+Financial+Machine+Learning-p-9781119482086", "type": "book", "free": False},
        {"title": "Machine Learning for Asset Managers", "author": "Marcos López de Prado", "description": "Portfolio construction with ML techniques", "url": "https://www.cambridge.org/core/books/machine-learning-for-asset-managers/6D9211305EA2E425D33A9F38D0AE3545", "type": "book", "free": False},
        {"title": "Hands-On Machine Learning", "author": "Aurélien Géron", "description": "Practical ML with Scikit-Learn, Keras, TensorFlow", "url": "https://www.oreilly.com/library/view/hands-on-machine-learning/9781098125974/", "type": "book", "free": False},
        {"title": "Python for Finance", "author": "Yves Hilpisch", "description": "Financial analysis and algorithmic trading with Python", "url": "https://www.oreilly.com/library/view/python-for-finance/9781492024323/", "type": "book", "free": False},
        {"title": "Deep Learning", "author": "Goodfellow, Bengio, Courville", "description": "The deep learning textbook - comprehensive foundation", "url": "https://www.deeplearningbook.org/", "type": "book", "free": True, "note": "Free online"}
    ],
    "Paid Courses": [
        {"title": "AI for Trading", "description": "Udacity - Comprehensive trading strategies with AI", "url": "https://www.udacity.com/course/ai-for-trading--nd880", "type": "course", "free": False},
        {"title": "Machine Learning Engineering for Production (MLOps)", "description": "DeepLearning.AI - Production ML systems", "url": "https://www.coursera.org/specializations/machine-learning-engineering-for-production-mlops", "type": "course", "free": False},
        {"title": "CFA Institute Fintech Certificate", "description": "Professional certificate in fintech", "url": "https://www.cfainstitute.org/en/programs/certificates/investment-foundations", "type": "course", "free": False}
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
    
    fig = go.Figure()
    
    colors = {
        "Supervised Learning": "#667eea",
        "Unsupervised Learning": "#38ef7d", 
        "Reinforcement Learning": "#f5576c",
        "Time Series & Specialized": "#F2994A"
    }
    
    for category in df["Category"].unique():
        cat_data = df[df["Category"] == category]
        fig.add_trace(go.Scatter(
            x=cat_data["Complexity"],
            y=cat_data["Interpretability"],
            mode='markers',
            name=category,
            marker=dict(
                size=18,
                color=colors.get(category, "#666"),
                line=dict(width=2, color='white'),
                opacity=0.9
            ),
            text=cat_data["Algorithm"],
            hovertemplate="<b>%{text}</b><br>" +
                          "Complexity: %{x:.1f}<br>" +
                          "Interpretability: %{y:.1f}<extra></extra>"
        ))
    
    # Smart label positioning
    label_positions = []
    for idx, row in df.iterrows():
        x, y = row["Complexity"], row["Interpretability"]
        x_anchor = "center"
        y_offset = 0.22
        x_offset = 0
        
        if x >= 3.5:
            x_anchor = "right"
            x_offset = -0.1
        elif x <= 1.5:
            x_anchor = "left"
            x_offset = 0.1
        
        nearby = [(lp["x"], lp["y"]) for lp in label_positions 
                  if abs(lp["x"] - x) < 0.4 and abs(lp["y"] - y) < 0.5]
        
        if nearby:
            y_offset = 0.32 + 0.15 * len(nearby)
        
        label_positions.append({
            "x": x, "y": y, "text": row["Short_Name"],
            "x_anchor": x_anchor, "x_offset": x_offset, "y_offset": y_offset
        })
    
    annotations = [dict(
        x=lp["x"] + lp["x_offset"], y=lp["y"] + lp["y_offset"],
        text=lp["text"], showarrow=False,
        font=dict(size=10, color="#374151", family="Plus Jakarta Sans"),
        xanchor=lp["x_anchor"], yanchor="bottom"
    ) for lp in label_positions]
    
    # Quadrant backgrounds
    shapes = [
        dict(type="rect", x0=0.5, y0=2.5, x1=2.25, y1=4.5, fillcolor="rgba(56, 239, 125, 0.08)", line_width=0),
        dict(type="rect", x0=2.25, y0=2.5, x1=4.5, y1=4.5, fillcolor="rgba(242, 153, 74, 0.08)", line_width=0),
        dict(type="rect", x0=0.5, y0=0.5, x1=2.25, y1=2.5, fillcolor="rgba(102, 126, 234, 0.08)", line_width=0),
        dict(type="rect", x0=2.25, y0=0.5, x1=4.5, y1=2.5, fillcolor="rgba(245, 87, 108, 0.08)", line_width=0),
    ]
    
    # Quadrant labels
    annotations.extend([
        dict(x=1.35, y=4.35, text="✦ Simple & Explainable", showarrow=False, font=dict(size=12, color="#38ef7d", family="Plus Jakarta Sans", weight="bold")),
        dict(x=3.35, y=4.35, text="✦ Complex but Explainable", showarrow=False, font=dict(size=12, color="#F2994A", family="Plus Jakarta Sans", weight="bold")),
        dict(x=1.35, y=0.65, text="✦ Simple but Opaque", showarrow=False, font=dict(size=12, color="#667eea", family="Plus Jakarta Sans", weight="bold")),
        dict(x=3.35, y=0.65, text="✦ Complex & Opaque", showarrow=False, font=dict(size=12, color="#f5576c", family="Plus Jakarta Sans", weight="bold"))
    ])
    
    fig.update_layout(
        title=dict(text="Algorithm Trade-offs: Complexity vs Interpretability", font=dict(size=20, color="#1a1a2e", family="Plus Jakarta Sans"), x=0.5),
        xaxis=dict(title="Complexity", ticktext=["Low", "Medium", "High", "Very High"], tickvals=[1, 2, 3, 4], range=[0.5, 4.5], gridcolor="#e2e8f0"),
        yaxis=dict(title="Interpretability", ticktext=["Very Low", "Low", "Medium", "High"], tickvals=[1, 2, 3, 4], range=[0.5, 4.5], gridcolor="#e2e8f0"),
        height=680, plot_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5, bgcolor="rgba(255,255,255,0.9)"),
        annotations=annotations, shapes=shapes, margin=dict(t=100, b=50)
    )
    
    return fig

def calculate_recommendation_scores(answers):
    """Calculate detailed scores for each algorithm based on questionnaire answers."""
    scores = {}
    explanations = {}
    
    for category, algos in ML_ALGORITHMS.items():
        for name in algos.keys():
            scores[name] = 0
            explanations[name] = []
    
    # Problem type scoring
    problem_type = answers.get("problem_type", "")
    problem_mappings = {
        "Binary Classification": ["Logistic Regression", "Random Forest", "Gradient Boosting (XGBoost/LightGBM)", "Neural Networks"],
        "Multi-class Classification": ["Random Forest", "Gradient Boosting (XGBoost/LightGBM)", "Neural Networks"],
        "Regression / Continuous Prediction": ["Linear Regression", "Random Forest", "Gradient Boosting (XGBoost/LightGBM)", "Neural Networks"],
        "Clustering / Segmentation": ["K-Means Clustering", "Hierarchical Clustering", "DBSCAN"],
        "Anomaly / Fraud Detection": ["Isolation Forest", "Autoencoders", "DBSCAN", "Gradient Boosting (XGBoost/LightGBM)"],
        "Time Series Forecasting": ["ARIMA/SARIMA", "LSTM/GRU", "Prophet"],
        "Sequential Decision Making": ["Q-Learning", "Deep Q-Networks (DQN)", "Policy Gradient Methods"]
    }
    
    if problem_type in problem_mappings:
        for name in problem_mappings[problem_type]:
            scores[name] = scores.get(name, 0) + 5
            explanations[name] = explanations.get(name, []) + [f"Excellent fit for {problem_type.lower()}"]
    elif problem_type == "Natural Language Processing":
        return [{"type": "Generative AI", "recommendation": "Large Language Models (LLMs)", "reason": "NLP tasks are best handled by transformer-based models like GPT-4, Claude, BERT, or similar LLMs"}]
    elif problem_type == "Content / Report Generation":
        return [{"type": "Generative AI", "recommendation": "Generative AI (LLMs)", "reason": "Content generation requires generative AI capabilities - consider GPT-4, Claude, or similar models"}]
    
    # Data size scoring
    data_size = answers.get("data_size", "")
    if "Very Small" in data_size or "Small" in data_size:
        for name in ["Linear Regression", "Logistic Regression", "Decision Trees"]:
            scores[name] = scores.get(name, 0) + 3
            explanations[name] = explanations.get(name, []) + ["Works well with limited data"]
        for name in ["Neural Networks", "LSTM/GRU", "Deep Q-Networks (DQN)"]:
            scores[name] = scores.get(name, 0) - 3
    elif "Large" in data_size or "Very Large" in data_size:
        for name in ["Neural Networks", "LSTM/GRU", "Gradient Boosting (XGBoost/LightGBM)"]:
            scores[name] = scores.get(name, 0) + 3
            explanations[name] = explanations.get(name, []) + ["Scales excellently with large data"]
    
    # Interpretability scoring
    interpretability = answers.get("interpretability", "")
    if "Critical" in interpretability:
        for name in ["Linear Regression", "Logistic Regression", "Decision Trees"]:
            scores[name] = scores.get(name, 0) + 4
            explanations[name] = explanations.get(name, []) + ["Highly interpretable for regulatory compliance"]
        for name in ["Neural Networks", "Autoencoders", "Deep Q-Networks (DQN)", "LSTM/GRU"]:
            scores[name] = scores.get(name, 0) - 4
    elif "Important" in interpretability:
        for name in ["Random Forest", "Gradient Boosting (XGBoost/LightGBM)"]:
            scores[name] = scores.get(name, 0) + 2
            explanations[name] = explanations.get(name, []) + ["Provides feature importance insights"]
    
    # Latency scoring
    if "Real-time" in answers.get("latency", ""):
        for name in ["Linear Regression", "Logistic Regression", "Decision Trees"]:
            scores[name] = scores.get(name, 0) + 2
            explanations[name] = explanations.get(name, []) + ["Fast inference for real-time use"]
    
    # Team expertise scoring
    expertise = answers.get("expertise", "")
    if "Beginner" in expertise:
        for name in ["Linear Regression", "Logistic Regression", "Decision Trees", "K-Means Clustering"]:
            scores[name] = scores.get(name, 0) + 2
            explanations[name] = explanations.get(name, []) + ["Excellent starting point for beginners"]
        for name in ["Neural Networks", "Deep Q-Networks (DQN)", "Policy Gradient Methods"]:
            scores[name] = scores.get(name, 0) - 2
    
    # Labeled data scoring
    if "No labeled data" in answers.get("labeled_data", ""):
        for name in ["K-Means Clustering", "Hierarchical Clustering", "DBSCAN", "Autoencoders", "Isolation Forest", "Principal Component Analysis (PCA)"]:
            scores[name] = scores.get(name, 0) + 4
            explanations[name] = explanations.get(name, []) + ["Works without labeled data"]
        for name in ["Logistic Regression", "Random Forest", "Gradient Boosting (XGBoost/LightGBM)"]:
            scores[name] = scores.get(name, 0) - 3
    
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
                        "description": details["description"],
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
    # Sidebar
    st.sidebar.markdown("""
    <div style="text-align: center; padding: 1.5rem 0;">
        <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">💹</div>
        <h2 style="color: white; margin: 0; font-size: 1.5rem; font-weight: 700;">FinTech AI/ML Portal</h2>
        <p style="color: rgba(255,255,255,0.7); font-size: 0.85rem; margin-top: 0.5rem;">AI/ML Information and Decision Support for Fintech Ideas</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.sidebar.markdown("---")
    
    page = st.sidebar.radio(
        "Navigation",
        ["🏠 Home", "🧠 AI Types", "⚙️ ML Algorithms", "🎯 Decision Advisor", 
         "💼 Use Cases", "📊 Comparison", "📚 Resources"],
        label_visibility="collapsed"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    <div style="padding: 1rem; background: rgba(255,255,255,0.05); border-radius: 12px;">
        <p style="color: rgba(255,255,255,0.8); font-size: 0.8rem; margin: 0;">
            <strong style="color: #667eea;">Pro Tip:</strong> Start with the 
            <strong>Decision Advisor</strong> to get personalized algorithm 
            recommendations for your project.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Clean page name
    page_name = page.split(" ", 1)[1] if " " in page else page
    
    # ========================================================================
    # HOME PAGE
    # ========================================================================
    if "Home" in page:
        # Hero Section
        st.markdown("""
        <div class="hero-container">
            <h1 class="hero-title">Fintech AI/ML Portal</h1>
            <p class="hero-subtitle">Your intelligent guide to selecting the right AI and machine learning approach for financial technology innovation</p>
            <div class="stats-container">
                <div class="stat-card">
                    <div class="stat-number">20+</div>
                    <div class="stat-label">ML Algorithms</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">6</div>
                    <div class="stat-label">Fintech Domains</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">10</div>
                    <div class="stat-label">Question Advisor</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">50+</div>
                    <div class="stat-label">Free Resources</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Feature Cards
        st.markdown('<div class="section-header"><span class="section-icon">✦</span>What You\'ll Learn</div>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="feature-card">
                <div class="feature-icon purple">🧠</div>
                <div class="feature-title">AI Fundamentals</div>
                <div class="feature-desc">Understand the difference between Generative AI and Analytical AI, their capabilities, and when to use each approach in financial services applications.</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown("""
            <div class="feature-card">
                <div class="feature-icon blue">⚙️</div>
                <div class="feature-title">Algorithm Mastery</div>
                <div class="feature-desc">Deep dive into 20+ machine learning algorithms with detailed explanations, use cases, pros/cons, and practical implementation guidance for fintech.</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col3:
            st.markdown("""
            <div class="feature-card">
                <div class="feature-icon green">🎯</div>
                <div class="feature-title">Personalized Guidance</div>
                <div class="feature-desc">Use our intelligent Decision Advisor questionnaire to receive tailored algorithm recommendations based on your specific project requirements.</div>
            </div>
            """, unsafe_allow_html=True)
        
        col4, col5, col6 = st.columns(3)
        
        with col4:
            st.markdown("""
            <div class="feature-card">
                <div class="feature-icon orange">💼</div>
                <div class="feature-title">Real-World Cases</div>
                <div class="feature-desc">Explore practical applications across payments, lending, trading, insurance, compliance, and customer experience with detailed implementation insights.</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col5:
            st.markdown("""
            <div class="feature-card">
                <div class="feature-icon pink">📊</div>
                <div class="feature-title">Visual Comparison</div>
                <div class="feature-desc">Interactive visualizations help you understand trade-offs between algorithm complexity and interpretability for regulatory compliance.</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col6:
            st.markdown("""
            <div class="feature-card">
                <div class="feature-icon purple">📚</div>
                <div class="feature-title">Curated Resources</div>
                <div class="feature-desc">Access a carefully selected collection of free libraries, courses, documentation, and research papers to continue your learning journey.</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Getting Started Steps
        st.markdown('<div class="section-header"><span class="section-icon">🚀</span>Getting Started</div>', unsafe_allow_html=True)
        
        steps = [
            ("Understand the Landscape", "Start with AI Types to learn the fundamental difference between Generative and Analytical AI approaches"),
            ("Explore Algorithms", "Browse the ML Algorithms section for detailed explanations of each technique and its fintech applications"),
            ("Get Recommendations", "Use the Decision Advisor to answer 10 questions and receive personalized algorithm recommendations"),
            ("Study Use Cases", "Examine real-world fintech applications to understand how algorithms solve specific business problems"),
            ("Access Resources", "Leverage our curated collection of free libraries, courses, and documentation to build your skills")
        ]
        
        for i, (title, desc) in enumerate(steps, 1):
            st.markdown(f"""
            <div class="step-container">
                <div class="step-number">{i}</div>
                <div class="step-content">
                    <h4>{title}</h4>
                    <p>{desc}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Key Considerations
        st.markdown('<div class="section-header"><span class="section-icon">⚠️</span>Key Considerations for Fintech AI/ML</div>', unsafe_allow_html=True)
        
        considerations = pd.DataFrame({
            "Factor": ["Regulatory Compliance", "Model Interpretability", "Data Privacy", "Model Governance", "Scalability", "Fairness & Bias"],
            "Why It Matters": [
                "GDPR, CCPA, fair lending laws (ECOA) restrict certain approaches",
                "Many decisions require explainable models for audit trails",
                "Financial data requires strict controls and anonymization",
                "Models need validation, monitoring, and retraining protocols",
                "Solutions must handle varying loads and growth",
                "Models must not discriminate against protected classes"
            ],
            "Impact on Algorithm Choice": [
                "May favor interpretable models over black-box neural networks",
                "Consider SHAP/LIME for complex models, or use inherently interpretable methods",
                "Explore federated learning, differential privacy, synthetic data",
                "Build MLOps pipelines from the start; plan for model decay",
                "Design for cloud-native deployment; consider inference latency",
                "Audit models for disparate impact; use fairness-aware algorithms"
            ]
        })
        
        st.dataframe(considerations, use_container_width=True, hide_index=True)
    
    # ========================================================================
    # AI TYPES PAGE
    # ========================================================================
    elif "AI Types" in page:
        st.markdown("""
        <div class="hero-container" style="padding: 2rem;">
            <h1 class="hero-title" style="font-size: 2.5rem;">Types of Artificial Intelligence</h1>
            <p class="hero-subtitle">Understanding Generative vs Analytical AI in Financial Services</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Comparison Overview
        st.markdown('<div class="section-header"><span class="section-icon">⚖️</span>Side-by-Side Comparison</div>', unsafe_allow_html=True)
        
        comparison_df = pd.DataFrame({
            "Aspect": ["Primary Function", "Output Type", "Key Technologies", "Data Requirements", 
                      "Interpretability", "Regulatory Fit", "Typical Fintech Uses", "Computational Cost"],
            "Generative AI": [
                "Creates new content",
                "Text, images, code, synthetic data",
                "LLMs, GANs, VAEs, Diffusion Models",
                "Massive unstructured datasets (billions of tokens)",
                "Very Low (black box)",
                "Challenging - outputs need verification",
                "Chatbots, report generation, synthetic data",
                "Very High (GPU clusters for training)"
            ],
            "Analytical AI": [
                "Analyzes and predicts",
                "Predictions, classifications, scores, clusters",
                "ML algorithms, statistical models, deep learning",
                "Structured, labeled data (thousands to millions)",
                "Low to High (varies by algorithm)",
                "Well-established frameworks exist",
                "Credit scoring, fraud detection, trading",
                "Low to High (varies by algorithm)"
            ]
        })
        
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)
        
        # Detailed Tabs
        tab1, tab2 = st.tabs(["🤖 Generative AI", "📊 Analytical AI"])
        
        with tab1:
            gen_ai = AI_TYPES["Generative AI"]
            
            st.markdown(f"""
            <div class="info-box">
                <strong>Definition:</strong> {gen_ai['description']}
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("#### How Generative AI Works")
            st.markdown(gen_ai['detailed_explanation'])
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Key Characteristics")
                for char in gen_ai["key_characteristics"]:
                    st.markdown(f"• {char}")
                
                st.markdown("#### Core Technologies")
                for tech, examples in gen_ai["technologies"]:
                    st.markdown(f"**{tech}:** {examples}")
            
            with col2:
                st.markdown("#### Fintech Applications")
                for app, desc in gen_ai["fintech_applications"]:
                    st.markdown(f"**{app}**")
                    st.markdown(f"_{desc}_")
            
            st.markdown("---")
            
            col3, col4 = st.columns(2)
            
            with col3:
                st.markdown("#### Advantages")
                for pro in gen_ai["pros"]:
                    st.markdown(f"""<div class="success-box" style="padding: 0.75rem; margin: 0.25rem 0;">✓ {pro}</div>""", unsafe_allow_html=True)
            
            with col4:
                st.markdown("#### Challenges")
                for con in gen_ai["cons"]:
                    st.markdown(f"""<div class="warning-box" style="padding: 0.75rem; margin: 0.25rem 0;">⚠ {con}</div>""", unsafe_allow_html=True)
            
            st.markdown("#### Regulatory Considerations")
            st.markdown(f"""<div class="danger-box">{gen_ai['regulatory_considerations']}</div>""", unsafe_allow_html=True)
        
        with tab2:
            ana_ai = AI_TYPES["Analytical AI"]
            
            st.markdown(f"""
            <div class="info-box">
                <strong>Definition:</strong> {ana_ai['description']}
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("#### How Analytical AI Works")
            st.markdown(ana_ai['detailed_explanation'])
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Key Characteristics")
                for char in ana_ai["key_characteristics"]:
                    st.markdown(f"• {char}")
                
                st.markdown("#### Core Technologies")
                for tech, examples in ana_ai["technologies"]:
                    st.markdown(f"**{tech}:** {examples}")
            
            with col2:
                st.markdown("#### Fintech Applications")
                for app, desc in ana_ai["fintech_applications"]:
                    st.markdown(f"**{app}**")
                    st.markdown(f"_{desc}_")
            
            st.markdown("---")
            
            col3, col4 = st.columns(2)
            
            with col3:
                st.markdown("#### Advantages")
                for pro in ana_ai["pros"]:
                    st.markdown(f"""<div class="success-box" style="padding: 0.75rem; margin: 0.25rem 0;">✓ {pro}</div>""", unsafe_allow_html=True)
            
            with col4:
                st.markdown("#### Challenges")
                for con in ana_ai["cons"]:
                    st.markdown(f"""<div class="warning-box" style="padding: 0.75rem; margin: 0.25rem 0;">⚠ {con}</div>""", unsafe_allow_html=True)
            
            st.markdown("#### Regulatory Considerations")
            st.markdown(f"""<div class="info-box">{ana_ai['regulatory_considerations']}</div>""", unsafe_allow_html=True)
    
    # ========================================================================
    # ML ALGORITHMS PAGE
    # ========================================================================
    elif "ML Algorithms" in page:
        st.markdown("""
        <div class="hero-container" style="padding: 2rem;">
            <h1 class="hero-title" style="font-size: 2.5rem;">Machine Learning Algorithms</h1>
            <p class="hero-subtitle">Comprehensive guide to 20+ algorithms for fintech applications</p>
        </div>
        """, unsafe_allow_html=True)
        
        category = st.selectbox(
            "Select Algorithm Category:",
            list(ML_ALGORITHMS.keys()),
            format_func=lambda x: f"📂 {x}"
        )
        
        st.markdown(f'<div class="section-header"><span class="section-icon">⚙️</span>{category}</div>', unsafe_allow_html=True)
        
        algorithms = ML_ALGORITHMS[category]
        
        for algo_name, algo_details in algorithms.items():
            with st.expander(f"🔹 {algo_name} — {algo_details['type']}", expanded=False):
                st.markdown(f"""
                <div class="info-box">
                    <strong>Overview:</strong> {algo_details['description']}
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("#### Detailed Explanation")
                st.markdown(algo_details['detailed_explanation'])
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown("#### Fintech Use Cases")
                    for use_case in algo_details['fintech_use_cases']:
                        st.markdown(f"• {use_case}")
                    
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.markdown("#### Advantages")
                        for pro in algo_details['pros']:
                            st.markdown(f"✓ {pro}")
                    with col_b:
                        st.markdown("#### Limitations")
                        for con in algo_details['cons']:
                            st.markdown(f"⚠ {con}")
                
                with col2:
                    st.markdown("#### Quick Reference")
                    st.markdown(f"""
                    <div class="algo-card">
                        <p><strong>Complexity:</strong><br>{algo_details['complexity']}</p>
                        <p><strong>Interpretability:</strong><br>{algo_details['interpretability']}</p>
                        <p><strong>Data Requirements:</strong><br>{algo_details['data_requirements']}</p>
                    </div>
                    """, unsafe_allow_html=True)
    
    # ========================================================================
    # DECISION ADVISOR PAGE
    # ========================================================================
    elif "Decision Advisor" in page:
        st.markdown("""
        <div class="hero-container" style="padding: 2rem;">
            <h1 class="hero-title" style="font-size: 2.5rem;">AI/ML Decision Advisor</h1>
            <p class="hero-subtitle">Answer 10 questions to receive personalized algorithm recommendations</p>
        </div>
        """, unsafe_allow_html=True)
        
        if 'advisor_submitted' not in st.session_state:
            st.session_state.advisor_submitted = False
        
        with st.form("advisor_form"):
            st.markdown('<div class="section-header"><span class="section-icon">📋</span>Project Assessment</div>', unsafe_allow_html=True)
            
            # Question 1
            st.markdown("#### 1. What type of problem are you trying to solve?")
            problem_type = st.selectbox(
                "Problem type:",
                ["Binary Classification", "Multi-class Classification", "Regression / Continuous Prediction",
                 "Clustering / Segmentation", "Anomaly / Fraud Detection", "Time Series Forecasting",
                 "Sequential Decision Making", "Natural Language Processing", "Content / Report Generation"],
                label_visibility="collapsed"
            )
            
            st.markdown("---")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 2. How much data do you have?")
                data_size = st.selectbox("Data size:", ["Very Small (<500 samples)", "Small (500-5,000)", "Medium (5,000-50,000)", "Large (50,000-500,000)", "Very Large (>500,000)"], label_visibility="collapsed")
                
                st.markdown("#### 3. Do you have labeled data?")
                labeled_data = st.selectbox("Labels:", ["Yes - Fully labeled", "Partial labels", "Limited labels", "No labeled data"], label_visibility="collapsed")
                
                st.markdown("#### 4. Data quality?")
                data_quality = st.selectbox("Quality:", ["Excellent - Clean, complete", "Good - Minor issues", "Fair - Some missing values", "Poor - Lots of issues"], label_visibility="collapsed")
                
                st.markdown("#### 5. Feature engineering capacity?")
                feature_eng = st.selectbox("Feature eng:", ["Extensive - Will invest heavily", "Moderate - Some possible", "Minimal - Prefer automatic"], label_visibility="collapsed")
            
            with col2:
                st.markdown("#### 6. How important is interpretability?")
                interpretability = st.selectbox("Interpretability:", ["Critical - Regulatory requirement", "Important - Need key drivers", "Moderate - Some explanation helpful", "Low - Accuracy is priority"], label_visibility="collapsed")
                
                st.markdown("#### 7. Latency requirements?")
                latency = st.selectbox("Latency:", ["Real-time (<10ms)", "Near real-time (<1s)", "Fast (<10s)", "Batch processing OK"], label_visibility="collapsed")
                
                st.markdown("#### 8. Team ML expertise?")
                expertise = st.selectbox("Expertise:", ["Beginner - New to ML", "Intermediate - Some experience", "Advanced - Strong background", "Expert - Deep experience"], label_visibility="collapsed")
                
                st.markdown("#### 9. Fintech domain?")
                domain = st.selectbox("Domain:", list(FINTECH_DOMAINS.keys()), label_visibility="collapsed")
            
            st.markdown("---")
            st.markdown("#### 10. Describe your project (optional)")
            project_desc = st.text_area("Description:", placeholder="E.g., 'Build a fraud detection system for real-time transaction monitoring...'", height=100, label_visibility="collapsed")
            
            submitted = st.form_submit_button("🎯 Generate Recommendations", use_container_width=True)
        
        if submitted:
            st.session_state.advisor_submitted = True
            st.session_state.answers = {
                "problem_type": problem_type, "data_size": data_size, "labeled_data": labeled_data,
                "data_quality": data_quality, "feature_engineering": feature_eng, "interpretability": interpretability,
                "latency": latency, "expertise": expertise, "domain": domain, "project_description": project_desc
            }
        
        if st.session_state.advisor_submitted:
            answers = st.session_state.answers
            st.markdown("---")
            st.markdown('<div class="section-header"><span class="section-icon">🎯</span>Your Personalized Recommendations</div>', unsafe_allow_html=True)
            
            recommendations = calculate_recommendation_scores(answers)
            
            if recommendations:
                gen_ai_recs = [r for r in recommendations if r.get("type") == "Generative AI"]
                ml_recs = [r for r in recommendations if r.get("algorithm")]
                
                if gen_ai_recs:
                    for rec in gen_ai_recs:
                        st.markdown(f"""
                        <div class="success-box">
                            <h3 style="margin-top: 0;">🤖 {rec['recommendation']}</h3>
                            <p>{rec['reason']}</p>
                            <p><strong>Recommended models:</strong> GPT-4, Claude, Llama 2, BERT, or fine-tuned domain-specific models</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                if ml_recs:
                    for i, rec in enumerate(ml_recs[:3], 1):
                        is_top = i == 1
                        st.markdown(f"""
                        <div class="rec-card {'top-pick' if is_top else ''}">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                                <h3 style="margin: 0;">{'🥇' if i==1 else '🥈' if i==2 else '🥉'} {rec['algorithm']}</h3>
                                <span class="rec-score">Score: {rec['score']}</span>
                            </div>
                            <p><strong>Category:</strong> {rec['category']} | <strong>Type:</strong> {rec['type']}</p>
                            <p>{rec['description'][:200]}...</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        if rec.get('explanations'):
                            with st.expander(f"Why {rec['algorithm']}?"):
                                for exp in rec['explanations']:
                                    st.markdown(f"✓ {exp}")
                                st.markdown(f"**Complexity:** {rec['complexity']} | **Interpretability:** {rec['interpretability']}")
    
    # ========================================================================
    # USE CASES PAGE
    # ========================================================================
    elif "Use Cases" in page:
        st.markdown("""
        <div class="hero-container" style="padding: 2rem;">
            <h1 class="hero-title" style="font-size: 2.5rem;">Fintech Use Cases</h1>
            <p class="hero-subtitle">Real-world AI/ML applications across financial services</p>
        </div>
        """, unsafe_allow_html=True)
        
        domain = st.selectbox("Select Domain:", list(FINTECH_DOMAINS.keys()), format_func=lambda x: f"💼 {x}")
        
        domain_info = FINTECH_DOMAINS[domain]
        
        st.markdown(f"""
        <div class="info-box">
            <strong>{domain}:</strong> {domain_info['description']}
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Key Problems")
            for prob in domain_info['problems']:
                st.markdown(f"""<div class="algo-card" style="padding: 1rem;">🔹 {prob}</div>""", unsafe_allow_html=True)
            
            st.markdown("#### Key Challenges")
            for challenge in domain_info['key_challenges']:
                st.markdown(f"• {challenge}")
        
        with col2:
            st.markdown("#### Recommended Approaches")
            for approach in domain_info['recommended_approaches']:
                st.markdown(f"""<div class="success-box" style="padding: 1rem;">✦ {approach}</div>""", unsafe_allow_html=True)
            
            st.markdown("#### Success Metrics")
            for metric in domain_info['success_metrics']:
                st.markdown(f"• {metric}")
    
    # ========================================================================
    # COMPARISON PAGE
    # ========================================================================
    elif "Comparison" in page:
        st.markdown("""
        <div class="hero-container" style="padding: 2rem;">
            <h1 class="hero-title" style="font-size: 2.5rem;">Algorithm Comparison</h1>
            <p class="hero-subtitle">Visual comparison across complexity and interpretability dimensions</p>
        </div>
        """, unsafe_allow_html=True)
        
        fig = create_tradeoff_chart(ML_ALGORITHMS)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        <div class="info-box">
            <strong>Reading the Chart:</strong> Algorithms in the <span style="color: #38ef7d; font-weight: bold;">top-left (green)</span> 
            are ideal for regulated applications requiring explainability. <span style="color: #f5576c; font-weight: bold;">Bottom-right (red)</span> 
            algorithms offer maximum performance but require explainability tools like SHAP for regulatory compliance.
        </div>
        """, unsafe_allow_html=True)
    
    # ========================================================================
    # RESOURCES PAGE
    # ========================================================================
    elif "Resources" in page:
        st.markdown("""
        <div class="hero-container" style="padding: 2rem;">
            <h1 class="hero-title" style="font-size: 2.5rem;">Learning Resources</h1>
            <p class="hero-subtitle">Curated collection of libraries, courses, and documentation — free resources first!</p>
        </div>
        """, unsafe_allow_html=True)
        
        for category, resources in RESOURCES.items():
            st.markdown(f'<div class="section-header"><span class="section-icon">{"📦" if "Lib" in category else "🎓" if "Course" in category else "📖" if "Doc" in category else "📄" if "Paper" in category else "📚"}</span>{category}</div>', unsafe_allow_html=True)
            
            for res in resources:
                icon_class = res.get("type", "lib")
                badge = ""
                if res.get("free", False):
                    note = res.get("note", "Free")
                    badge = f'<span class="resource-badge badge-free">{note}</span>'
                else:
                    badge = '<span class="resource-badge badge-paid">Paid</span>'
                
                author = f" by {res['author']}" if res.get('author') else ""
                
                st.markdown(f"""
                <a href="{res['url']}" target="_blank" class="resource-card">
                    <div class="resource-icon {icon_class}">{"📦" if icon_class == "lib" else "🎓" if icon_class == "course" else "📖" if icon_class == "doc" else "📚" if icon_class == "book" else "📄"}</div>
                    <div class="resource-info">
                        <div class="resource-title">{res['title']}{author}</div>
                        <div class="resource-meta">{res['description']}</div>
                    </div>
                    {badge}
                </a>
                """, unsafe_allow_html=True)
    # ========================================================================
    # FOOTER (appears on all pages)
    # ========================================================================
    st.markdown("---")
    st.markdown("""
    <div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); border-radius: 20px; padding: 2.5rem 2rem; margin-top: 3rem;">
       <div style="text-align: center; margin-bottom: 1.5rem;">
            <h3 style="color: white; margin: 0; font-size: 1.3rem; font-weight: 700;">Fintech AI/ML Portal</h3>
            <p style="color: rgba(255,255,255,0.9); font-size: 0.9rem; margin-top: 0.5rem;">
                Educational Tool Students
            </p>
       </div>
    
       <div style="text-align: center; padding: 1.5rem; background: rgba(255,255,255,0.05); border-radius: 12px; margin-bottom: 1rem;">
        <p style="color: #667eea; font-size: 1rem; margin-bottom: 0.5rem; font-weight: 600;">
            👨‍🏫 Developed by <a href="https://www.ntu.ac.uk/staff-profiles/business/vangelis-tsiligkiris" target="_blank" style="color: #4facfe; text-decoration: none; font-weight: 700; border-bottom: 2px solid #4facfe;">Professor Vangelis Tsiligkiris</a>
        </p>
        <p style="color: rgba(255,255,255,0.95); font-size: 0.85rem; margin: 0;">
            Designed for students in Fintech, Financial Engineering, and Data Science programs
        </p>
        </div>
        
       <div style="text-align: center;">
            <p style="color: rgba(255,255,255,0.85); font-size: 0.85rem; margin-bottom: 0.5rem;">
                © 2025 Professor Vangelis Tsiligkiris | Licensed under the 
                <a href="https://opensource.org/licenses/MIT" target="_blank" 
                style="color: #4facfe; text-decoration: none; font-weight: 600;">MIT License</a>
            </p>
            <p style="color: rgba(255,255,255,0.75); font-size: 0.75rem; margin: 0; font-style: italic;">
                This educational tool is provided as-is for learning purposes. Always validate with domain experts for production use.
            </p>
       </div>
    
       <div style="text-align: center; margin-top: 1.5rem; padding-top: 1rem; border-top: 1px solid rgba(255,255,255,0.1);">
            <p style="color: rgba(255,255,255,0.6); font-size: 0.75rem; margin: 0;">
                Built with Streamlit, Plotly, and Python | Version 1.0 | 2025
            </p>
       </div>
    </div>
    """, unsafe_allow_html=True)  


if __name__ == "__main__":
    main()