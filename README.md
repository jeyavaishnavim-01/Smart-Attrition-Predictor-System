# Smart-Attrition-Predictor-System #
End-to-end ML web app predicting employee attrition using scikit-learn, TensorFlow &amp; PyTorch with Flask + AI-powered insights

**Project Overview**

Employee attrition is one of the most costly challenges for any organisation. Replacing a single employee can cost 50–200% of their annual salary in recruitment, onboarding, and lost productivity. The Smart Attrition Predictor System solves this by giving HR teams a data-driven early warning system. Simply enter an employee's profile — department, salary, overtime, satisfaction scores — and the system instantly predicts their attrition risk using three independently trained ML models that vote together for the most accurate consensus result. On top of the ML predictions, the system connects to a Large Language Model (LLM) via the Groq API to generate plain English recommendations, specific risk factors and concrete retention actions tailored to that employee's profile. Every prediction is automatically saved to a local SQLite database, giving HR teams a full audit trail of all assessments over time.

**Features**

1. Triple Model Prediction — scikit-learn ensemble, TensorFlow ANN, and PyTorch neural network run simultaneously and return a consensus probability.
2. Live Dashboard — real-time stats showing total predictions, attrition rate, and model accuracy cards.
3. AI Powered HR Insights — Groq LLM (LLaMA 3.3 70B) generates risk factors and retention actions for each employee.
4. Model Comparison — side-by-side accuracy, ROC-AUC, classification reports, and feature importance charts.
5. Persistent Storage — SQLite database auto-created on first run, no setup needed and every result saved to SQLite with live search and filter.
6. Modern Dark UI — responsive dashboard with animated progress bars, toast notifications, and loading spinners.

**Tech Stack**
1. Language - Python 3.11 -Core programming language for all backend, ML, and data logic.
2. Data & Processing - Libraries : pandas - 2.0.3 - Data loading, manipulation, and preprocessing of dataset, numpy - 1.24.3 - Numerical operations, array handling, and mathematical computations, scikit-learn - 1.3.0 - LabelEncoder, StandardScaler, train/test split, classification metrics.

**Machine Learning Models**

1. RandomForestClassifier : scikit-learn - Ensemble of 200 decision trees — learns feature patterns via bagging.
2. GradientBoostingClassifier : scikit-learn - Boosted trees — focuses on hard-to-predict cases iteratively.
3. LogisticRegressions : cikit-learn - Linear model — captures straightforward attrition signals.
4. VotingClassifier: scikit-learn - Combines the 3 above with soft voting (weights 3:2:1) for consensus.
5. TensorFlow ANN : TensorFlow / Keras - 4-layer deep neural network (256→128→64→32→1) with BatchNorm + Dropout.
6. PyTorch Attrition : NetPyTorch - Custom neural network with GELU activations, Kaiming initialisation, CosineAnnealingLR.

**Deep Learning Frameworks**

1. TensorFlow - 2.15.0 - Building and training the Artificial Neural Network (ANN).
2. Keras - 2.15.0 - High-level API on top of TensorFlow for model definition and training callbacks.
3. PyTorch - 2.1.0 - Building and training the custom AttritionNet from scratch.

**Web Framework & Backend**

1. Flask - 3.0.0 - Python web framework — serves all routes, handles form POST requests.
2. flask-cors - 6.0.2 - Cross-Origin Resource Sharing — allows browser to call Flask API endpoints.
3. Jinja - 23.1.6 - HTML templating engine built into Flask — renders dynamic pages.
4. Werkzeug - 3.1.6 - WSGI utilities used internally by Flaskpython.
5. dotenv - 1.2.2 - Loads environment variables from .env file (API keys).

**Database**

1. SQLite (built-in)Lightweight local database — stores every prediction with timestamp, probabilities, and risk level.
2. sqlite3 (built-in)Python standard library module for SQLite operations.

**AI & LLM Integration**

Groq APIgroq - 1.1.1 - LLaMA 3.3 70B - VersatileGenerates HR risk factors and retention recommendations in plain English.

**ML Utilities & Serialisation**

1. pickle (built-in)Saves and loads trained ML models, scalers, and encoders to disk.
2. WeightedRandomSamplerPyTorch — handles class imbalance by oversampling minority class.
3. EarlyStoppingKeras callback — stops training when validation AUC stops improving.
4. ReduceLROnPlateauKeras callback — reduces learning rate when loss plateaus.
5. CosineAnnealingLRPyTorch scheduler — smoothly reduces learning rate over epochs.

**Frontend**

1. HTML5 - Page structure and prediction form.
2. CSS3 - Dark theme UI, animations, responsive grid layout.
3. Vanilla JavaScript - Fetch API calls to /api/ai, live table search, toast notifications, loading spinners.
4. Google Fonts — OutfitClean modern UI font for all textGoogle Fonts — JetBrains MonoMonospace font for numbers, probabilities, and code.

**Model Performance**

1. scikit-learn VotingClassifier~92.7%
2. TensorFlow ANN~88.1%
3. PyTorch AttritionNet~88.8%

Models are trained on a synthetic IBM-style HR dataset of 1,500 employee records with engineered attrition scores based on real-world HR research.
