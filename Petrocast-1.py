# Import necessary libraries
import pandas as pd  # Data manipulation and analysis
import numpy as np   # Numerical computations
import streamlit as st  # Web application framework
import shap  # Model explanation library
import matplotlib.pyplot as plt  # Plotting library
import seaborn as sns  # Enhanced visualization

# Scikit-learn components
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.linear_model import BayesianRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline  # ML workflow management
from sklearn.preprocessing import StandardScaler  # Feature normalization

# System and utility libraries
import logging  # Error logging
import traceback  # Stack trace handling
import joblib  # Model serialization
from io import BytesIO  # In-memory file handling

# Configure logging system
logging.basicConfig(
    level=logging.INFO,  # Set logging level to INFO messages
    format='%(asctime)s - %(levelname)s - %(message)s',  # Log format
    handlers=[logging.FileHandler("app.log"), logging.StreamHandler()]  # Output destinations
)

# Configure matplotlib to use non-interactive backend for server environments
plt.switch_backend('Agg')

# Set seaborn visualization style
sns.set_theme(style='whitegrid')

# Initialize SHAP's JavaScript visualization library
shap.initjs()

# Define model configurations with hyperparameters and pipelines
MODELS = {
    "XGBoost": {
        "class": XGBRegressor,  # Gradient boosting framework
        "params": {  # Hyperparameter grid
            'n_estimators': [300],  # Number of sequential trees
            'max_depth': [4],  # Maximum tree depth
            'learning_rate': [0.05],  # Step size shrinkage
            'reg_alpha': [0.1],  # L1 regularization
            'reg_lambda': [0.1],  # L2 regularization
            'subsample': [0.8]  # Subsample ratio
        },
        "pipeline": Pipeline([('model', XGBRegressor())])  # Model pipeline
    },
    "Random Forest": {
        "class": RandomForestRegressor,  # Ensemble method
        "params": {
            'n_estimators': [200],  # Number of trees
            'max_depth': [10],  # Tree depth limit
            'min_samples_split': [2],  # Minimum samples to split
            'max_features': ['sqrt']  # Features considered per split
        },
        "pipeline": Pipeline([('model', RandomForestRegressor())])
    },
    "Bayesian Ridge": {
        "class": BayesianRidge,  # Bayesian regression
        "params": {'alpha_1': [1e-6]},  # Precision parameter
        "pipeline": Pipeline([('model', BayesianRidge())])
    },
    "SVM": {
        "class": SVR,  # Support Vector Regression
        "params": {  # SVM parameters
            'C': [10],  # Regularization strength
            'kernel': ['rbf'],  # Radial basis function kernel
            'epsilon': [0.1]  # Margin of tolerance
        },
        "pipeline": Pipeline([
            ('scaler', StandardScaler()),  # Feature standardization
            ('model', SVR())  # SVR model
        ])
    },
    "Decision Tree": {
        "class": DecisionTreeRegressor,  # Decision tree regressor
        "params": {  # Tree parameters
            'max_depth': [5, 10, None],  # Depth constraints
            'min_samples_split': [2, 5]  # Split constraints
        },
        "pipeline": Pipeline([('model', DecisionTreeRegressor())])
    }
}

def check_leakage(X, y, threshold=0.95):
    """Identify features with suspiciously high correlation to target"""
    # Combine features and target for correlation analysis
    corr_matrix = X.join(y).corr()
    
    # Get absolute correlations with target sorted descendingly
    target_corr = corr_matrix['Oil_volume'].abs().sort_values(ascending=False)
    
    # Filter features exceeding correlation threshold
    leaked_features = target_corr[target_corr > threshold].index.tolist()
    
    # Remove target variable if present
    if 'Oil_volume' in leaked_features:
        leaked_features.remove('Oil_volume')
    
    return leaked_features

def validate_data(df):
    """Ensure dataset meets structural and quality requirements"""
    # Define mandatory columns
    required_columns = ['DEPTH_MD', 'Reservoir_pressure',
                       'Working_hours', 'Oil_volume', 'Date', 'WELL']
    
    # Check for missing required columns
    missing_required = [col for col in required_columns if col not in df.columns]
    if missing_required:
        raise ValueError(f"Missing required columns: {', '.join(missing_required)}")
    
    # Validate dataset is not empty
    if df.empty:
        raise ValueError("Empty dataset")
    
    # Check target variable integrity
    if df['Oil_volume'].isnull().any():
        raise ValueError("Null values in target column")
    
    # Handle negative oil volume values
    if (df['Oil_volume'] < 0).any():
        st.error("Negative Oil_volume values detected - correcting to absolute values")
        df['Oil_volume'] = df['Oil_volume'].abs()
    
    return df

def cap_outliers(df, cols, lower_percentile=0.01, upper_percentile=0.99):
    """Winsorize numerical columns to cap extreme values"""
    for col in cols:
        # Only process numerical columns
        if df[col].dtype in ['int64', 'float64']:
            # Calculate percentile thresholds
            lower = df[col].quantile(lower_percentile)
            upper = df[col].quantile(upper_percentile)
            # Clip values to calculated bounds
            df[col] = np.clip(df[col], lower, upper)
    return df

def train_model(df, model_name):
    """Complete model training workflow"""
    try:
        # Validate dataset structure and content
        df = validate_data(df)
        
        # Filter wells with insufficient data points (minimum 6 records)
        df = df.groupby('WELL').filter(lambda x: len(x) > 5)
        
        # Identify numerical columns for outlier handling
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        df = cap_outliers(df, numeric_cols)
        
        # Base feature configuration
        features = ['DEPTH_MD', 'Reservoir_pressure', 'Working_hours']
        
        # Check for potential data leakage
        leaked_features = check_leakage(df[features], df['Oil_volume'])
        
        if leaked_features:
            st.warning(f"High leakage detected in: {', '.join(leaked_features)}")
            features = [f for f in features if f not in leaked_features]
        
        # Time-based data splitting (80-20 split)
        df = df.sort_values('Date')
        total_len = len(df)
        train_size = int(total_len * 0.8)
        train_df = df.iloc[:train_size]
        test_df = df.iloc[train_size:]
        
        # Prepare training and testing data
        X_train, X_test = train_df[features], test_df[features]
        y_train, y_test = train_df['Oil_volume'], test_df['Oil_volume']
        
        # Retrieve model configuration
        model_config = MODELS[model_name]
        model_pipe = model_config['pipeline']
        
        # Format parameter grid for GridSearchCV
        param_grid = {f'model__{key}': value for key, value in model_config['params'].items()}
        
        # Configure time-series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)  # 5-fold time-series split
        
        # Set up grid search with cross-validation
        grid_search = GridSearchCV(
            estimator=model_pipe,
            param_grid=param_grid,
            cv=tscv,
            scoring='neg_mean_squared_error',  # Optimization metric
            verbose=0  # Suppress output
        )
        
        # Execute model training
        grid_search.fit(X_train, y_train)
        
        # Extract best performing model
        best_model = grid_search.best_estimator_
        
        # Calculate cross-validated R² score
        cv_r2 = np.mean(cross_val_score(
            best_model, X_train, y_train, 
            cv=tscv, scoring='r2'
        ))
        
        # Calculate test set R² score
        test_r2 = r2_score(y_test, best_model.predict(X_test))
        
        # Calculate test set mean squared error
        test_mse = mean_squared_error(y_test, best_model.predict(X_test))
        
        # Calculate cross-validation mean squared error
        cv_mse = -np.mean(cross_val_score(
            best_model, X_train, y_train, 
            cv=tscv, scoring='neg_mean_squared_error'
        ))
        
        # Return training artifacts and metrics
        return {
            'model': best_model,  # Trained model object
            'features': features,  # Final feature list
            'cv_r2': cv_r2,  # Cross-validation R²
            'test_r2': test_r2,  # Test set R²
            'test_data': (X_test, y_test),  # Holdout dataset
            'cv_mse': cv_mse,  # Cross-validation MSE
            'test_mse': test_mse,  # Test set MSE
            'feature_stats': {  # Feature statistics for validation
                col: {
                    'min': float(df[col].min()),  # Minimum value
                    'max': float(df[col].max())  # Maximum value
                } for col in features
            }
        }
    except Exception as e:
        # Log full error details
        logging.error(f"Training failed: {str(e)}\n{traceback.format_exc()}")
        # Show user-friendly error message
        st.error(f"Error training model: {str(e)}")
        return None

def main():
    """Main application function"""
    # Configure page settings
    st.set_page_config(
        page_title="Oil Production Forecasting",
        layout="wide",  # Full-width layout
        page_icon="⛽"  # Browser tab icon
    )
    st.title("Oil Production Forecasting Platform")  # Main header
    
    # File upload widget in sidebar
    uploaded_file = st.sidebar.file_uploader(
        "Upload production data CSV",
        type=['csv']  # Restrict to CSV files
    )
    df = pd.DataFrame()  # Initialize empty dataframe
    
    if uploaded_file:
        try:
            # Read and sanitize uploaded data
            df = pd.read_csv(uploaded_file)
            # Clean column names by stripping whitespace
            df.columns = [col.strip() for col in df.columns]
            st.sidebar.success("File uploaded successfully!")
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
            return
    else:
        st.warning("Please upload a CSV file")
        return

    # Data validation checkpoint
    try:
        df = validate_data(df)
    except ValueError as e:
        st.error(f"Data validation failed: {str(e)}")
        return

    # Sidebar configuration panel
    st.sidebar.header("Configuration")
    
    # Model selection dropdown
    model_name = st.sidebar.selectbox("Select Model", list(MODELS.keys()))
    
    # Unit conversion radio buttons
    unit = st.sidebar.radio(
        "Select output unit:", 
        ["barrels/day", "liters/day"], 
        index=0, 
        key="unit"
    )
    
    # Model training trigger
    if st.sidebar.button("Train Model"):
        with st.spinner("Training model..."):  # Loading indicator
            model_data = train_model(df, model_name)
            if model_data:
                # Store trained model in session state
                st.session_state.model = model_data
                st.success(f"{model_name} trained successfully!")

    # Prediction interface
    model_data = st.session_state.get("model", {})
    feature_stats = model_data.get("feature_stats", {})
    
    if 'model' in st.session_state:
        features = model_data['features']
        
        # Dynamic input form generation
        input_values = {}
        cols = st.columns(4)  # 4-column layout
        for idx, feature in enumerate(features):
            with cols[idx % 4]:  # Distribute inputs across columns
                if feature in feature_stats:
                    # Input validation parameters
                    min_val = float(feature_stats[feature]['min'])
                    max_val = float(feature_stats[feature]['max'])
                    value = min_val  # Default to minimum value
                    step = 0.1  # Increment step size
                    
                    # Create number input widget
                    input_values[feature] = st.number_input(
                        label=feature.replace('_', ' ').title(),  # Format label
                        min_value=min_val,
                        max_value=max_val,
                        value=value,
                        step=step,
                        format="%.1f"  # Single decimal precision
                    )
                else:
                    st.error("Model not trained yet")
        
        # Prediction execution
        if st.button("Predict"):
            if feature_stats:
                # Create input dataframe from user values
                input_df = pd.DataFrame([input_values])[features]
                # Generate prediction using trained model
                prediction = model_data['model'].predict(input_df)[0]
                
                # Unit conversion logic
                if unit == "liters/day":
                    prediction_liters = prediction * 158.987  # Conversion factor
                    st.success(f"Predicted Oil Production: **{prediction_liters:,.2f} liters/day**")
                else:
                    st.success(f"Predicted Oil Production: **{prediction:,.2f} barrels/day**")
            else:
                st.error("Train a model first")

    # Model evaluation section
    if 'model' in st.session_state:
        model_data = st.session_state.model
        model = model_data['model']
        X_test, y_test = model_data['test_data']
        y_pred = model.predict(X_test)
        
        # Performance metrics display
        st.header("Performance Metrics")
        st.write(f"Cross-Validation R²: {model_data['cv_r2']:.3f}")  # Training accuracy
        st.write(f"Test Set R²: {model_data['test_r2']:.3f}")  # Generalization accuracy
        st.write(f"Cross-Validation MSE: {model_data['cv_mse']:.2f}")
        st.write(f"Test MSE: {model_data['test_mse']:.2f}")

        # Overfitting detection heuristic
        if abs(model_data['cv_r2'] - model_data['test_r2']) > 0.2:
            st.warning("⚠️ Significant overfitting detected")

        # Residual analysis visualization
        residuals = y_test - y_pred
        fig_res, ax_res = plt.subplots(figsize=(10, 6))
        ax_res.scatter(y_test, residuals, alpha=0.6)  # Residual plot
        ax_res.axhline(0, color='red', lw=1)  # Zero-error reference line
        ax_res.set_xlabel("Actual Oil Volume (barrels/day)")
        ax_res.set_ylabel("Residuals")
        st.pyplot(fig_res)

        # Enhanced SHAP Analysis with model-specific handling
        try:
            # Check model type for appropriate SHAP handling
            if model_name == "SVM":
                # Handle SVM with KernelExplainer
                scaler = model.named_steps['scaler']  # Extract scaler from pipeline
                svm_model = model.named_steps['model']  # Extract SVM model
                X_test_scaled = scaler.transform(X_test)  # Apply feature scaling
                
                # Create explainer with background sample
                explainer = shap.KernelExplainer(
                    svm_model.predict,  # Model prediction function
                    shap.sample(X_test_scaled, 100)  # Background data sample
                )
                # Calculate SHAP values
                shap_values_np = explainer.shap_values(X_test_scaled)
                
                # Create Explanation object for beeswarm plot
                shap_values = shap.Explanation(
                    values=shap_values_np,  # SHAP values array
                    base_values=explainer.expected_value,  # Base prediction value
                    data=X_test_scaled,  # Scaled input data
                    feature_names=features  # Feature names list
                )
                
            elif model_name in ["XGBoost", "Random Forest", "Decision Tree"]:
                # Handle tree-based models with TreeExplainer
                raw_model = model.named_steps['model']  # Extract model from pipeline
                explainer = shap.TreeExplainer(raw_model)  # Create explainer
                shap_values = explainer(X_test)  # Get SHAP values
                
            else:
                # Default handler for other models
                explainer = shap.Explainer(model.named_steps['model'], X_test)
                shap_values = explainer(X_test)
            
            # Generate SHAP beeswarm plot
            fig_shap, ax_shap = plt.subplots(figsize=(10, 6))
            shap.plots.beeswarm(shap_values, show=False)  # Create visualization
            st.pyplot(fig_shap)  # Display in Streamlit
            
        except Exception as e:
            st.warning(f"SHAP analysis failed: {str(e)}")

    # Model comparison section
    if st.checkbox("Compare All Models"):
        results = []
        for name in MODELS:
            with st.spinner(f"Training {name}..."):  # Progress indicator
                try:
                    res = train_model(df.copy(), name)
                    if res:
                        results.append({
                            'Model': name,
                            'Test R²': round(res['test_r2'], 4),
                            'CV R²': round(res['cv_r2'], 4),
                            'Test MSE': round(res['test_mse'], 4),
                            'CV MSE': round(res['cv_mse'], 4)
                        })
                except Exception as e:
                    st.error(f"Error training {name}: {str(e)}")
        
        if results:
            # Comparative performance table
            comp_df = pd.DataFrame(results)
            st.dataframe(comp_df)
            
            # Visual model comparison
            fig_comp, ax_comp = plt.subplots(figsize=(12, 6))
            sns.barplot(x='Model', y='Test R²', data=comp_df, ax=ax_comp)
            ax_comp.set_xticklabels(ax_comp.get_xticklabels(), rotation=45)
            st.pyplot(fig_comp)
        else:
            st.warning("No models trained successfully")

    # Data exploration section
    if st.checkbox("Show Data Analysis"):
        st.subheader("Data Summary")
        st.write(df.describe())  # Statistical summary
        
        # Box plot visualization
        fig_box, ax_box = plt.subplots(figsize=(12, 6))
        sns.boxplot(data=df[['DEPTH_MD', 'Reservoir_pressure', 'Working_hours', 'Oil_volume']], 
                    orient='h', ax=ax_box)
        st.pyplot(fig_box)
        
        # Correlation matrix visualization
        fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
        corr = df[['DEPTH_MD', 'Reservoir_pressure', 'Working_hours', 'Oil_volume']].corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", ax=ax_corr)
        st.pyplot(fig_corr)

    # Model export functionality
    if 'model' in st.session_state:
        model_data = st.session_state.model
        # Prepare model binary for download
        buffer = BytesIO()
        joblib.dump(model_data['model'], buffer)
        st.download_button(
            "Export Model",
            buffer.getvalue(),  # In-memory file contents
            f"{model_name}_model.pkl",  # Download filename
            "application/octet-stream"  # MIME type
        )

if __name__ == "__main__":
    main()