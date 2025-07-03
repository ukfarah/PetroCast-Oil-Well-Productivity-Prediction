import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import shap

def train_and_evaluate():
    # Load preprocessed data
    X_train = pd.read_csv('data/X_train.csv')
    y_train = pd.read_csv('data/y_train.csv')['OilRate']
    X_test = pd.read_csv('data/X_test.csv')
    y_test = pd.read_csv('data/y_test.csv')['OilRate']
    
    # Train model
    model = xgb.XGBRegressor()
    model.fit(X_train, y_train)
    
    # Save model
    joblib.dump(model, 'models/xgboost_model.joblib')
    
    # Evaluate
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # SHAP analysis
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)
    
    # Save summary plot
    shap.summary_plot(shap_values, X_train, plot_type="bar", show=False)
    plt.savefig('models/shap_summary.png', bbox_inches='tight')
    
    return mse, r2

if __name__ == "__main__":
    mse, r2 = train_and_evaluate()
    print(f"MSE: {mse:.2f}, R²: {r2:.2f}")