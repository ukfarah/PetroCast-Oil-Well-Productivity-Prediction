# 🛢️ PetroCast: Oil Well Productivity Prediction

**PetroCast** is a predictive analytics application designed to estimate oil well productivity based on operational, geological, and spatial features. It empowers petroleum engineers, geologists, and data scientists with the ability to make data-driven decisions about oil field performance and well planning using machine learning techniques.

The project includes a machine learning pipeline for data preprocessing, training, evaluation, and a fully interactive dashboard using **Streamlit**.

---

## 🧠 Project Motivation

Predicting oil well performance before drilling can significantly reduce costs and increase operational efficiency. Traditional methods rely heavily on manual analysis and domain expertise. With the rise of data availability in the petroleum industry, we can leverage machine learning to support and enhance engineering insights.

The goal of this project is to:
- Clean and preprocess well log and operational data
- Train predictive models to estimate well productivity
- Build a visual interface for real-time inference and analysis

---

## 🚀 Key Features

- ✅ **Data Preprocessing**: Cleans noisy or missing values and transforms geological and operational data into usable features.
- ✅ **Model Training**: Implements supervised ML models to predict productivity values (regression task).
- ✅ **Streamlit Dashboard**: A web-based interface for users to interact with the model, input parameters, and get predictions.
- ✅ **Modular Codebase**: Clear separation of scripts for preprocessing, training, and app deployment.
- ✅ **Visual Analytics**: Visualize the data and model behavior to interpret predictions.

---

## 🏗️ Project Architecture

```bash

PetroCast/
│
├── Book2.csv # Auxiliary raw data
├── Final_Cleaned_Well_Logs.csv # Primary cleaned dataset
│
├── preprocess.py # Data cleaning and transformation logic
├── train_model.py # Model training and evaluation script
├── streamlit_app.py # Web app for interactive prediction
├── Petrocast-1.py # Main execution or experimental script
│
├── Requirement.txt # Python dependencies
└── Oil_Well_Project_Report_Without_Images.docx # Documentation in Arabic

```


Each component plays a distinct role:
- **Data Files**: Contain historical geological and operational records.
- **Scripts**: Organize the machine learning pipeline and interface logic.
- **App**: Exposes predictions and visualizations via Streamlit.

---

## 🧪 Technologies Used

| Category | Tools / Libraries |
|---------|-------------------|
| **Language** | Python 3.x |
| **Data Handling** | pandas, numpy |
| **Modeling** | scikit-learn |
| **Visualization** | matplotlib, seaborn |
| **Web App** | streamlit |
| **Development** | Jupyter, VSCode |
| **Documentation** | Markdown, Word |

---

## 📊 Features Used in Modeling

The dataset includes a combination of:

- 🔹 **Operational Features**: Drilling pressure, flow rate, mud weight, etc.
- 🔸 **Geological Features**: Formation name, rock porosity, lithology, etc.
- 🔻 **Geographic Location**: Longitude, latitude, well name.
- 🔺 **Depth and Formation Columns**: Zone thickness, formation top/bottom, etc.

These variables are crucial in evaluating the productivity of an oil well before or after drilling.

---

## 📈 Model Workflow

1. **Data Loading**: Load raw and cleaned datasets (`Final_Cleaned_Well_Logs.csv`)
2. **Preprocessing**: Handle missing values, encode categorical data, scale numerical features
3. **Training**: Use `train_model.py` to train a regression model (e.g., Random Forest)
4. **Evaluation**: Assess model performance using MAE, RMSE, and R²
5. **Deployment**: Launch the Streamlit dashboard via `streamlit_app.py`

---

## 🌐 Running the App Locally

To launch the full application on your local machine:

### 1. Clone the repository

```bash
git clone https://github.com/your-username/PetroCast.git
cd PetroCast
```

### 2. Install dependencies
```bash
pip install -r Requirement.txt
```

### 3. Run the Streamlit app
```bash
streamlit run streamlit_app.py
```

## 🧭 Sample Use Cases

- 🎯 **Production Planning**: Estimate future well output before drilling  
- 🛠️ **Operational Optimization**: Identify which operational settings yield better productivity  
- 🛰️ **Geological Analysis**: Understand how geological zones impact well performance  
- 🧪 **ML Experimentation**: Use this as a template for similar geoscientific ML applications  

---

## 🔧 Future Enhancements

- 📡 Integrate real-time data feeds from sensors and SCADA systems  
- 📉 Time-series forecasting of production curves  
- 🗺️ GIS integration for spatial visualization of productivity  
- 🧠 Deep learning model (e.g., LSTM or TabNet) for improved accuracy  

---

## 🗂️ Dataset Summary

The cleaned dataset (`Final_Cleaned_Well_Logs.csv`) includes:

- ~100+ geological and operational features  
- Numerical, categorical, and spatial attributes  
- **Target variable**: *Productivity Rate* (e.g., barrels/day or similar)  

---

## 📝 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

