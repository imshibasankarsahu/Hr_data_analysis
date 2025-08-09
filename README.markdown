# HR Attrition Prediction Project

## Overview
This project involves building a machine learning model to predict employee attrition based on the "HR DATA.csv" dataset. The code is implemented in a Jupyter Notebook (`Hr_Code.ipynb`) using Python and focuses on data preprocessing, model building, and hyperparameter tuning with a Random Forest Classifier.

## Dataset
The dataset (`HR DATA.csv`) contains employee-related features such as age, business travel, department, and more, with the target variable being `Attrition` (whether an employee leaves the company). The dataset has 1470 rows and 35 columns initially.

## Project Structure
- **File**: `Hr_Code.ipynb`
- **Dependencies**:
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `seaborn`
  - `scikit-learn`
  - `joblib`
- **Output**:
  - Trained model saved as `rf_model.joblib`

## Workflow
1. **Data Loading**:
   - Loads the dataset using `pandas.read_csv`.
   - Displays the first five rows to understand the structure.

2. **Data Preprocessing**:
   - Drops irrelevant columns: `EmployeeCount`, `StandardHours`, `EmployeeNumber`, `Over18`.
   - Encodes the target variable (`Attrition`) using `LabelEncoder`.
   - Creates dummy variables for categorical features (`BusinessTravel`, `Department`, `EducationField`, `Gender`, `JobRole`, `MaritalStatus`, `OverTime`).
   - Removes original categorical columns after encoding.
   - Final dataset shape: `(1470, 49)` for features and `(1470,)` for the target.

3. **Data Splitting**:
   - Splits data into training (75%) and testing (25%) sets using `train_test_split` with `random_state=40`.

4. **Model Building**:
   - Uses a `RandomForestClassifier` with `n_estimators=10` and `criterion='entropy'`.
   - Evaluates the model on training and testing sets using:
     - Classification report
     - Confusion matrix
     - Cross-validation accuracy (10-fold)
   - Training accuracy: ~98%, Testing accuracy: ~87%.

5. **Hyperparameter Tuning**:
   - Performs `GridSearchCV` to optimize the Random Forest model with:
     - Parameters: `n_estimators`, `criterion`, `max_depth`, `min_samples_split`, `min_samples_leaf`.
     - Cross-validation: 5-fold.
     - Best parameters: `{'criterion': 'entropy', 'max_depth': None, 'min_samples_leaf': 2, 'min_samples_split': 10, 'n_estimators': 10}`.
     - Best cross-validation accuracy: ~85.39%.

6. **Model Saving**:
   - Saves the best model from `GridSearchCV` as `rf_model.joblib` using `joblib`.

## How to Run
1. **Install Dependencies**:
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn joblib
   ```
2. **Prepare Dataset**:
   - Ensure `HR DATA.csv` is in the same directory as `Hr_Code.ipynb`.
3. **Run the Notebook**:
   - Open `Hr_Code.ipynb` in Jupyter Notebook or JupyterLab.
   - Execute cells sequentially to preprocess data, train the model, and save the output.
4. **Load Saved Model** (optional):
   ```python
   import joblib
   loaded_model = joblib.load('rf_model.joblib')
   ```

## Results
- **Training Performance**:
  - High precision and recall for non-attrition (class 0), slightly lower recall for attrition (class 1).
  - Average cross-validation accuracy: ~83.76%.
- **Testing Performance**:
  - Accuracy: ~86.96%.
  - Lower recall for attrition (class 1, ~21%), indicating potential for improvement in predicting positive cases.

## Potential Improvements
- Address class imbalance (e.g., using SMOTE or class weights).
- Experiment with other algorithms (e.g., XGBoost, Logistic Regression).
- Add feature selection to reduce dimensionality.
- Increase `n_estimators` or explore deeper hyperparameter tuning for better performance.

## License
This project is for educational purposes and does not include a specific license. Ensure compliance with dataset usage terms if applicable.