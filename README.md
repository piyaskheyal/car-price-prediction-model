# Car Price Prediction using Machine Learning

This project implements a machine learning model to predict car prices based on various features such as horsepower, car brand, and fuel system. It utilizes algorithms like **Lasso Regression** and employs techniques like **target encoding** to process categorical features.

## Libraries Used

The following Python libraries have been used in this project:

- `pandas` for data manipulation and analysis
- `numpy` for numerical operations
- `matplotlib` for plotting graphs
- `seaborn` for data visualization
- `joblib` for model saving and loading
- `sklearn` (scikit-learn) for machine learning model creation and evaluation:
  - `train_test_split` for splitting data into training and testing sets
  - `KFold` for cross-validation
  - `Lasso` for regression modeling
  - `StandardScaler` for feature scaling
  - `mean_absolute_percentage_error`, `r2_score` for model evaluation

## Dataset

The dataset used for this project can be found here:

[Car Data from Kaggle](https://www.kaggle.com/datasets/goyalshalini93/car-data/data)

It contains several car attributes such as:
- Car brand
- Horsepower
- Price
- Fuel system
- Aspiration
- And other relevant features

## Steps in the Project

1. **Data Preprocessing:**
   - Loading the dataset and performing exploratory data analysis (EDA).
   - Handling missing values, encoding categorical features (e.g., using target encoding).
   - Scaling numerical features using `StandardScaler`.

2. **Model Building:**
   - Splitting the data into training and testing sets.
   - Applying **Lasso Regression** to predict car prices.
   - Using **K-Fold Cross-Validation** for model evaluation.

3. **Model Evaluation:**
   - The model performance is evaluated using:
     - **R-squared score** (to understand the goodness of fit).
     - **Mean Absolute Percentage Error** (MAPE) to assess prediction accuracy.

4. **Saving the Model:**
   - The trained model is saved using **joblib** for later use.

## How to Run

1. Clone the repository:
   ```bash
   https://github.com/piyaskheyal/car-price-prediction-model
   cd car-price-prediction
   ```

2. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

## Data Preprocessing and Feature Engineering

- **Target Encoding**: Categorical features, like car company names, are encoded by replacing each category with the average price of the respective car company.
- **Feature Scaling**: Numerical features such as horsepower and car width are scaled using **StandardScaler**.

## Results

The model achieves good predictive performance, with a high R-squared score and low MAPE, showing that it can reliably predict car prices based on the provided features.

## Dependencies

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `joblib`
- `scikit-learn`

## Contributing

1. Fork the repository.
2. Create a new branch.
3. Commit your changes.
4. Open a pull request to contribute to the project.

## Acknowledgments

- [Kaggle Car Dataset](https://www.kaggle.com/datasets/goyalshalini93/car-data/data) for providing the dataset.