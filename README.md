# Score-Prediction-Linear-Regression

A simple Python machine learning project that predicts student math scores using linear regression.

This project reads training and test data from Excel files, trains a custom linear regression model from scratch using the normal equation, predicts final scores for the test set, prints the learned coefficients, calculates the training `R^2` score, and visualizes the two most influential features.

## What this project does

- Loads training data from `TrainMath.xlsx`
- Loads test data from `TestMath.xlsx`
- Uses `final` as the prediction target
- Trains a custom linear regression model without scikit-learn
- Predicts final scores for the test dataset
- Prints model coefficients
- Computes the training `R^2` score
- Ranks the strongest input features by coefficient magnitude
- Plots the top two features against the final score

## Tech stack

- Python
- NumPy
- pandas
- matplotlib
- Excel datasets (`.xlsx`)

## Project structure

```text
Score-Prediction-Linear-Regression/
|-- main.py
|-- ScorePredict.py
|-- TrainMath.xlsx
`-- TestMath.xlsx
```

## Files overview

- `main.py`: Runs the full workflow
- `ScorePredict.py`: Contains the dataset loader and custom `LinearRegression` class
- `TrainMath.xlsx`: Training dataset
- `TestMath.xlsx`: Test dataset

## How it works

1. The `data()` function loads both Excel files with pandas.
2. The target column `final` is separated from the input features.
3. The model adds a bias column of ones to the feature matrix.
4. The coefficients are calculated using the normal equation:

```text
(X^T X)^-1 X^T y
```

5. The trained model predicts final scores for the test data.
6. The code calculates `R^2` using the training set.
7. The two strongest features are selected using the absolute coefficient values.
8. A matplotlib plot is shown for those top two features.

## Requirements

- Python 3.9 or newer
- Required libraries:
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `openpyxl`

## Installation

Install the dependencies with:

```bash
pip install numpy pandas matplotlib openpyxl
```

## How to run

From the project folder, run:

```bash
python main.py
```

## Example output

The script prints:

- predicted final scores for the test set
- the regression coefficients
- the training `R^2` score

It also opens a graph showing the top two most important features compared with the final score.

## Model details

- The model is implemented manually in `ScorePredict.py`
- It does not use `scikit-learn`
- A bias term is added as the last coefficient
- Feature importance is based on the absolute value of each feature weight

## Notes

- The target column is currently hardcoded as `final`
- The code expects both Excel files to include the same feature columns
- The plotting function only visualizes the top two ranked features
- `R^2` is currently calculated on the training data, not the test data

## Limitations

- The code assumes `X^T X` is invertible
- There is no feature scaling or normalization
- There is no train/test split logic inside the code because the datasets are already separated
- There is no error handling for missing files, missing columns, or invalid Excel content
- The model is limited to linear relationships

## Future improvements

- Add error handling for invalid input files
- Evaluate the model on the test set with extra metrics such as MAE or RMSE
- Save predictions to a new Excel or CSV file
- Support custom target column selection
- Add feature scaling
- Compare the custom model with `scikit-learn`

## Author

Built as a linear regression project for predicting student scores from Excel-based datasets.
