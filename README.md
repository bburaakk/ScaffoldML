
# End to end ML Data Preprocessing Pipeline



## Description

This project is a **configurable data preprocessing tool** for machine learning workflows. Its purpose is to enable data scientists and engineers to dynamically manage data cleaning, transformation, and scaling steps without modifying Python code, simply by configuring a `config.yaml` file. This accelerates experimentation processes and makes the preprocessing logic more transparent and reproducible.

## Features


- **Configuration with YAML:** All preprocessing steps and parameters are managed from the `config.yaml` file.
- **Dynamic Component Loading:** Dynamically loads classes from modules such as `sklearn.preprocessing` and `sklearn.impute`.
- **Flexible Pipeline:** Creates separate pipelines for numerical and categorical features.
- **Easy Extensibility:** Add new features or steps by simply updating the YAML file.
- **Modern Python Tools:** Uses the `uv` package manager and `pyproject.toml` for a standards-compliant structure.


## Installation

Follow these steps to install and run the project on your local machine.

Requirements
- Git
- Python 3.12 or newer
- uv (Python package manager)

#### Clone the project
To download the project files to your computer, run the following command in the terminal:
```bash
  git clone https://github.com/FirstRealMyProject/FirstRealMyProject.git
  cd ScaffoldML
```
#### Build Dependencies
This project uses uv as its Python package manager. If uv is not installed on your system, install it first:
```bash
  pip install uv
```
Then, optionally install all necessary packages (including development tools) with the following command:
```bash
  uv pip install -e .[dev]
```
## Usage/Examples

```python
import pandas as pd
import numpy as np

# Import the main components of the project
from ScaffoldML.src.data_preprocessing.data_preprocessor import DataPreprocessor
from ScaffoldML.src.data_preprocessing.load_config import load_config

# --- 1. Load the configuration ---
# This function reads the YAML file and converts it to Pydantic models.
config = load_config(config_path=“config.yaml”)

# --- 2. Create Sample Dataset ---
# Let's create a DataFrame containing the columns specified in the configuration file and some missing values.
# Let's create a DataFrame.
data = {
‘city’: [‘New
York’, ‘London’, ‘Paris’, ‘Tokyo’, ‘New
York’, None],
‘payment_method’: [‘Credit
Card’, ‘PayPal’, ‘Credit
Card’, None, ‘Bank
Transfer’, ‘PayPal’],
‘age’: [25, 30, 35, 40, 28, np.nan],
‘claim_amount’: [100.5, 250.0, np.nan, 500.75, 120.0, 300.0]
}
df = pd.DataFrame(data)

print(“--- Original
Data - --”)
print(df)
print(“\nNumber
of
Missing
Values:\n”, df.isnull().sum())


# --- 3. Create and Run the Preprocessor ---
# Start DataPreprocessor using the configuration object.
preprocessor = DataPreprocessor(config=config)

# The fit_transform method both learns (fit) and transforms (transform) the data.
transformed_data = preprocessor.fit_transform(df)

# Let's convert the transformed data into a more understandable DataFrame
# Note: The get_feature_names_out method can be used with scikit-learn 1.0+.
transformed_df = pd.DataFrame(
    transformed_data,
    columns=preprocessor.pipeline.get_feature_names_out()
)

# --- 4. Review the result ---
print(“\n\n - -- Processed
Data - --”)
print(transformed_df)
```
### Explanation of Sample Output

When you run the code above, you will see the following results:
1. Missing Values Are Filled:
- NaN values in the age and claim_amount columns are filled with the mean value specified in config.yaml.
- None values in the city and payment_method columns are filled with the most frequently occurring value (most_frequent).
2. Numeric Data Is Scaled:
- The age and claim_amount columns are standardized using StandardScaler (with a mean of 0 and a standard deviation of 1).
3. Categorical Data is Encoded:
- The city and payment_method columns are converted to numerical format using OneHotEncoder. A new column is created for each category (city_London, city_New York, etc.).
## Testing

To run all tests in the project and generate a coverage report showing how much of the code has been tested, simply run the following command in the project root directory:

```bash
  pytest --cov
```
## Acknowledgements

- Special thanks to Kaan Bıçakçı and his YouTube channel for the valuable content he provides on modern Python development practices, clean code, and structuring machine learning projects.
[🔗 You can access his channel here](https://awesomeopensource.com/project/elangosundar/awesome-README-templates)


