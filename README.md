# Predicting-Molecular-Solubility-using-Linear-Regression-and-Random-Forest

---

## 📁 Dataset

- **Name**: `delaney_solubility_with_descriptors.csv`
- **Source**: https://github.com/dataprofessor/data/blob/master/delaney_solubility_with_descriptors.csv
- **Target Variable**: `logS` (aqueous solubility)
- **Features**:
  - `MolLogP`: Octanol-water partition coefficient
  - `MolWt`: Molecular weight
  - `NumRotatableBonds`: Number of rotatable bonds
  - `AromaticProportion`: Ratio of aromatic atoms to total atoms

---

## 📊 Objective

Build a **Linear Regression** model to predict the `logS` value (aqueous solubility) from molecular descriptors.

---

## 🧠 Technologies Used

- Python 🐍
- Pandas
- Scikit-learn
- Jupyter Notebook / Google Colab

---

## ⚙️ How it Works

1. **Load Dataset** using `pandas`
2. **Split Data** into features (`X`) and target (`y`)
3. **Train-Test Split** to evaluate performance
4. **Fit** a Linear Regression model
5. **Predict** solubility values on test data
6. **Evaluate** using metrics like R² and Mean Squared Error

---

## 📌 Example Code

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# Load data
df = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/master/delaney_solubility_with_descriptors.csv')

# Split into features and target
X = df.drop('logS', axis=1)
y = df['logS']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)

# Train the model
lr = LinearRegression()
lr.fit(X_train, y_train)

# Predict and evaluate
y_pred = lr.predict(X_test)
print("R2 Score:", r2_score(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
📈 Results

Metric	Value (Example Output)
R² Score	~0.72
MSE	~0.65
Note: Exact results may vary slightly due to random data split.

📚 References
Delaney, J. S. (2004). ESOL: estimating aqueous solubility directly from molecular structure. Journal of Chemical Information and Computer Sciences, 44(3), 1000–1005.

Dataset from Data Professor GitHub

🙌 Author
Faizan Maqbool
ML Enthusiast | Developer | Educator

🔗 License
This project is for educational and research purposes. Attribution to the original data source is maintained.

yaml
Copy
Edit

---

Want help making a GitHub repo structure or uploading it via Colab/Git?








