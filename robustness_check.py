"""Robustness check: was the model comparison tilted toward logistic regression?

The feature set (log tenure, MonthlyCharges dropped) was chosen with logistic regression,
and every model got the same 25-draw search. Tree models don't need a log transform and
have more hyperparameters, so this gives XGBoost and LightGBM:
  - a 100-draw search over wider ranges, and
  - the raw numeric features (tenure, MonthlyCharges, TotalCharges) as an alternative.
Training data only, 5-fold CV log loss. Takes a few minutes.

Run from the repo root:  python robustness_check.py
"""
import warnings; warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from scipy.stats import loguniform, randint, uniform
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

SEED = 42
df = pd.read_csv('data/telco_raw.csv')
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
df['churn'] = (df['Churn'] == 'Yes').astype(int)
addons = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
for c in addons: df[c] = df[c].replace('No internet service', 'No')
df['MultipleLines'] = df['MultipleLines'].replace('No phone service', 'No')
df['log_tenure'] = np.log1p(df['tenure'])
cat = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService'] + addons + \
      ['Contract', 'PaperlessBilling', 'PaymentMethod']
train, _ = train_test_split(df, test_size=0.2, stratify=df['churn'], random_state=SEED)
y = train['churn']
cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
pre = lambda num: ColumnTransformer([('n', StandardScaler(), num),
                                     ('c', OneHotEncoder(drop='first', handle_unknown='ignore'), cat)])

xgb_grid = {'n_estimators': randint(100, 800), 'learning_rate': loguniform(0.005, 0.3), 'max_depth': randint(1, 7),
            'subsample': uniform(0.5, 0.5), 'colsample_bytree': uniform(0.3, 0.7), 'min_child_weight': randint(1, 40),
            'reg_lambda': loguniform(1e-2, 30), 'gamma': loguniform(1e-3, 5)}
lgb_grid = {'n_estimators': randint(100, 800), 'learning_rate': loguniform(0.005, 0.3), 'num_leaves': randint(2, 40),
            'min_child_samples': randint(10, 300), 'subsample': uniform(0.5, 0.5), 'subsample_freq': [1],
            'colsample_bytree': uniform(0.3, 0.7), 'reg_lambda': loguniform(1e-2, 30)}
feature_sets = {'lean (used in notebook)': ['log_tenure', 'SeniorCitizen'],
                'full raw numeric': ['tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen']}

rows = []
for fname, num in feature_sets.items():
    for mname, est, grid in [('XGBoost', XGBClassifier(random_state=SEED, n_jobs=1, verbosity=0), xgb_grid),
                             ('LightGBM', LGBMClassifier(random_state=SEED, n_jobs=1, verbose=-1), lgb_grid)]:
        s = RandomizedSearchCV(Pipeline([('p', pre(num)), ('m', est)]), {f'm__{k}': v for k, v in grid.items()},
                               n_iter=100, cv=cv, scoring='neg_log_loss', random_state=SEED, n_jobs=-1)
        s.fit(train[num + cat], y)
        rows.append({'model': mname, 'features': fname, 'CV log loss': round(-s.best_score_, 4)})

lean = feature_sets['lean (used in notebook)']
lr = Pipeline([('p', pre(lean)), ('m', LogisticRegression(solver='saga', max_iter=5000, C=0.1907, l1_ratio=1))])
rows.append({'model': 'Logistic regression (notebook settings)', 'features': 'lean (used in notebook)',
             'CV log loss': round(-cross_val_score(lr, train[lean + cat], y, cv=cv, scoring='neg_log_loss').mean(), 4)})
print(pd.DataFrame(rows).sort_values('CV log loss').to_string(index=False))
