
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None
try:
    from lightgbm import LGBMClassifier
except Exception:
    LGBMClassifier = None
try:
    from catboost import CatBoostClassifier
except Exception:
    CatBoostClassifier = None

def get_models(random_state=42):
    models = {
        "DecisionTree": DecisionTreeClassifier(random_state=random_state),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=random_state)
    }
    if XGBClassifier is not None:
        models["XGBoost"] = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=random_state)
    if LGBMClassifier is not None:
        models["LightGBM"] = LGBMClassifier(random_state=random_state)
    if CatBoostClassifier is not None:
        models["CatBoost"] = CatBoostClassifier(verbose=0, random_state=random_state)
    return models
