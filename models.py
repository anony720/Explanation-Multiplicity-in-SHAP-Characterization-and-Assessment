import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import GridSearchCV, ShuffleSplit
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from rtdl_revisiting_models import FTTransformer
from tabpfn import TabPFNClassifier

from utils import clean_memory

# ---------------------------------------------------------
# FT-Transformer Wrapper
# ---------------------------------------------------------
class FTTransformerClf(ClassifierMixin, BaseEstimator):
    
    _estimator_type = "classifier"

    def __init__(self, 
                 num_features, cat_features, 
                 n_epochs=100, batch_size=256, lr=1e-4, weight_decay=1e-5,
                 d_block=192, n_blocks=3, attention_n_heads=8, dropout=0.1,
                 device=None, random_state=0):
        self.num_features = num_features
        self.cat_features = cat_features
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.d_block = d_block
        self.n_blocks = n_blocks
        self.attention_n_heads = attention_n_heads
        self.dropout = dropout
        self.random_state = random_state
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model_ = None
        self.scaler_ = StandardScaler()
        self.ordinal_encoder_ = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        self.cat_cardinalities_ = []
        self.classes_ = None

    def _set_seed(self):
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)
            if self.device == 'cuda':
                torch.cuda.manual_seed_all(self.random_state)

    def fit(self, X, y):
        self._set_seed()
        self.classes_ = np.unique(y)
        
        X = X.copy()
        
        # Preprocessing
        if self.num_features:
            X[self.num_features] = self.scaler_.fit_transform(X[self.num_features].astype(float))
            X_num = X[self.num_features].values.astype(np.float32)
        else:
            X_num = np.zeros((len(X), 0), dtype=np.float32)

        if self.cat_features:
            X[self.cat_features] = X[self.cat_features].astype(str)
            X_cat_enc = self.ordinal_encoder_.fit_transform(X[self.cat_features])
            self.cat_cardinalities_ = []
            for i, col in enumerate(self.cat_features):
                unique_cnt = len(self.ordinal_encoder_.categories_[i])
                self.cat_cardinalities_.append(unique_cnt + 1)
                mask = X_cat_enc[:, i] == -1
                X_cat_enc[mask, i] = unique_cnt
            X_cat = X_cat_enc.astype(np.int64)
        else:
            X_cat = np.zeros((len(X), 0), dtype=np.int64)
            self.cat_cardinalities_ = []

        X_num_t = torch.tensor(X_num, device=self.device)
        X_cat_t = torch.tensor(X_cat, device=self.device)
        y_t = torch.tensor(y.values if hasattr(y, 'values') else y, dtype=torch.float32, device=self.device).reshape(-1, 1)

        # Model Init
        self.model_ = FTTransformer(
            n_cont_features=len(self.num_features),
            cat_cardinalities=self.cat_cardinalities_,
            d_out=1,
            n_blocks=self.n_blocks,
            d_block=self.d_block,
            attention_n_heads=self.attention_n_heads,
            attention_dropout=self.dropout,
            ffn_d_hidden=None,
            ffn_d_hidden_multiplier=4/3,
            ffn_dropout=self.dropout,
            residual_dropout=0.0,
        ).to(self.device)

        # Training Loop
        optimizer = torch.optim.AdamW(self.model_.make_parameter_groups(), lr=self.lr, weight_decay=self.weight_decay)
        loss_fn = nn.BCEWithLogitsLoss()
        
        dataset = TensorDataset(X_num_t, X_cat_t, y_t)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model_.train()
        for epoch in range(self.n_epochs):
            for x_n, x_c, batch_y in loader:
                optimizer.zero_grad()
                
                if self.cat_cardinalities_:
                    pred = self.model_(x_n, x_c)
                else:
                    pred = self.model_(x_n, None)
                
                loss = loss_fn(pred, batch_y)
                loss.backward()
                optimizer.step()
        return self

    def predict_proba(self, X):
        self.model_.eval()
        X = X.copy()
        
        if self.num_features:
            X[self.num_features] = self.scaler_.transform(X[self.num_features].astype(float))
            X_num = X[self.num_features].values.astype(np.float32)
        else:
            X_num = np.zeros((len(X), 0), dtype=np.float32)
            
        if self.cat_features:
            X[self.cat_features] = X[self.cat_features].astype(str)
            X_cat_enc = self.ordinal_encoder_.transform(X[self.cat_features])
            for i in range(len(self.cat_features)):
                unique_cnt = len(self.ordinal_encoder_.categories_[i])
                mask = X_cat_enc[:, i] == -1
                X_cat_enc[mask, i] = unique_cnt
            X_cat = X_cat_enc.astype(np.int64)
        else:
            X_cat = np.zeros((len(X), 0), dtype=np.int64)

        probs = []
        dataset = TensorDataset(torch.tensor(X_num), torch.tensor(X_cat))
        loader = DataLoader(dataset, batch_size=512, shuffle=False)
        
        with torch.no_grad():
            for x_n, x_c in loader:
                x_n, x_c = x_n.to(self.device), x_c.to(self.device)
                if self.cat_cardinalities_:
                    logits = self.model_(x_n, x_c)
                else:
                    logits = self.model_(x_n, None)
                prob = torch.sigmoid(logits).cpu().numpy()
                probs.append(prob)
                
        prob_class_1 = np.concatenate(probs).flatten()
        prob_class_0 = 1 - prob_class_1
        return np.vstack([prob_class_0, prob_class_1]).T

    def predict(self, X):
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)

# ---------------------------------------------------------
# TabPFN Wrapper
# ---------------------------------------------------------
class TabPFNWrapper(BaseEstimator, ClassifierMixin):
    
    _estimator_type = "classifier"

    def __init__(self, device='cuda', random_state=0):
        self.device = device
        self.random_state = random_state
        self.model = None
        self.classes_ = None

    def _set_seed(self):
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)
            if self.device == 'cuda':
                torch.cuda.manual_seed_all(self.random_state)

    def fit(self, X, y):
        self._set_seed()
        self.model = TabPFNClassifier(device=self.device)
        self.model.fit(X, y)
        if hasattr(self.model, 'classes_'):
            self.classes_ = self.model.classes_
        else:
            self.classes_ = np.unique(y)
        return self

    def predict_proba(self, X):
        self._set_seed()
        # SHAP 샘플링의 무작위성을 위해 추론 시 Seed 고정 안 함
        return self.model.predict_proba(X)

    def predict(self, X):
        self._set_seed()
        return self.model.predict(X)

# ---------------------------------------------------------
# Train Model Function
# ---------------------------------------------------------
def train_model(X_train, Y_train, MODEL, MODEL_SEED, num_features, cat_features):
    
    if MODEL == "ftt":
        clf = FTTransformerClf(
            num_features=num_features,
            cat_features=cat_features,
            n_epochs=100,
            batch_size=256,
            device='cuda',
            random_state=MODEL_SEED
        )
        param_grid = {
            "d_block": [64, 128],
            "n_blocks": [2, 3],
            "lr": [1e-4, 3e-4],
        }
        model = clf 

    elif MODEL == "tabpfn":
        tabpfn_preprocess = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), num_features),
                ("cat", OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), cat_features),
            ],
            remainder='drop'
        )
        clf = TabPFNWrapper(device='cuda', random_state=MODEL_SEED)
        model = Pipeline(steps=[("preprocess", tabpfn_preprocess), ("clf", clf)])
        param_grid = {}

    else:
        # Classical ML Models
        num_transformer = Pipeline(steps=[("scaler", StandardScaler())])
        onehot_transformer = Pipeline(steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))])
        preprocess = ColumnTransformer(
            transformers=[
                ("num", num_transformer, num_features),
                ("cat", onehot_transformer, cat_features),
            ]
        )

        if MODEL == "lr":
            clf = LogisticRegression(max_iter=1000, n_jobs=-1, random_state=MODEL_SEED)
            param_grid = {"clf__C": [0.1, 1.0, 10.0], "clf__penalty": ["l2"], "clf__solver": ["lbfgs"]}
        elif MODEL == "dt":
            clf = DecisionTreeClassifier(random_state=MODEL_SEED)
            param_grid = {"clf__max_depth": [3, 5, None], "clf__min_samples_leaf": [1, 5, 10]}
        elif MODEL == "rf":
            clf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=MODEL_SEED)
            param_grid = {"clf__max_depth": [None, 7, 15], "clf__min_samples_leaf": [1, 5]}
        elif MODEL == "xgb":
            clf = XGBClassifier(n_estimators=200, tree_method="hist", eval_metric="logloss", n_jobs=-1, random_state=MODEL_SEED)
            param_grid = {"clf__max_depth": [3, 5], "clf__learning_rate": [0.05, 0.1], "clf__subsample": [0.8, 1.0]}
        
        model = Pipeline(steps=[("preprocess", preprocess), ("clf", clf)])

    print(f"Running GridSearchCV for {MODEL}...")
    if param_grid:
        grid = GridSearchCV(
            model,
            param_grid=param_grid,
            cv=ShuffleSplit(test_size=0.2, n_splits=1, random_state=0), 
            scoring="roc_auc",
            n_jobs=1,
            verbose=1 
        )
        grid.fit(X_train, Y_train)
        print(f"Best Params: {grid.best_params_}")
        best_model = grid.best_estimator_
        
        # GridSearch 객체 삭제로 메모리 확보
        del grid
        clean_memory()
    else:
        # TabPFN 등 튜닝 없는 경우
        model.fit(X_train, Y_train)
        best_model = model    
    
    return best_model