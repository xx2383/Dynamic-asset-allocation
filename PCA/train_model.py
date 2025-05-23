import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from itertools import product
from sklearn.metrics import accuracy_score
from sklearn.metrics import adjusted_rand_score


def plot_market_regime(df, asset, train_start, train_end):
    """
    绘制市场状态 (Market Regime) 可视化
    - 绿色：牛市 (Bullish)
    - 红色：熊市 (Bearish)
    """
    df_view = df[(df['DateTime']>=train_start) & (df['DateTime']<=train_end)].copy()
    
    # 确保 DateTime 是 datetime 类型
    df_view["DateTime"] = pd.to_datetime(df_view["DateTime"])

    # 只绘制已有 Market_Regime_Num 的数据
    df_view = df_view.dropna(subset=["Market_Regime_Num"])

    plt.figure(figsize=(12, 6))

    # 确保 x 轴是 datetime 类型
    plt.plot(df_view["DateTime"].values, df_view[f"{asset}"].values, label="S&P 500", color="black", linewidth=1)

    # 填充牛熊市场背景颜色
    for i in range(len(df_view) - 1):
        color = "green" if df_view["Market_Regime_Num"].iloc[i] == 1 else "red"
        plt.axvspan(df_view["DateTime"].iloc[i], df_view["DateTime"].iloc[i + 1], color=color, alpha=0.3)

    plt.title("S&P 500 with Market Regimes")
    plt.xlabel("Date")
    plt.ylabel("S&P 500 Price")
    plt.legend()
    plt.grid()
    plt.xticks(rotation=45)
    plt.show()




class PenalizedKMeansDP:
    def __init__(self, n_clusters=2, max_iter=10000, tol=1e-4, penalty=0.1, J_threshold=1e-3, random_state=42):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.penalty = penalty
        self.J_threshold = J_threshold
        self.random_state = random_state

    def fit(self, X):
        np.random.seed(self.random_state)
        initial_indices = np.random.choice(len(X), self.n_clusters, replace=False)
        self.centroids_ = X[initial_indices]

        J_old = np.inf
        penalty_mx = self.penalty * (np.ones((self.n_clusters, self.n_clusters)) - np.eye(self.n_clusters))

        for iteration in range(self.max_iter):
            prev_centroids = self.centroids_.copy()
            loss_mx = cdist(X, self.centroids_, "sqeuclidean")
            values, assign = self.dp(loss_mx, penalty_mx)

            new_centroids = np.array([
                np.average(X[assign == k], axis=0) if np.any(assign == k) else prev_centroids[k]
                for k in range(self.n_clusters)
            ])
            self.centroids_ = new_centroids

            loss_term_1 = np.sum([np.linalg.norm(X[i] - self.centroids_[assign[i]])**2 for i in range(len(X))])
            loss_term_2 = self.penalty * np.sum(assign[1:] != assign[:-1])
            J_new = loss_term_1 + loss_term_2

            if abs(J_new - J_old) < self.tol or J_new < self.J_threshold:
                break

            J_old = J_new

        self.labels_ = assign
        return self

    def dp(self, loss_mx, penalty_mx):
        n_s, n_c = loss_mx.shape
        values = np.empty((n_s, n_c))
        assign = np.empty(n_s, dtype=int)
        values[0] = loss_mx[0]

        for t in range(1, n_s):
            values[t] = loss_mx[t] + (values[t-1][:, np.newaxis] + penalty_mx).min(axis=0)

        assign[-1] = values[-1].argmin()
        for t in range(n_s - 1, 0, -1):
            assign[t-1] = (values[t-1] + penalty_mx[:, assign[t]]).argmin()

        return values, assign

    def predict(self, X):
        distances = cdist(X, self.centroids_, "sqeuclidean")
        return np.argmin(distances, axis=1)

    


    
    
    
# def train_kmeans(df, asset, feature_columns, feature_columns_xgb, train_start, train_end, validation_years=5, next_update_date=None, pca=False, n_comp=5):
#     df = df.copy()

#     # Split data
#     validation_start = pd.to_datetime(train_end) - pd.DateOffset(years=validation_years)
#     train_mask = (df["DateTime"] >= pd.to_datetime(train_start)) & (df["DateTime"] < validation_start)
#     val_mask = (df["DateTime"] >= validation_start) & (df["DateTime"] < pd.to_datetime(train_end))
#     future_mask = (df["DateTime"] >= pd.to_datetime(train_end)) & (df["DateTime"] <= pd.to_datetime(next_update_date))

#     df_train = df.loc[train_mask].copy()
#     df_val = df.loc[val_mask].copy()

#     # If using PCA, define PCA features only for KMeans
#     if pca:
#         pca_model = PCA(n_components=n_comp)
#         X_train_pca = pca_model.fit_transform(df_train[feature_columns].dropna())
#         df_train = df_train.loc[df_train[feature_columns].dropna().index]
#         pca_cols = [f"PCA_{i+1}" for i in range(n_comp)]
#         df_train[pca_cols] = X_train_pca
#         feature_columns_kmeans = pca_cols
#     else:
#         feature_columns_kmeans = feature_columns

#     penalties = range(2, 20)
#     xgb_param_grid = {
#         'max_depth': [5, 10, 20],
#         'n_estimators': range(5, 40, 10),
#         'learning_rate': np.arange(0.05, 0.35, 0.1)
#     }
#     xgb_param_combinations = list(product(xgb_param_grid['max_depth'], xgb_param_grid['n_estimators'], xgb_param_grid['learning_rate']))

#     best_score = -np.inf
#     best_penalty = None
#     best_xgb_params = {}
#     best_train_labels = None

#     for penalty in penalties:
#         kmeans = PenalizedKMeansDP(n_clusters=2, penalty=penalty).fit(df_train[feature_columns_kmeans].dropna().values)
#         df_train["Market_Regime_Train"] = kmeans.labels_

#         if df_train["Market_Regime_Train"].nunique() == 1:
#             continue

#         regime_means = df_train.groupby("Market_Regime_Train")[f"{asset}_returns"].mean().sort_values()
#         regime_map = {regime_means.index[0]: 0, regime_means.index[1]: 1}
#         df_train["Market_Regime_Num_Train"] = df_train["Market_Regime_Train"].map(regime_map)

#         y_train = df_train["Market_Regime_Num_Train"].shift(-1).dropna()
#         X_train_xgb = df.loc[y_train.index, feature_columns_xgb].dropna()
#         y_train = y_train.loc[X_train_xgb.index]

#         y_val = df_val["Market_Regime_Num_Train"].shift(-1).dropna()
#         X_val_xgb = df.loc[y_val.index, feature_columns_xgb].dropna()
#         y_val = y_val.loc[X_val_xgb.index]

#         if y_train.empty or y_val.empty:
#             continue

#         for max_depth, n_estimators, learning_rate in xgb_param_combinations:
#             model = xgb.XGBClassifier(max_depth=max_depth, n_estimators=n_estimators, learning_rate=learning_rate)
#             model.fit(X_train_xgb.values, y_train)

#             preds = model.predict(X_val_xgb.values)
#             df.loc[y_val.index, "XGB_Regime_Forecast"] = preds

#             df["Position"] = df["XGB_Regime_Forecast"].shift(1)
#             df["Strategy_Return"] = df[f"{asset}_returns"] * df["Position"]
#             strat_returns = df.loc[val_mask, "Strategy_Return"].dropna()

#             score = cumulative_return(strat_returns)
#             if score > best_score:
#                 best_score = score
#                 best_penalty = penalty
#                 best_xgb_params = {
#                     'max_depth': max_depth,
#                     'n_estimators': n_estimators,
#                     'learning_rate': learning_rate,
#                     'strat_returns': score
#                 }
#                 best_train_labels = df_train["Market_Regime_Num_Train"].copy()

#     # Retrain on full train+val with best_penalty (matching regime pattern)
#     full_train_mask = (df["DateTime"] >= pd.to_datetime(train_start)) & (df["DateTime"] < pd.to_datetime(train_end))
#     df_full = df.loc[full_train_mask].copy()

#     # If PCA, also apply to df_full
#     if pca:
#         X_full_pca = pca_model.transform(df_full[feature_columns].dropna())
#         df_full = df_full.loc[df_full[feature_columns].dropna().index]
#         df_full[pca_cols] = X_full_pca

#     similarity_scores = {}
#     for penalty in penalties:
#         km = PenalizedKMeansDP(n_clusters=2, penalty=penalty).fit(df_full[feature_columns_kmeans].dropna().values)
#         full_labels = km.labels_
#         df_full["Market_Regime"] = full_labels

#         if df_full["Market_Regime"].nunique() == 1:
#             continue

#         score = adjusted_rand_score(best_train_labels.loc[df_train[feature_columns_kmeans].dropna().index], full_labels[:len(best_train_labels)])
#         similarity_scores[penalty] = score

#     best_penalty_final = max(similarity_scores, key=similarity_scores.get)
#     print(f"✅ Final chosen penalty: {best_penalty_final}, based on training pattern similarity.")
#     print(f"✅ Final chosen xgb params: {best_xgb_params}")

#     final_kmeans = PenalizedKMeansDP(n_clusters=2, penalty=best_penalty_final).fit(df_full[feature_columns_kmeans].dropna().values)
#     df.loc[full_train_mask, "Market_Regime"] = final_kmeans.labels_

#     regime_mean_final = df.loc[full_train_mask].groupby("Market_Regime")[f"{asset}_returns"].mean().sort_values()
#     final_map = {regime_mean_final.index[0]: "Bearish", regime_mean_final.index[1]: "Bullish"}
#     df.loc[full_train_mask, "Market_Regime"] = df["Market_Regime"].map(final_map)
#     df.loc[full_train_mask, "Market_Regime_Num"] = df["Market_Regime"].map({"Bullish": 1, "Bearish": 0})

#     # Final XGB prediction
#     y_final = df.loc[full_train_mask, "Market_Regime_Num"].shift(-1).dropna()
#     X_final = df.loc[y_final.index, feature_columns_xgb].dropna()
#     y_final = y_final.loc[X_final.index]

#     final_model = xgb.XGBClassifier(**best_xgb_params)
#     final_model.fit(X_final.values, y_final)

#     X_pred = df.loc[future_mask, feature_columns_xgb].dropna()
#     if not X_pred.empty:
#         df.loc[X_pred.index, "Regime_Forecast"] = final_model.predict(X_pred.values)

#     return df




def train_kmeans(df, asset, feature_columns, feature_columns_xgb, train_start, train_end, validation_years=5, next_update_date=None, pca=False, n_comp_kmeans=5, pca_xgb=False, n_comp_xgb=5):
    df = df.copy()

    # Split data
    validation_start = pd.to_datetime(train_end) - pd.DateOffset(years=validation_years)
    train_mask = (df["DateTime"] >= pd.to_datetime(train_start)) & (df["DateTime"] < validation_start)
    val_mask = (df["DateTime"] >= validation_start) & (df["DateTime"] < pd.to_datetime(train_end))
    future_mask = (df["DateTime"] >= pd.to_datetime(train_end)) & (df["DateTime"] <= pd.to_datetime(next_update_date))

    df_train = df.loc[train_mask].copy()
    df_val = df.loc[val_mask].copy()

    if pca:
        pca_model_kmeans = PCA(n_components=n_comp_kmeans)
        X_train_pca = pca_model_kmeans.fit_transform(df_train[feature_columns].dropna())
        df_train = df_train.loc[df_train[feature_columns].dropna().index]
        pca_cols_kmeans = [f"PCA_KM_{i+1}" for i in range(n_comp_kmeans)]
        df_train[pca_cols_kmeans] = X_train_pca
        feature_columns_kmeans = pca_cols_kmeans
    else:
        feature_columns_kmeans = feature_columns

    if pca_xgb:
        pca_model_xgb = PCA(n_components=n_comp_xgb)
        X_xgb_pca = pca_model_xgb.fit_transform(df[feature_columns_xgb].dropna())
        df = df.loc[df[feature_columns_xgb].dropna().index]
        pca_cols_xgb = [f"PCA_XGB_{i+1}" for i in range(n_comp_xgb)]
        df[pca_cols_xgb] = X_xgb_pca
        feature_columns_xgb_used = pca_cols_xgb
    else:
        feature_columns_xgb_used = feature_columns_xgb

    penalties = range(0, 20)
    xgb_param_grid = {
        'max_depth': [5, 10, 20],
        'n_estimators': range(5, 40, 10),
        'learning_rate': np.arange(0.05, 0.35, 0.1)
    }
    xgb_param_combinations = list(product(xgb_param_grid['max_depth'], xgb_param_grid['n_estimators'], xgb_param_grid['learning_rate']))

    best_score = -np.inf
    best_penalty = None
    best_xgb_params = {}
    best_train_labels = None

    for penalty in penalties:
        kmeans = PenalizedKMeansDP(n_clusters=2, penalty=penalty).fit(df_train[feature_columns_kmeans].dropna().values)
        df_train["Market_Regime_Train"] = kmeans.labels_

        if df_train["Market_Regime_Train"].nunique() == 1:
            continue

        regime_means = df_train.groupby("Market_Regime_Train")[f"{asset}_returns"].mean().sort_values()
        regime_map = {regime_means.index[0]: 0, regime_means.index[1]: 1}
        df_train["Market_Regime_Num_Train"] = df_train["Market_Regime_Train"].map(regime_map)

        y_train = df_train["Market_Regime_Num_Train"].shift(-1).dropna()
        X_train_xgb = df.loc[y_train.index, feature_columns_xgb_used].dropna()
        y_train = y_train.loc[X_train_xgb.index]

        y_val = df_val["Market_Regime_Num_Train"].shift(-1).dropna()
        X_val_xgb = df.loc[y_val.index, feature_columns_xgb_used].dropna()
        y_val = y_val.loc[X_val_xgb.index]

        if y_train.empty or y_val.empty:
            continue

        for max_depth, n_estimators, learning_rate in xgb_param_combinations:
            model = xgb.XGBClassifier(max_depth=max_depth, n_estimators=n_estimators, learning_rate=learning_rate)
            model.fit(X_train_xgb.values, y_train)

            preds = model.predict(X_val_xgb.values)
            df.loc[y_val.index, "XGB_Regime_Forecast"] = preds

            df["Position"] = df["XGB_Regime_Forecast"].shift(1)
            df["Strategy_Return"] = df[f"{asset}_returns"] * df["Position"]
            strat_returns = df.loc[val_mask, "Strategy_Return"].dropna()

            score = cumulative_return(strat_returns)
            if score > best_score:
                best_score = score
                best_penalty = penalty
                best_xgb_params = {
                    'max_depth': max_depth,
                    'n_estimators': n_estimators,
                    'learning_rate': learning_rate,
                    'strat_returns': score
                }
                best_train_labels = df_train["Market_Regime_Num_Train"].copy()

    # Retrain on train + val with best_penalty (matching regime pattern)
    full_train_mask = (df["DateTime"] >= pd.to_datetime(train_start)) & (df["DateTime"] < pd.to_datetime(train_end))
    df_full = df.loc[full_train_mask].copy()

    if pca:
        X_full_pca = pca_model_kmeans.transform(df_full[feature_columns].dropna())
        df_full = df_full.loc[df_full[feature_columns].dropna().index]
        df_full[pca_cols_kmeans] = X_full_pca

    similarity_scores = {}
    for penalty in penalties:
        km = PenalizedKMeansDP(n_clusters=2, penalty=penalty).fit(df_full[feature_columns_kmeans].dropna().values)
        full_labels = km.labels_
        df_full["Market_Regime"] = full_labels

        if df_full["Market_Regime"].nunique() == 1:
            continue

        score = adjusted_rand_score(best_train_labels.loc[df_train[feature_columns_kmeans].dropna().index], full_labels[:len(best_train_labels)])
        similarity_scores[penalty] = score

    best_penalty_final = max(similarity_scores, key=similarity_scores.get)
    print(f"✅ Final chosen penalty: {best_penalty_final}, based on training pattern similarity.")
    print(f"✅ Final chosen xgb params: {best_xgb_params}")

    final_kmeans = PenalizedKMeansDP(n_clusters=2, penalty=best_penalty_final).fit(df_full[feature_columns_kmeans].dropna().values)
    df.loc[full_train_mask, "Market_Regime"] = final_kmeans.labels_

    regime_mean_final = df.loc[full_train_mask].groupby("Market_Regime")[f"{asset}_returns"].mean().sort_values()
    final_map = {regime_mean_final.index[0]: "Bearish", regime_mean_final.index[1]: "Bullish"}
    df.loc[full_train_mask, "Market_Regime"] = df["Market_Regime"].map(final_map)
    df.loc[full_train_mask, "Market_Regime_Num"] = df["Market_Regime"].map({"Bullish": 1, "Bearish": 0})

    y_final = df.loc[full_train_mask, "Market_Regime_Num"].shift(-1).dropna()
    X_final = df.loc[y_final.index, feature_columns_xgb_used].dropna()
    y_final = y_final.loc[X_final.index]

    final_model = xgb.XGBClassifier(**best_xgb_params)
    final_model.fit(X_final.values, y_final)

    X_pred = df.loc[future_mask, feature_columns_xgb_used].dropna()
    if not X_pred.empty:
        df.loc[X_pred.index, "Regime_Forecast"] = final_model.predict(X_pred.values)

    return df




def information_coefficient(predictions, actual_returns):
    return np.corrcoef(predictions, actual_returns)[0, 1] if len(predictions) > 1 else np.nan

def sharpe_ratio(returns):
    """计算 Sharpe Ratio"""
    return np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) != 0 else 0

def cumulative_return(returns):
    """计算累积收益 (Cumulative Return)"""
    return (1 + returns).cumprod().iloc[-1] - 1 if not returns.isna().all() else -np.inf


