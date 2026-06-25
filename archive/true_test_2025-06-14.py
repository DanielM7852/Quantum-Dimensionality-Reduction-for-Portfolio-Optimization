import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.decomposition import PCA, KernelPCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

CONFIG = {
    'ticker': 'NVDA',
    'start_date': '2019-01-01',
    'end_date': '2024-11-01',
    'target_days': 5,
    'n_components': 8,
    'cv_splits': 5,
    'random_state': 42,
}

def download_data(ticker, start, end):
    print("Downloading " + ticker + "...")
    data = yf.download(ticker, start=start, end=end, progress=False)
    print("   Downloaded " + str(len(data)) + " days")
    return data

def create_features(df):
    print("Creating features...")
    df = df.copy()
    df['returns_5'] = df['Close'].pct_change(5).fillna(0)
    df['returns_10'] = df['Close'].pct_change(10).fillna(0)
    df['volatility'] = df['Close'].pct_change().rolling(20).std().fillna(0)
    for period in [10, 20, 50]:
        ma = df['Close'].rolling(period).mean()
        df['ma_ratio_' + str(period)] = (df['Close'] / (ma + 1e-10)).fillna(1.0)
    delta = df['Close'].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    df['rsi'] = (100 - (100 / (1 + rs))).fillna(50)
    vol_ma = df['Volume'].rolling(20).mean()
    df['volume_ratio'] = (df['Volume'] / (vol_ma + 1)).fillna(1.0)
    df = df.replace([np.inf, -np.inf], 0)
    df = df.fillna(0)
    print("   Created " + str(len(df)) + " samples with " + str(df.shape[1]) + " features")
    return df

def prepare_targets(df, target_days=5):
    print("Creating targets...")
    df = df.copy().reset_index(drop=True)
    future_prices = df['Close'].shift(-target_days)
    current_prices = df['Close']
    future_returns = (future_prices / current_prices) - 1
    targets = (future_returns > 0).astype(float)
    valid_idx = ~future_returns.isna()
    df_valid = df[valid_idx].copy()
    targets_valid = targets[valid_idx].copy()
    returns_valid = future_returns[valid_idx].copy()
    targets_valid = np.nan_to_num(targets_valid, nan=0).astype(int)
    returns_valid = np.nan_to_num(returns_valid, nan=0)
    dropped = (~valid_idx).sum()
    valid_count = len(df_valid)
    pos_class = targets_valid.mean() * 100
    print("   Valid samples: " + str(valid_count) + " (dropped " + str(dropped) + ")")
    print("   Positive class: " + str(round(pos_class, 1)) + "%")
    return df_valid, targets_valid, returns_valid

def extract_features(df):
    exclude = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    features = [c for c in df.columns if c not in exclude]
    X = df[features].values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X, features

class PCAMethod:
    def __init__(self, n_components=8):
        self.n_components = n_components
        self.pca = PCA(n_components=n_components)
        self.name = "PCA"
    def fit_transform(self, X, y=None):
        return np.nan_to_num(self.pca.fit_transform(X), nan=0.0)
    def transform(self, X):
        return np.nan_to_num(self.pca.transform(X), nan=0.0)

class KernelPCAMethod:
    def __init__(self, n_components=8):
        self.n_components = n_components
        self.kpca = KernelPCA(n_components=n_components, kernel='rbf', gamma=0.01)
        self.name = "Kernel-PCA"
    def fit_transform(self, X, y=None):
        try:
            return np.nan_to_num(self.kpca.fit_transform(X), nan=0.0)
        except:
            return np.zeros((X.shape[0], self.n_components))
    def transform(self, X):
        try:
            return np.nan_to_num(self.kpca.transform(X), nan=0.0)
        except:
            return np.zeros((X.shape[0], self.n_components))

class QuantumInspiredMethod:
    def __init__(self, n_components=8):
        self.n_components = n_components
        self.name = "Quantum-Inspired"
    def fit_transform(self, X, y=None):
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0) + 1e-10
        X_std = (X - self.mean) / self.std
        X_std = np.nan_to_num(X_std, nan=0.0)
        corr = np.corrcoef(X_std.T)
        corr = np.nan_to_num(corr, nan=0.0)
        entanglement = np.abs(corr)
        transformation = np.eye(X.shape[1])
        for _ in range(3):
            for i in range(X.shape[1]-1):
                for j in range(i+1, X.shape[1]):
                    if entanglement[i, j] > 0.7:
                        theta = (np.pi / 36) * entanglement[i, j]
                        rotation = np.eye(X.shape[1])
                        rotation[i, i] = np.cos(theta)
                        rotation[i, j] = -np.sin(theta)
                        rotation[j, i] = np.sin(theta)
                        rotation[j, j] = np.cos(theta)
                        transformation = rotation @ transformation
        X_transformed = X_std @ transformation.T
        X_transformed = np.nan_to_num(X_transformed, nan=0.0)
        variances = np.var(X_transformed, axis=0)
        top_indices = np.argsort(variances)[-self.n_components:][::-1]
        self.transformation = transformation
        self.selected_indices = top_indices
        return X_transformed[:, top_indices]
    def transform(self, X):
        X_std = (X - self.mean) / self.std
        X_std = np.nan_to_num(X_std, nan=0.0)
        X_transformed = X_std @ self.transformation.T
        X_transformed = np.nan_to_num(X_transformed, nan=0.0)
        return X_transformed[:, self.selected_indices]

class RidgeTrader:
    def __init__(self):
        self.name = "Ridge"
        self.model = LogisticRegression(C=0.1, max_iter=1000, random_state=42)
    def train(self, X, y):
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        y = np.nan_to_num(y, nan=0).astype(int)
        self.model.fit(X, y)
    def predict(self, X):
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        return self.model.predict(X)

class RandomForestTrader:
    def __init__(self):
        self.name = "Random-Forest"
        self.model = RandomForestClassifier(n_estimators=50, max_depth=4, min_samples_split=20, min_samples_leaf=10, random_state=42)
    def train(self, X, y):
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        y = np.nan_to_num(y, nan=0).astype(int)
        self.model.fit(X, y)
    def predict(self, X):
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        return self.model.predict(X)

class GradientBoostingTrader:
    def __init__(self):
        self.name = "Gradient-Boosting"
        self.model = GradientBoostingClassifier(n_estimators=30, max_depth=2, learning_rate=0.05, min_samples_split=20, random_state=42)
    def train(self, X, y):
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        y = np.nan_to_num(y, nan=0).astype(int)
        self.model.fit(X, y)
    def predict(self, X):
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        return self.model.predict(X)

def calculate_sharpe(returns):
    if len(returns) == 0 or np.std(returns) == 0:
        return 0.0
    return (np.mean(returns) / np.std(returns)) * np.sqrt(252)

def evaluate_cv(X, y, returns, dr_method, trading_model, cv_splits=5):
    tscv = TimeSeriesSplit(n_splits=cv_splits)
    train_accs, test_accs, test_f1s, test_sharpes, test_rets = [], [], [], [], []
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        if len(train_idx) < 100 or len(test_idx) < 20:
            continue
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        y_train = np.nan_to_num(y_train, nan=0).astype(int)
        y_test = np.nan_to_num(y_test, nan=0).astype(int)
        ret_test = np.nan_to_num(returns[test_idx], nan=0)
        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_test_sc = scaler.transform(X_test)
        try:
            X_train_dr = dr_method.fit_transform(X_train_sc, y_train)
            X_test_dr = dr_method.transform(X_test_sc)
        except:
            continue
        trading_model.train(X_train_dr, y_train)
        y_train_pred = trading_model.predict(X_train_dr)
        y_test_pred = trading_model.predict(X_test_dr)
        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)
        test_f1 = f1_score(y_test, y_test_pred, zero_division=0)
        strat_ret = np.where(y_test_pred == 1, ret_test, 0)
        test_sharpe = calculate_sharpe(strat_ret)
        test_ret = np.sum(strat_ret)
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        test_f1s.append(test_f1)
        test_sharpes.append(test_sharpe)
        test_rets.append(test_ret)
    if not test_accs:
        return None
    return {
        'DR_Method': dr_method.name,
        'Trading_Model': trading_model.name,
        'train_accuracy': np.mean(train_accs),
        'accuracy': np.mean(test_accs),
        'overfit_gap': np.mean(train_accs) - np.mean(test_accs),
        'f1': np.mean(test_f1s),
        'sharpe_ratio': np.mean(test_sharpes),
        'cumulative_return': np.mean(test_rets),
    }

def create_dashboard(results_df, save_path='dashboard_final.png'):
    sns.set_style("whitegrid")
    plt.rcParams['font.size'] = 10
    fig = plt.figure(figsize=(22, 13))
    fig.patch.set_facecolor('#fafafa')
    gs = fig.add_gridspec(3, 3, hspace=0.45, wspace=0.35, left=0.06, right=0.96, top=0.92, bottom=0.06)

    # Accuracy
    ax1 = fig.add_subplot(gs[0, 0])
    pivot = results_df.pivot_table(index='Trading_Model', columns='DR_Method', values='accuracy')
    pivot.plot(kind='bar', ax=ax1, width=0.65, edgecolor='black', linewidth=1.2)
    ax1.axhline(0.50, color='red', linestyle='--', linewidth=2, alpha=0.6)
    ax1.set_title('TEST ACCURACY\n(Higher = Better Predictions)', fontweight='bold', pad=10, fontsize=11)
    ax1.set_ylabel('Accuracy', fontweight='bold')
    ax1.set_xlabel('')
    ax1.set_ylim([0.40, 0.80])
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=0, fontsize=9)
    ax1.legend(fontsize=8, loc='upper left')
    ax1.grid(True, alpha=0.3)

    # Sharpe
    ax2 = fig.add_subplot(gs[0, 1])
    pivot = results_df.pivot_table(index='Trading_Model', columns='DR_Method', values='sharpe_ratio')
    pivot.plot(kind='bar', ax=ax2, width=0.65, edgecolor='black', linewidth=1.2, colormap='viridis')
    ax2.axhline(1.0, color='orange', linestyle='--', linewidth=2, alpha=0.6, label='Good')
    ax2.axhline(2.0, color='green', linestyle='--', linewidth=2, alpha=0.6, label='Excellent')
    ax2.set_title('SHARPE RATIO\n(Risk-Adjusted Returns)', fontweight='bold', pad=10, fontsize=11)
    ax2.set_ylabel('Sharpe Ratio', fontweight='bold')
    ax2.set_xlabel('')
    ax2.set_ylim([0, 3.5])
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=0, fontsize=9)
    ax2.legend(fontsize=8, loc='upper left')
    ax2.grid(True, alpha=0.3)

    # Returns
    ax3 = fig.add_subplot(gs[0, 2])
    pivot = results_df.pivot_table(index='Trading_Model', columns='DR_Method', values='cumulative_return')
    pivot.plot(kind='bar', ax=ax3, width=0.65, edgecolor='black', linewidth=1.2, colormap='RdYlGn')
    ax3.axhline(0, color='black', linestyle='-', linewidth=1.5)
    ax3.set_title('PROFITABILITY\n(Total Profit Over Test Period)', fontweight='bold', pad=10, fontsize=11)
    ax3.set_ylabel('Cumulative Return', fontweight='bold')
    ax3.set_xlabel('')
    ax3.set_xticklabels(ax3.get_xticklabels(), rotation=0, fontsize=9)
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: str(int(y * 100)) + '%'))
    ax3.legend(fontsize=8, loc='upper left')
    ax3.grid(True, alpha=0.3)

    # Overfitting - FIXED: Color-coded by DR method
    ax4 = fig.add_subplot(gs[1, :2])
    pivot = results_df.pivot_table(index='Trading_Model', columns='DR_Method', values='overfit_gap')
    x = np.arange(len(pivot.index))
    width = 0.25
    
    # Define distinct colors for each DR method (colorblind-friendly palette)
    dr_method_colors = {
        'PCA': '#1f77b4',              # Blue
        'Kernel-PCA': '#ff7f0e',       # Orange
        'Quantum-Inspired': '#2ca02c'  # Green
    }
    
    for i, col in enumerate(pivot.columns):
        offset = width * (i - 1)
        base_color = dr_method_colors.get(col, '#7f7f7f')  # Default gray if method not found
        ax4.bar(x + offset, pivot[col], width, label=col, color=base_color, edgecolor='black', linewidth=1.5, alpha=0.85)
    
    ax4.axhline(0.05, color='orange', linestyle='--', linewidth=2, alpha=0.6, label='Warning (5%)')
    ax4.axhline(0.10, color='red', linestyle='--', linewidth=2, alpha=0.6, label='Severe (10%)')
    ax4.set_title('OVERFITTING GAP ANALYSIS\n(Train Accuracy - Test Accuracy)', fontweight='bold', pad=10, fontsize=11)
    ax4.set_ylabel('Gap', fontweight='bold')
    ax4.set_xlabel('Trading Model', fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(pivot.index, fontsize=9)
    ax4.legend(fontsize=8, loc='upper left', ncol=2)
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.set_ylim([0, max(0.25, pivot.max().max() * 1.1)])

    # FIXED: Heatmap with objective normalization
    ax5 = fig.add_subplot(gs[1, 2])
    summary = results_df.groupby('DR_Method')[['accuracy', 'f1', 'sharpe_ratio', 'cumulative_return']].mean()
    
    # Define objective performance thresholds (based on financial trading literature)
    thresholds = {
        'accuracy': {'poor': 0.50, 'excellent': 0.70},
        'f1': {'poor': 0.50, 'excellent': 0.70},
        'sharpe_ratio': {'poor': 0.0, 'excellent': 2.0},
        'cumulative_return': {'poor': 0.0, 'excellent': 0.20}
    }
    
    # Normalize against objective thresholds (not relative min/max)
    summary_norm = pd.DataFrame(index=summary.index, columns=summary.columns)
    for metric in summary.columns:
        poor = thresholds[metric]['poor']
        excellent = thresholds[metric]['excellent']
        summary_norm[metric] = (summary[metric] - poor) / (excellent - poor)
        summary_norm[metric] = summary_norm[metric].clip(0, 1)
    
    sns.heatmap(summary_norm.T, annot=True, fmt='.2f', cmap='RdYlGn', ax=ax5, 
                linewidths=2, linecolor='white', vmin=0, vmax=1, 
                cbar_kws={'label': 'Normalized Performance'})
    ax5.set_title('PERFORMANCE MATRIX\n(0 = Poor, 1 = Excellent)', fontweight='bold', pad=10, fontsize=11)
    ax5.set_xlabel('DR Method', fontweight='bold')
    ax5.set_ylabel('Metric', fontweight='bold')
    ax5.set_yticklabels(['Accuracy', 'F1-Score', 'Sharpe', 'Return'], rotation=0, fontsize=9)

    # Overfitting Check Panel
    ax6 = fig.add_subplot(gs[2, 2])
    ax6.set_facecolor('#fff3cd')
    ax6.axis('off')
    ax6.text(0.5, 0.95, 'OVERFITTING CHECK', ha='center', va='top', fontsize=14, fontweight='bold',
            bbox=dict(boxstyle='round,pad=1', facecolor='orange', edgecolor='darkorange', linewidth=3),
            transform=ax6.transAxes)
    
    y_pos = 0.75
    overfitting = results_df[results_df['overfit_gap'] > 0.1]
    warning = results_df[(results_df['overfit_gap'] > 0.05) & (results_df['overfit_gap'] <= 0.1)]
    
    if len(overfitting) > 0:
        ax6.text(0.5, y_pos, 'SEVERE OVERFITTING DETECTED', ha='center', va='top', fontsize=11, 
                fontweight='bold', color='darkred', transform=ax6.transAxes)
        y_pos -= 0.10
        for _, row in overfitting.iterrows():
            text = row['DR_Method'] + " + " + row['Trading_Model'] + "\nGap: " + str(round(row['overfit_gap']*100, 1)) + "%"
            ax6.text(0.5, y_pos, text, ha='center', va='top', fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='#ffcccc', edgecolor='red', linewidth=2),
                    transform=ax6.transAxes)
            y_pos -= 0.15
    
    if len(warning) > 0:
        ax6.text(0.5, y_pos, 'MODERATE RISK', ha='center', va='top', fontsize=10, 
                fontweight='bold', color='darkorange', transform=ax6.transAxes)
        y_pos -= 0.10
        for _, row in warning.iterrows():
            text = row['DR_Method'] + " + " + row['Trading_Model'] + "\nGap: " + str(round(row['overfit_gap']*100, 1)) + "%"
            ax6.text(0.5, y_pos, text, ha='center', va='top', fontsize=8,
                    bbox=dict(boxstyle='round,pad=0.4', facecolor='#fff3cd', edgecolor='orange', linewidth=2),
                    transform=ax6.transAxes)
            y_pos -= 0.13

    # Recommendation
    ax7 = fig.add_subplot(gs[2, :2])
    ax7.set_facecolor('#e7f3ff')
    ax7.axis('off')
    
    safe_methods = results_df[results_df['overfit_gap'] <= 0.10]
    if len(safe_methods) == 0:
        text = "ALL METHODS OVERFITTING - DO NOT USE"
        ax7.text(0.5, 0.5, text, ha='center', va='center', fontsize=16, fontweight='bold',
                bbox=dict(boxstyle='round,pad=1.5', facecolor='red', alpha=0.8), color='white')
    else:
        best = safe_methods.loc[safe_methods['sharpe_ratio'].idxmax()]
        rec_text = "RECOMMENDED STRATEGY: " + best['DR_Method'] + " + " + best['Trading_Model']
        ax7.text(0.5, 0.80, rec_text, ha='center', va='center', fontsize=14, fontweight='bold',
                bbox=dict(boxstyle='round,pad=1', facecolor='navy', alpha=0.9), color='white')
        
        acc = str(round(best['accuracy']*100, 1)) + "%"
        sharpe = str(round(best['sharpe_ratio'], 2))
        ret = str(round(best['cumulative_return']*100, 1)) + "%"
        gap = str(round(best['overfit_gap']*100, 2)) + "%"
        
        metrics = "Test Accuracy: " + acc + " | Sharpe Ratio: " + sharpe + " | Return: " + ret + " | Overfit Gap: " + gap
        ax7.text(0.5, 0.55, metrics, ha='center', va='center', fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.8', facecolor='white', edgecolor='gray', linewidth=2))
        
        if best['sharpe_ratio'] > 1.5 and best['overfit_gap'] < 0.03:
            trust = "Trust Level:\nHIGH CONFIDENCE"
            trust_color = '#28a745'
            action = "Action:\nRECOMMENDED FOR LIVE TRADING"
            action_color = 'lightgreen'
        elif best['sharpe_ratio'] > 1.0 and best['overfit_gap'] < 0.05:
            trust = "Trust Level:\nMODERATE CONFIDENCE"
            trust_color = '#ffc107'
            action = "Action:\nPAPER TRADE FIRST"
            action_color = 'lightyellow'
        else:
            trust = "Trust Level:\nLOW CONFIDENCE"
            trust_color = '#dc3545'
            action = "Action:\nNOT RECOMMENDED"
            action_color = 'lightcoral'
        
        ax7.text(0.25, 0.25, trust, ha='center', va='center', fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.8', facecolor=trust_color, edgecolor='black', linewidth=2, alpha=0.8))
        ax7.text(0.75, 0.25, action, ha='center', va='center', fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.8', facecolor=action_color, edgecolor='black', linewidth=2, alpha=0.8))

    plt.suptitle('ALGORITHMIC TRADING STRATEGY ANALYSIS DASHBOARD', fontsize=18, fontweight='bold', y=0.97)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='#fafafa')
    plt.close()
    print("\nDashboard saved: " + save_path)

def main():
    print("="*80)
    print("QUANTUM TRADING - COLOR-CODED VERSION")
    print("="*80)
    data = download_data(CONFIG['ticker'], CONFIG['start_date'], CONFIG['end_date'])
    data = create_features(data)
    data_valid, y, returns = prepare_targets(data, CONFIG['target_days'])
    X, feature_names = extract_features(data_valid)
    print("\nDataset: " + str(X.shape[0]) + " samples, " + str(X.shape[1]) + " features")
    
    dr_methods = [
        PCAMethod(CONFIG['n_components']),
        KernelPCAMethod(CONFIG['n_components']),
        QuantumInspiredMethod(CONFIG['n_components']),
    ]
    
    trading_models = [
        RidgeTrader(),
        RandomForestTrader(),
        GradientBoostingTrader(),
    ]
    
    print("\n[EVALUATION]")
    print("-"*80)
    results = []
    for dr in dr_methods:
        for trader in trading_models:
            print("\n" + dr.name + " + " + trader.name + ":")
            metrics = evaluate_cv(X, y, returns, dr, trader, CONFIG['cv_splits'])
            if metrics:
                results.append(metrics)
                acc = str(round(metrics['accuracy']*100, 1)) + "%"
                gap = str(round(metrics['overfit_gap']*100, 2)) + "%"
                print("  Accuracy: " + acc + ", Gap: " + gap)
    
    results_df = pd.DataFrame(results)
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    print(results_df.to_string(index=False))
    
    results_df.to_csv('results_final.csv', index=False)
    print("\nSaved: results_final.csv")
    
    create_dashboard(results_df, 'dashboard_final.png')
    print("\nCOMPLETE - Ready for publication!")

if __name__ == "__main__":
    main()