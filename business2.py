import os

# 将工作目录切换为脚本所在目录
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, roc_curve, accuracy_score
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

df = pd.read_excel('Project Dataset.xlsx', sheet_name='E Comm')

print(f"{df.shape}")
print(df.head())
print(df.columns.tolist())

df.columns = df.columns.str.replace(' ', '')

print(df.isnull().sum())

# 填充
categorical_cols = ['PreferredLoginDevice', 'Gender', 'MaritalStatus', 'PreferedOrderCat']
for col in categorical_cols:
    if col in df.columns and df[col].dtype == 'object':
        df[col] = df[col].fillna('Unknown')

numeric_cols = ['SatisfactionScore', 'HourSpendOnApp', 'WarehouseToHome', 
               'OrderAmountHikeFromlastYear', 'CouponUsed', 'OrderCount',
               'DaySinceLastOrder', 'DiscountAmount', 'Tenure', 'NumberOfStreamerFollowed']

for col in numeric_cols:
    if col in df.columns:
        if df[col].isnull().sum() > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)

print(df.isnull().sum().sum())


plt.figure(figsize=(15, 12))

#饼图
plt.subplot(3, 3, 1)
churn_counts = df['Churn'].value_counts()
colors = ['lightblue', 'lightcoral']
plt.pie(churn_counts, labels=['未流失', '流失'], autopct='%1.1f%%', colors=colors, startangle=90)
plt.title('流失率分布')

#分特征的柱状图
features_to_analyze = ['PreferredLoginDevice', 'Gender', 'MaritalStatus', 
                       'CityTier', 'PreferedOrderCat', 'SatisfactionScore',
                       'Tenure', 'NumberOfStreamerFollowed', 'Complain']

for i, feature in enumerate(features_to_analyze[:8], 2):
    plt.subplot(3, 3, i)
    if df[feature].dtype == 'object' or df[feature].nunique() < 10:
        # 分类变量或低基数数值变量
        churn_rate = df.groupby(feature)['Churn'].mean().sort_values()
        colors = plt.cm.Blues(np.linspace(0.4, 0.8, len(churn_rate)))
        churn_rate.plot(kind='bar', color=colors)
        plt.axhline(y=df['Churn'].mean(), color='red', linestyle='--', alpha=0.7, label='平均流失率')
        plt.legend()
    else:
        # 连续数值变量
        sns.boxplot(x='Churn', y=feature, data=df)
    plt.title(f'{feature} vs 流失')
    plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig('探索性分析.png', dpi=300, bbox_inches='tight')
plt.show()

# 特征工作
df['AvgDiscountPerOrder'] = df['DiscountAmount'] / (df['OrderCount'] + 1)  # 防止除以0
df['CouponUsageRate'] = df['CouponUsed'] / (df['OrderCount'] + 1)
df['IsHighSpender'] = (df['DiscountAmount'] > df['DiscountAmount'].quantile(0.75)).astype(int)

label_encoders = {}
for col in categorical_cols:
    if col in df.columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

# 训练模型
features = ['PreferredLoginDevice', 'Gender', 'MaritalStatus', 'CityTier', 
           'WarehouseToHome', 'AgeGroup', 'PreferedOrderCat', 'SatisfactionScore',
           'NumberOfStreamerFollowed', 'Complain', 'OrderAmountHikeFromlastYear',
           'CouponUsed', 'OrderCount', 'DaySinceLastOrder', 'DiscountAmount',
           'Tenure', 'HourSpendOnApp', 'AvgDiscountPerOrder', 'CouponUsageRate',
           'IsHighSpender']

X = df[features]
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=666, stratify=y)

print(f"训练集: {X_train.shape}, 测试集: {X_test.shape}")
print(f"训练集流失率: {y_train.mean():.2%}, 测试集流失率: {y_test.mean():.2%}")

# 标准化数值特征
scaler = StandardScaler()
numeric_features = ['WarehouseToHome', 'OrderAmountHikeFromlastYear', 'CouponUsed',
                   'OrderCount', 'DaySinceLastOrder', 'DiscountAmount', 'Tenure',
                   'HourSpendOnApp', 'AvgDiscountPerOrder', 'CouponUsageRate']

X_train[numeric_features] = scaler.fit_transform(X_train[numeric_features])
X_test[numeric_features] = scaler.transform(X_test[numeric_features])


models = {
    'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
    'RandomForest': RandomForestClassifier(random_state=42),
    'GradientBoosting': GradientBoostingClassifier(random_state=42),
    'LightGBM': lgb.LGBMClassifier(random_state=42)
}

results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    auc = roc_auc_score(y_test, y_pred_proba)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    results.append({
        'Model': name,
        'AUC': auc,
        'Accuracy': accuracy,
        'Precision': report['weighted avg']['precision'],
        'Recall': report['weighted avg']['recall'],
        'F1': report['weighted avg']['f1-score']
    })
    
    print(f"\n{name} 性能:")
    print(f"AUC: {auc:.4f}")
    print(f"准确率: {accuracy:.4f}")
    print(classification_report(y_test, y_pred))

# 结果比较
results_df = pd.DataFrame(results).sort_values('AUC', ascending=False)
print("\n模型性能比较:")
print(results_df)

# 选择最佳模型
best_model_name = results_df.iloc[0]['Model']
best_model = models[best_model_name]

if best_model_name == 'LightGBM':
    
    param_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'num_leaves': [15, 31, 63]
    }
    
    grid_search = GridSearchCV(
        lgb.LGBMClassifier(random_state=42),
        param_grid,
        cv=3,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"最佳参数: {grid_search.best_params_}")
    print(f"最佳AUC: {grid_search.best_score_:.4f}")
    
    best_model = grid_search.best_estimator_
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]
    optimized_auc = roc_auc_score(y_test, y_pred_proba)
    print(f"优化后测试集AUC: {optimized_auc:.4f}")

# 特征重要性
if hasattr(best_model, 'feature_importances_'):
    feature_importance = pd.DataFrame({
        'Feature': features,
        'Importance': best_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("\n特征重要性排序:")
    print(feature_importance.head(10))
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importance.head(15))
    plt.title('Top 15 Feature Importance')
    plt.tight_layout()
    plt.savefig('特征重要性.png', dpi=300, bbox_inches='tight')
    plt.show()

# 统计信息
top_features = feature_importance.head(5)['Feature'].tolist()
for feature in top_features:
    if feature in df.columns:
        if df[feature].dtype == 'object' or df[feature].nunique() < 10:
            # 分类变量
            churn_rate = df.groupby(feature)['Churn'].mean()
            print(f"\n{feature} 流失率分布:")
            print(churn_rate)
        else:
            # 数值变量
            print(f"\n{feature} 统计信息:")
            print(f"均值: {df[feature].mean():.2f}")
            print(f"中位数: {df[feature].median():.2f}")
            print(f"流失用户均值: {df[df['Churn']==1][feature].mean():.2f}")
            print(f"非流失用户均值: {df[df['Churn']==0][feature].mean():.2f}")

print("\n已保存")