# 1. 导入库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
sns.set(font='SimHei')

# 2. 加载数据
df = pd.read_csv("G:/zuoye/archive/adult.csv")
# 替换缺失值（?为缺失标记）
df = df.replace('?', np.nan).dropna()
# 标签编码
df['income'] = df['income'].map({'<=50K': 0, '>50K': 1})

# 3. 划分特征与标签
X = df.drop('income', axis=1)
y = df['income']
# 划分训练集测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# 4. 特征预处理
cat_features = ['workclass', 'education', 'marital.status', 'occupation', 'relationship', 'race', 'sex', 'native.country']
num_features = ['age', 'fnlwgt', 'education.num', 'capital.gain', 'capital.loss', 'hours.per.week']
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
    ])

# 5. 模型定义与超参数网格
# 逻辑回归
lr_pipe = Pipeline(steps=[('pre', preprocessor), ('model', LogisticRegression(random_state=42, max_iter=1000))])
lr_param = {'model__C': [0.1, 1, 10]}
# 随机森林
rf_pipe = Pipeline(steps=[('pre', preprocessor), ('model', RandomForestClassifier(random_state=42, n_jobs=-1))])
rf_param = {'model__n_estimators': [100, 200], 'model__max_depth': [10, 20]}
# XGBoost
xgb_pipe = Pipeline(steps=[('pre', preprocessor), ('model', xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss', tree_method='gpu_hist'))])
xgb_param = {'model__learning_rate': [0.01, 0.1], 'model__max_depth': [3, 6]}

# 6. 网格搜索+交叉验证
def train_model(pipe, param, name):
    grid = GridSearchCV(pipe, param, cv=5, scoring='roc_auc', n_jobs=-1)
    grid.fit(X_train, y_train)
    y_pred = grid.predict(X_test)
    y_prob = grid.predict_proba(X_test)[:,1]
    # 评估指标
    res = {
        '模型': name,
        '准确率': round(accuracy_score(y_test, y_pred),3),
        '精确率': round(precision_score(y_test, y_pred),3),
        '召回率': round(recall_score(y_test, y_pred),3),
        'F1': round(f1_score(y_test, y_pred),3),
        'AUC': round(roc_auc_score(y_test, y_prob),3)
    }
    return res, grid, y_pred, y_prob

# 训练三个模型
lr_res, lr_grid, lr_pred, lr_prob = train_model(lr_pipe, lr_param, '逻辑回归')
rf_res, rf_grid, rf_pred, rf_prob = train_model(rf_pipe, rf_param, '随机森林')
xgb_res, xgb_grid, xgb_pred, xgb_prob = train_model(xgb_pipe, xgb_param, 'XGBoost')

# 输出结果
result = pd.DataFrame([lr_res, rf_res, xgb_res])
print(result)

# 7. 可视化：ROC曲线
plt.figure(figsize=(10,6))
models = [('逻辑回归', lr_prob), ('随机森林', rf_prob), ('XGBoost', xgb_prob)]
for name, prob in models:
    fpr, tpr, _ = roc_curve(y_test, prob)
    plt.plot(fpr, tpr, label=f'{name} (AUC={round(roc_auc_score(y_test, prob),3)})')
plt.plot([0,1],[0,1],'k--')
plt.xlabel('FPR'), plt.ylabel('TPR'), plt.title('ROC曲线'), plt.legend()
plt.show()

# 8. 混淆矩阵热力图
cm = confusion_matrix(y_test, xgb_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('XGBoost混淆矩阵')
plt.show()