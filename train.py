# coding:utf-8
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import ensemble, metrics
from datetime import datetime as dt
from sklearn.preprocessing import LabelEncoder

# pandas
import pandas as pd

pd.set_option("display.max_rows", 500)
pd.set_option("display.max_colwidth", 100)

# LightGBM
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 交差検証
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error

# AUC
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

purchase_df = pd.read_csv('../input/purchase_record.csv')
user_df = pd.read_csv('../input/user_info.csv')
test_df = pd.read_csv('../input/purchase_record_test.csv')

purchase_df.shape

purchase_df.info()

purchase_df.head()

purchase_df.fillna(0, inplace=True)

purchase_df.purchase.value_counts()

user_df.shape

user_df.info()

user_df.head()

train_df = pd.merge(purchase_df, user_df, how='left', on='user_id')

train_df.head()

# train_dfから不要列を削除する
# train_df = train_df.drop('date_x', axis=1)
# train_df = train_df.drop('date_y', axis=1)
# train_df = train_df.drop('product_id', axis=1)

# product_idをfrequency_encodingする
grouped_product_id = train_df.groupby("product_id").size().reset_index(name='product_id_counts')  # 出現回数を計算
# もとのデータセットにカテゴリーをproduct_idとして結合
train_df = train_df.merge(grouped_product_id, how="left", on="product_id")
train_df["product_id_frequency"] = train_df["product_id_counts"] / train_df["product_id_counts"].count()
# product_idを削除する
# train_df = train_df.drop('product_id_counts', axis=1)

# ユーザーの出現回数(検討回数)列をfrequency_encodingで追加
grouped_user_id = train_df.groupby("user_id").size().reset_index(name='user_id_counts')  # 出現回数を計算
# もとのデータセットにカテゴリーをuser_idとして結合
train_df = train_df.merge(grouped_user_id, how="left", on="user_id")
train_df["user_id_frequency"] = train_df["user_id_counts"] / train_df["user_id_counts"].count()
# user_idを削除する
train_df = train_df.drop('user_id', axis=1)
# train_df = train_df.drop('product_id_counts', axis=1)

# データ開始から何日経っているかを表す列を追加date_x
train_df['date_x_days'] = np.nan  # 欠損値で埋める
train_df['date_x_days'] = train_df.index.values  # インデックス値を使用
train_df['date_x_days'] = pd.to_datetime(train_df['date_x'])  # 日付データにリフォーマット
train_df['date_x_days'] = train_df['date_x_days'] - dt(2015, 1, 1)
train_df['date_x_days'] = train_df['date_x_days'].dt.days
# データ開始から何日経っているかを表す列を追加date_y
train_df['date_y_days'] = np.nan  # 欠損値で埋める
train_df['date_y_days'] = train_df.index.values  # インデックス値を使用
train_df['date_y_days'] = pd.to_datetime(train_df['date_y'])  # 日付データにリフォーマット
train_df['date_y_days'] = train_df['date_y_days'] - dt(2015, 1, 1)
train_df['date_y_days'] = train_df['date_y_days'].dt.days
# 日付列削除
train_df = train_df.drop('date_x', axis=1)
train_df = train_df.drop('date_y', axis=1)

# 顧客属性登録日から購入検討日までの検討時間列を追加
train_df['days_y-x'] = np.nan  # 欠損値で埋める
train_df['days_y-x'] = train_df.index.values  # インデックス値を使用
train_df['days_y-x'] = train_df['date_x_days'] - train_df['date_y_days']

train_df.head()

train_df.info()

train_df.shape

train_df.isnull().sum().sum()

#
# pdp.ProfileReport(train_df)
test_df = pd.merge(test_df, user_df, how='left', on='user_id')

test_df.shape

# sample_train_df = train_df.sample(n=len(train_df)//500)

# sample_test_df = test_df.sample(n=len(test_df)//500)

# train_df = sample_train_df
# test_df = sample_test_df

# test_dfから不要列を削除する
# test_df = test_df.drop('product_id', axis=1)

# product_idをfrequency_encodingする
grouped_product_id = test_df.groupby("product_id").size().reset_index(name='product_id_counts')  # 出現回数を計算
# もとのデータセットにカテゴリーをproduct_idとして結合
test_df = test_df.merge(grouped_product_id, how="left", on="product_id")
test_df["product_id_frequency"] = test_df["product_id_counts"] / test_df["product_id_counts"].count()
# product_idを削除する
# test_df = test_df.drop('product_id_counts', axis=1)

# データ開始から何日経っているかを表す列を追加date_x
test_df['date_x_days'] = np.nan  # 欠損値で埋める
test_df['date_x_days'] = test_df.index.values  # インデックス値を使用
test_df['date_x_days'] = pd.to_datetime(test_df['date_x'])  # 日付データにリフォーマット
test_df['date_x_days'] = test_df['date_x_days'] - dt(2015, 1, 1)
test_df['date_x_days'] = test_df['date_x_days'].dt.days
# データ開始から何日経っているかを表す列を追加date_y
test_df['date_y_days'] = np.nan  # 欠損値で埋める
test_df['date_y_days'] = test_df.index.values  # インデックス値を使用
test_df['date_y_days'] = pd.to_datetime(test_df['date_y'])  # 日付データにリフォーマット
test_df['date_y_days'] = test_df['date_y_days'] - dt(2015, 1, 1)
test_df['date_y_days'] = test_df['date_y_days'].dt.days
# 日付列削除
test_df = test_df.drop('date_x', axis=1)
test_df = test_df.drop('date_y', axis=1)

# 顧客属性登録日から購入検討日までの検討時間列を追加
test_df['days_y-x'] = np.nan  # 欠損値で埋める
test_df['days_y-x'] = test_df.index.values  # インデックス値を使用
test_df['days_y-x'] = test_df['date_x_days'] - test_df['date_y_days']

test_df = test_df.drop('user_id', axis=1)

test_df.head()

train_df.info()

test_df.info()

# train = pd.get_dummies(train_df, drop_first=True)
train = pd.get_dummies(train_df,
                       columns=['product_id', 'attribute_1', 'attribute_2', 'attribute_3', 'parts_1', 'parts_2',
                                'parts_3', 'parts_4', 'parts_5', 'parts_6', 'parts_7', 'parts_8', 'parts_9'],
                       drop_first=True)
train = train * 1
train = train.set_index('purchase_id')
# train = pd.get_dummies(train_df, columns!=['purchase_id', 'purchase'], drop_first=True)

# test = pd.get_dummies(test_df, drop_first=True)
# test = pd.get_dummies(test_df, drop_first=True)
test = pd.get_dummies(test_df, columns=['product_id', 'attribute_1', 'attribute_2', 'attribute_3', 'parts_1', 'parts_2',
                                        'parts_3', 'parts_4', 'parts_5', 'parts_6', 'parts_7', 'parts_8', 'parts_9'],
                      drop_first=True)
test = test * 1
test = test.set_index('purchase_id')

train.head()

train.info()

test.info()

# data_yに目的変数を代入
y_train = train['purchase']

# data_yの表示
print(y_train)

# data_Xに説明変数を代入
X_train = train.drop('purchase', axis=1)

# data_Xの表示
print(X_train)
# 5分割交差検証を指定し、インスタンス化
n_splits = 5
kf = KFold(n_splits, shuffle=True)

# スコアとモデルを格納するリスト
score_list = []
models = []

for fold_, (train_index, valid_index) in enumerate(kf.split(X_train, y_train)):
    print(f'fold{fold_ + 1} start')
    train_x = X_train.iloc[train_index]
    valid_x = X_train.iloc[valid_index]
    train_y = y_train[train_index]
    valid_y = y_train[valid_index]
    # lab.Datasetを使って,trainとvalidを作っておく
    lgb_train = lgb.Dataset(train_x, train_y)
    lgb_valid = lgb.Dataset(valid_x, valid_y)

    # パラメータを定義
    lgbm_params = {'objective': 'binary'}

    # 学習
    gbm = lgb.train(params=lgbm_params,
                    train_set=lgb_train,
                    valid_sets=[lgb_train, lgb_valid],
                    early_stopping_rounds=20,
                    verbose_eval=-1  # 学習の状況を表示しない
                    )

    oof = (gbm.predict(valid_x) > 0.5).astype(int)
    # oof = gbm.predict(valid_x)
    print('oof:', oof)
    # AUC算出
    # auc = metrics.auc(valid_y, oof)
    auc = roc_auc_score(valid_y, oof)
    score_list.append(auc)  # スコアリストに保存
    # score_list.append(np.sqrt(mean_squared_error(valid_y, oof))) #RMSEを出す
    # score_list.append(round(accuracy_score(valid_y, oof)*100,2)) # 正解率Accuracyを出す。(検証用y, 予測結果)
    print('AUC:', score_list)
    models.append(gbm)  # 学習が終わったモデルをリストに入れておく
    print(f'fold{fold_ + 1} end\n')

    # FPR, TPR(, しきい値) を算出
    fpr, tpr, thresholds = metrics.roc_curve(valid_y, oof)

    # ROC曲線をプロット
    plt.plot(fpr, tpr, label='ROC curve (area = %.2f)' % auc)
    plt.legend()
    plt.title('ROC curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.grid(True)

print(score_list, '平均score', round(np.mean(score_list), 2))

# 比較
# p = pd.DataFrame({"actual":valid_y,"pred":oof})
# p.plot(figsize=(15,4))
# print("AUC",round(np.mean(score_list), 2))

# テストデータの予測を格納する。テストデータ列数*分割数列のnumpy行列を作成
test_pred = np.zeros((len(test), n_splits))

# 分割検証結果の平均を取る
for fold_, gbm in enumerate(models):
    pred_ = gbm.predict(test)  # testを予測
    test_pred[:, fold_] = pred_

result = pd.DataFrame({
    'purchase_id': test.index.values,
    'probability': np.mean(test_pred, axis=1)
    # 'probability': (np.mean(test_pred, axis=1) > 0.5).astype(int)
})

# pred = (np.mean(test_pred, axis=1) > 0.5).astype(int)
# result['probability'] = pred
result.to_csv('../output/submit.csv', header=False, index=False)

test_pred

np.set_printoptions(threshold=np.inf)
print(pred_)

import datetime

dt_now = datetime.datetime.now()
print(dt_now)
print("予測完了")

#import slackweb

#slack = slackweb.Slack(url="https://hooks.slack.com/services/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")
#slack.notify(text="PCメーカー　pythonスクリプト予測完了")
