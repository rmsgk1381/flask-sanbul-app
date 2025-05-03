import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1-1 Data 불러오기
fires = pd.read_csv("data/sanbul2district-divby100.csv", sep=",")

# 1-2 fires.head(), fires.info(), fires.describe(), 카테고리형 특성 month, day에 대해 value_counts() 출력하기
print("상위 5행을 출력\n",fires.head())
print("\n데이터에 대한 전반적인 정보")
fires.info()
print("\n데이터의 컬럼별 요약 통계량\n",fires.describe())
print("\nmonth 컬럼 내에 각각의 값이 나온 횟수\n",fires['month'].value_counts())
print("\nday 컬럼 내에 각각의 값이 나온 횟수\n",fires['day'].value_counts())

# 1-3 데이터 시각화
fires.hist(bins=30, figsize=(12, 10), color='skyblue', edgecolor='black')
plt.tight_layout()
plt.show()

# 1-4 특성 burned_area 왜곡 현상(아래 왼쪽 그림) 개선을 위해 로그 함수를 이용한 변환
fires['burned_area'] = np.log(fires['burned_area'] + 1)
plt.figure(figsize=(8, 4))
plt.hist(fires['burned_area'], bins=30, color='orange', edgecolor='black')
plt.title('Distribution of Burned Area (log-transformed)')
plt.xlabel('Burned Area (log)')
plt.ylabel('Frequency')
plt.show()

# 1-5 Scikit-Learn의 train_test_split을 이용하여 training/test set 분리 / Test set 비율 확인하기
from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(fires, test_size=0.2, random_state=42)
test_set.head()
fires["month"].hist()
from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(fires, fires["month"]):
 strat_train_set = fires.loc[train_index]
 strat_test_set = fires.loc[test_index]
print("\nMonth category proportion: \n",
 strat_test_set["month"].value_counts()/len(strat_test_set))
print("\nOverall month category proportion: \n",
 fires["month"].value_counts()/len(fires))

# 1-6 Pandas scatter_matrix() 함수를 이용하여 4개 이상의 특성에 대해 matrix 출력하기
from pandas.plotting import scatter_matrix

attributes = ["burned_area", "max_temp", "avg_temp", "max_wind_speed"]
scatter_matrix(fires[attributes], figsize=(10, 8), diagonal='hist')
plt.show()

# 1-7 지역별로 ‘burned_area’에 대해 plot 하기
fires.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
 s=fires["max_temp"], label="max_temp",
 c="burned_area", cmap=plt.get_cmap("jet"), colorbar=True)

# 1-8 카테고리형 특성 month, day에 대해 OneHotEncoder()를 이용한 인코딩/출력
from sklearn.preprocessing import OneHotEncoder

fires = strat_train_set.drop(["burned_area"], axis=1)
fires_labels = strat_train_set["burned_area"].copy()

fires_test = strat_test_set.drop("burned_area", axis=1)
fires_test_labels = strat_test_set["burned_area"].copy()

fires_num = fires.drop(["month", "day"], axis=1)
fires_cat = fires[["month", "day"]]

cat_encoder = OneHotEncoder()
fires_cat_1hot = cat_encoder.fit_transform(fires_cat)

print("cat_day_encoder.categories_:\n", cat_encoder.categories_[1])  # day
print("\ncat_month_encoder.categories_:\n", cat_encoder.categories_[0])  # month

# 1-9 Scikit-Learn의 Pipeline, StandardScaler를 이용하여 카테고리형 특성을 인코딩한 training set 생성하기
print("\n\n########################################################################")
print("Now let's build a pipeline for preprocessing the numerical attributes:")
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

num_attribs = ["longitude", "latitude", "avg_temp", "max_temp", "max_wind_speed", "avg_wind"]
cat_attribs = ["month", "day"]

num_pipeline = Pipeline([
    ('std_scaler', StandardScaler()),
])

full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", OneHotEncoder(), cat_attribs),
])

fires_prepared = full_pipeline.fit_transform(fires)
fires_test_prepared = full_pipeline.transform(fires_test)