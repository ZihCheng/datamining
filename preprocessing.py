# -*- coding: utf-8 -*-
"""
Created on Sun Jun  4 21:24:10 2023

@author: zicheng
"""
#%% 導入需使用的模組
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectFromModel

#%% 資料預處理
def DataPreprocessing(): #資料預處理
    OriginalData = pd.read_csv("OriginalData.csv") #讀檔
    OriginalData.columns = ["統計期","統計區","總站日數","良好站日數","普通站日數","對敏感族群不健康日數","對所有族群不健康日數","非常不健康日數","危害站日數","AQI>=100日數比例"]
    
    duplicateData = OriginalData[(OriginalData["統計區"]=="總計") | (OriginalData["統計期"]=="106年") | (OriginalData["統計期"]=="107年") | (OriginalData["統計期"]=="107年") | (OriginalData["統計期"]=="108年") | (OriginalData["統計期"]=="109年") | (OriginalData["統計期"]=="110年") | (OriginalData["統計期"]=="111年") | (OriginalData["統計期"]=="112年")].index #年份不分月資料列
    OriginalData = OriginalData.drop(duplicateData) #刪除重複資料
    
    OriginalData.reset_index(drop = True,inplace=True) #reset index
    
    def date_interval():
        tmp1 = []
        tmp2 = []
        for i in range(len(OriginalData["統計期"])):
            tmp1.append((OriginalData["統計期"][i].split(' '))[0])
            tmp2.append((OriginalData["統計期"][i].split(' '))[1])
        return pd.DataFrame(tmp1, columns = ["統計年"]), pd.DataFrame(tmp2, columns = ["統計月"])
    value0, value1 = date_interval()
    
    def percent_caculate(arr): #計算資料百分比
        days_count = OriginalData["總站日數"]
        temp = []
        for i in range(len(days_count)):
            temp.append(arr[i]/days_count[i])
        result = pd.DataFrame(temp, columns = [pd.DataFrame(arr).columns[0]])
        return result
    
    value2 = percent_caculate(OriginalData["良好站日數"])
    value3 = percent_caculate(OriginalData["普通站日數"])
    value4 = percent_caculate(OriginalData["對敏感族群不健康日數"])
    value5 = percent_caculate(OriginalData["對所有族群不健康日數"])
    value6 = percent_caculate(OriginalData["非常不健康日數"])
    value7 = percent_caculate(OriginalData["危害站日數"])
    #總站日數 = 良好站日數 + 普通站日數 + 對敏感族群不健康日數 + 對所有族群不健康日數 + 非常不健康日數 + 危害站日數
    
    #預測值離散化，以等距法進行離散化，分為低、中、高三個級距
    pd.cut(OriginalData["AQI>=100日數比例"],bins = 3).value_counts()
    """
    In : pd.cut(OriginalData["AQI>=100日數比例"],bins = 3).value_counts()
    Out: 
    (-0.0742, 24.73]    1431
    (24.73, 49.46]       225
    (49.46, 74.19]        60
    Name: AQI>=100日數比例, dtype: int64
    """
    Discretization_label = ["低","中","高"]
    value8 = pd.DataFrame(pd.cut(OriginalData["AQI>=100日數比例"], bins = 3, labels = Discretization_label), columns = ["AQI>=100日數比例"])
    
    #原始資料是無序離散值的話 => One Hot Encoding (Dummies)
    onehot_encoder = OneHotEncoder(sparse = False)
    encoder_region = onehot_encoder.fit_transform(OriginalData[["統計區"]])
    encoder_region = pd.DataFrame(encoder_region)
    encoder_region.columns = onehot_encoder.categories_[0]
    
    PreprocessingData = pd.concat([value0,value1,encoder_region,value2,value3,value4,value5,value6,value7,value8], axis = 1)
    
    #原始資料是有序離散值的話 => Label Encoding
    le = LabelEncoder()
    PreprocessingData["統計年"] = le.fit_transform(PreprocessingData["統計年"])
    PreprocessingData["統計月"] = le.fit_transform(PreprocessingData["統計月"])
    PreprocessingData["AQI>=100日數比例"] = le.fit_transform(PreprocessingData["AQI>=100日數比例"])
    
    #資料分割，訓練8，測試2
    trainingData, testingData = train_test_split(PreprocessingData, test_size=0.2, random_state=20220622)
    
    return trainingData, testingData

trainingData, testingData = DataPreprocessing() 
"""
trainingData : 訓練資料集
testingData : 測試資料集
"""

X_train = trainingData[['統計年', '統計月', '南投縣', '嘉義市', '嘉義縣', '基隆市', '宜蘭縣', '屏東縣', '彰化縣', '新北市','新竹市', '新竹縣', '桃園市', '澎湖縣', '臺中市', '臺北市', '臺南市', '臺東縣', '花蓮縣', '苗栗縣','連江縣', '金門縣', '雲林縣', '高雄市', '良好站日數', '普通站日數', '對敏感族群不健康日數','對所有族群不健康日數', '非常不健康日數', '危害站日數']]
y_train = trainingData[['AQI>=100日數比例']]

X_test = testingData[['統計年', '統計月', '南投縣', '嘉義市', '嘉義縣', '基隆市', '宜蘭縣', '屏東縣', '彰化縣', '新北市','新竹市', '新竹縣', '桃園市', '澎湖縣', '臺中市', '臺北市', '臺南市', '臺東縣', '花蓮縣', '苗栗縣','連江縣', '金門縣', '雲林縣', '高雄市', '良好站日數', '普通站日數', '對敏感族群不健康日數','對所有族群不健康日數', '非常不健康日數', '危害站日數']]
y_test = testingData[['AQI>=100日數比例']]

X = pd.concat([X_train, X_test])
y = pd.concat([y_train, y_test])

"""
X : 未分割資料集自變數
y : 未分割資料集應變數(目標變數)

X_train : 訓練資料集自變數
y_train : 訓練資料集應變數(目標變數)

X_test : 測試資料集自變數
y_test : 測試資料集應變數(目標變數)
"""

#%% 挑選變數
clf = DecisionTreeClassifier(criterion = 'gini', min_samples_split = 0.25, min_samples_leaf = 2)
clf.fit(X_train, y_train)

clf.score(X_train, y_train)
"""
訓練正確率
In : clf.score(X_train, y_train)
Out: 0.9493769470404985
"""

clf.feature_importances_
"""
變數重要性
In : clf.feature_importances_
Out: 
array([0.       , 0.       , 0.       , 0.       , 0.       , 0.       ,
       0.       , 0.       , 0.       , 0.       , 0.       , 0.       ,
       0.       , 0.       , 0.       , 0.       , 0.       , 0.       ,
       0.       , 0.       , 0.       , 0.       , 0.       , 0.       ,
       0.       , 0.       , 0.9476171, 0.0523829, 0.       , 0.       ])
"""

sm = SelectFromModel(clf, max_features = 2)
sm.fit(X_train, y_train)
sm.get_feature_names_out()
X_train_new = pd.DataFrame(sm.transform(X_train), columns = ['對敏感族群不健康日數', '對所有族群不健康日數'])
"""
X_train_new : 經過變數挑選(使用feature selecting)後的訓練資料集自變數
由變數重要性輸出值可以得知 
對敏感族群不健康日數, 對所有族群不健康日數
此二項變數對整體模型具有極大影響力
"""

def test_feature_select_correct_rate():
    test_clf = DecisionTreeClassifier(criterion = 'gini', min_samples_split = 0.25, min_samples_leaf = 2)
    test_clf.fit(X_train_new, y_train)
    print("變數挑選後訓練正確率=",test_clf.score(X_train_new, y_train))
    return
test_feature_select_correct_rate()
"""
測試變數挑選後模型訓練正確率
In : 
    def test_feature_select_correct_rate():
        test_clf = DecisionTreeClassifier(criterion = 'gini', min_samples_split = 0.25, min_samples_leaf = 2)
        test_clf.fit(X_train_new, y_train)
        print("變數挑選後訓練正確率=",test_clf.score(X_train_new, y_train))
        return
    test_feature_select_correct_rate()
Out: 變數挑選後訓練正確率= 0.9493769470404985
由此結果可以得知 全變數 與 變數挑選 後，模型訓練正確率無差異，故可以使用變數挑選後之資料集進行預測
"""