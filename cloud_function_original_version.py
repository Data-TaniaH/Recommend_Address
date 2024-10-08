import functions_framework
from mysql.connector import pooling
import pymysql
from google.cloud.sql.connector import Connector, IPTypes
import os
from flask import escape,jsonify
import pytz
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.naive_bayes import CategoricalNB
import json

# Initialize a global pool variable
pool = None

# Create a MySQL connection pool
def create_pool():
    global pool
    if pool is None:
        try:
            pool=pooling.MySQLConnectionPool(
                user=os.getenv('DB_USER'), # Database user from environment variables
                password=os.getenv('DB_PASSWORD'),  # Database password from environment variables
                database=os.getenv('DB_DATABASE'), # Database name from environment variables
                host=os.getenv('PRIVATE_IP'),
                pool_size=2 # Set the pool size to 2 (the maximum number of connections in the pool)
                )

        except Exception as e:
            print(f"Error creating connection pool: {e}")
            raise
    
    return pool

#計算兩個點的距離
def haversine(lat1, lon1, lat2, lon2):
    # 將經緯度從度數轉換為弧度
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    # 計算差值
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    # Haversine 公式
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))

    # 地球半徑（公里）
    r = 6371.0
    return c * r

#有傳入上車經緯度的function
def process_has_lat_lng_data(df_raw, target_lat, target_lon,start_address, hour_type, is_holiday,dayofweek,mapping_data):
  try: 
   #### DATA PREPROCESSING ####
    df_raw[['latitude', 'longitude']] = df_raw['start_latlng'].str.split(',', expand=True) # 分割 start_latlng 為 latitude 和 longitude
    df_raw['latitude'] = df_raw['latitude'].astype(float)
    df_raw['longitude'] = df_raw['longitude'].astype(float)
    df_raw[['end_latitude', 'end_longitude']] = df_raw['end_latlng'].str.split(',', expand=True)   # 分割 end_latlng 為 end_latitude 和 end_longitude
    df_raw['end_address_rounded']=df_raw['end_latitude'].astype(float).round(2).astype(str) + ',' + df_raw['end_longitude'].astype(float).round(2).astype(str) #將下車經緯度進位到小數點2位後再concat起來
    df_raw['distance_km'] = df_raw.apply(lambda row: haversine(row['latitude'], row['longitude'], target_lat, target_lon), axis=1) #計算目標經緯度與歷史上車點距離
    df_raw = df_raw[df_raw['distance_km'] < 30] #過濾距離小於30km的資料
    df_raw['count'] = df_raw.groupby('end_latlng')['end_latlng'].transform('count') #計算下車點紀錄的次數
    df_raw['created_at'] = pd.to_datetime(df_raw['created_at']) #created_at 轉換為datetime
    df_sorted = df_raw.sort_values(by=['count', 'created_at'], ascending=[False, False]) #根據下車地點紀錄次數以及時間由大到小排序

    df_unique_all = df_sorted.drop_duplicates(subset='end_latlng', keep='first') #去除重複下車點，僅保留前20筆記錄
    df_unique=df_unique_all[df_unique_all['end_address_rounded']!=start_address] #過濾與上車經緯度相同的下車地點
    top_20_unique = df_unique['end_latlng'].head(20) #取出 top 20 個不重複的下車經緯度

    df_filtered = df_raw[df_raw['end_latlng'].isin(top_20_unique)]#撈取包含top 20下車地點的原始資料
    df_process = df_filtered.loc[:,['start_latlng', 'end_latlng', 'hour_type', 'is_holiday','dayofweek']] # 返回所需的欄位

  except Exception as e:
    print(f"An error occurred during data preprocessing(has lat/lng version)",e) # 當上方資料處理出現錯誤時，回傳錯誤訊息
    return 'An unexpected error occurred during data processing', 500
  
    #### MODEL BUILDING ####
  try:
        # 建立用戶輸入數值的 DataFrame
    data = {
        'start_latlng': [start_address],
        'hour_type': [hour_type],
        'is_holiday': [is_holiday],
        'dayofweek':[dayofweek],
        'is_end_address': [np.nan]  # 预测值未知，因此設置為NaN
    }
    df_test = pd.DataFrame(data)

        # 建立一個DataFrame 存取最大index
    default_data = {
        'start_latlng': [start_address],
        'hour_type': ['午夜'],
        'is_holiday': ['1'],
        'dayofweek':['7'],
        'is_end_address': [1]  
    }
    df_default = pd.DataFrame(default_data)

    final_result = pd.DataFrame(columns=['end_latlng', 'prob'])  # 初始化结果的DataFrame


    distinct_end_latlng = df_process['end_latlng'].unique().tolist() #將所有不重複下車地點列出

    for end_location in distinct_end_latlng:       # 循環每一個下車地點

        df_process['is_end_address'] = df_process['end_latlng'].apply(lambda x: 1 if x == end_location else 0) #建立is_end_address欄位

        # 合并处理数据和测试数据
        df_combined_1 = pd.concat([df_process.loc[:,['start_latlng', 'hour_type', 'is_holiday','dayofweek', 'is_end_address']], df_test], ignore_index=True) #合併歷史資料與用戶輸入資料
        df_combined = pd.concat([df_combined_1[['start_latlng', 'hour_type', 'is_holiday','dayofweek', 'is_end_address']], df_default], ignore_index=True) #合併歷史資料、用戶輸入資料與存取最大index的資料

        # Define the categories explicitly for features handled by OrdinalEncoder
        categories = {
            'start_latlng':df_combined['start_latlng'].unique().tolist(), #start_latlng
            'hour_type':['凌晨', '早尖峰', '早離峰', '午離峰','晚尖峰','小晚尖','午夜'],  # hour_type
            'is_holiday':['0', '1'],            # is_holiday
            'dayofweek':['1', '2', '3', '4', '5', '6', '7']  # dayofweek
            }

        # Encode other features with OrdinalEncoder using explicit categories
        ordinal_encoder = OrdinalEncoder(categories=[categories['start_latlng'],categories['hour_type'], categories['is_holiday'], categories['dayofweek']])
        ordinal_encoded=ordinal_encoder.fit_transform(df_combined[['start_latlng','hour_type', 'is_holiday', 'dayofweek']])
        encoded_df = pd.DataFrame(ordinal_encoded, columns=['start_latlng','hour_type', 'is_holiday', 'dayofweek'])


        # Combine is_end_address with encoded features
        encoded_df['is_end_address'] = df_combined['is_end_address'].reset_index(drop=True)

        # 分割train and predict data
        train_data = encoded_df[encoded_df['is_end_address'].notna()]
        predict_data = encoded_df[encoded_df['is_end_address'].isna()]

        # 訓練模型
        model = CategoricalNB()
        model.fit(train_data.loc[:,['start_latlng', 'hour_type', 'is_holiday','dayofweek']], train_data['is_end_address'])


        unique_classes = train_data['is_end_address'].nunique() # 確認model的training data的y的class有幾個
        if unique_classes > 1:
          probability_predictions = model.predict_proba(predict_data[['start_latlng', 'hour_type', 'is_holiday', 'dayofweek']])[:,1]
        else:
          probability_predictions = np.ones(len(predict_data)) # Handle the single-class case

        # 存取结果
        temp_result = pd.DataFrame({
            'end_latlng': [end_location],
            'prob': probability_predictions
        })
        final_result = pd.concat([final_result, temp_result], ignore_index=True)

    # 按照概率排序取前5個結果
    final_result = final_result.sort_values(by='prob', ascending=False).reset_index(drop=True)
    top_result = final_result.head(5)

  except Exception as e:
    print(f"An error occurred during modeling(has lat/lng version)",e) # 當模型運算出現錯誤時，回傳錯誤訊息
    return 'An unexpected error occurred during modeling', 500
  
  try:
    mapping_result = pd.merge(top_result, mapping_data, how='left', on='end_latlng')     # Perform mapping
    results = mapping_result[mapping_result['end_address'].notna() & (mapping_result['end_address'] != '')]     # Filter out rows where 'end_address' is blank or null
  except Exception as e:
    print(f"An error occurred during mapping result(has lat/lng version)",e) # 當mapping出現錯誤時，回傳錯誤訊息
    return 'An unexpected error occurred during mapping result', 500
  
  return results

def process_no_lat_lng_data(df_raw, target_lat, target_lon,start_address, hour_type, is_holiday,dayofweek,mapping_data):
   #### DATA PREPROCESSING ####
  try: 
    df_raw[['end_latitude', 'end_longitude']] = df_raw['end_latlng'].str.split(',', expand=True) # 分割 end_latlng 為 end_latitude 和 end_longitude
    df_raw['end_address_rounded']=df_raw['end_latitude'].astype(float).round(2).astype(str) + ',' + df_raw['end_longitude'].astype(float).round(2).astype(str)#將下車經緯度進位到小數點2位後再concat起來
    df_raw['count'] = df_raw.groupby('end_latlng')['end_latlng'].transform('count') #計算下車點紀錄的次數
    df_raw['created_at'] = pd.to_datetime(df_raw['created_at']) #created_at 轉換為datetime
    df_sorted = df_raw.sort_values(by=['count', 'created_at'], ascending=[False, False]) #根據下車地點紀錄次數以及時間由大到小排序

    df_unique_all = df_sorted.drop_duplicates(subset='end_latlng', keep='first') #去除重複下車點，僅保留前20筆記錄
    df_unique=df_unique_all[df_unique_all['end_address_rounded']!=start_address]#過濾與上車經緯度相同的下車地點
    top_20_unique = df_unique['end_latlng'].head(20) #取出 top 20 個不重複的下車經緯度

    df_filtered = df_raw[df_raw['end_latlng'].isin(top_20_unique)] #撈取包含top 20下車地點的原始資料
    df_process = df_filtered.loc[:,['end_latlng', 'hour_type', 'is_holiday','dayofweek']] # 返回所需的欄位
  
  except Exception as e:
    print(f"An error occurred during data preprocessing(no lat/lng version)",e) # 當上方資料處理出現錯誤時，回傳錯誤訊息
    return 'An unexpected error occurred during data processing', 500
  
  #### MODEL BUILDING ####
  try:  # 建立用戶輸入數值的 DataFrame
    data = {
        'start_latlng': [start_address],
        'hour_type': [hour_type],
        'is_holiday': [is_holiday],
        'dayofweek':[dayofweek],
        'is_end_address': [np.nan]  # 预测值未知，因此設置為 NaN
    }
    df_test = pd.DataFrame(data)


    default_data = {   # 建立一個DataFrame 存取最大index
        'hour_type': ['午夜'],
        'is_holiday': ['1'],
        'dayofweek':['7'],
        'is_end_address': [1] 
    }
    df_default = pd.DataFrame(default_data)

    final_result = pd.DataFrame(columns=['end_latlng', 'prob'])  # 初始化结果的DataFrame

    distinct_end_latlng = df_process['end_latlng'].unique().tolist() #將所有不重複下車地點列出

    for end_location in distinct_end_latlng: # 循環每一個下車地點

        df_process['is_end_address'] = df_process['end_latlng'].apply(lambda x: 1 if x == end_location else 0) #建立is_end_address欄位
        df_combined_1 = pd.concat([df_process.loc[:,[ 'hour_type', 'is_holiday','dayofweek', 'is_end_address']], df_test.loc[:,[ 'hour_type', 'is_holiday','dayofweek', 'is_end_address']]], ignore_index=True) #合併歷史資料與用戶輸入資料
        df_combined = pd.concat([df_combined_1[[ 'hour_type', 'is_holiday','dayofweek', 'is_end_address']], df_default], ignore_index=True) #合併歷史資料、用戶輸入資料與存取最大

        # Define the categories explicitly for features handled by OrdinalEncoder
        categories = {
            'hour_type':['凌晨', '早尖峰', '早離峰', '午離峰','晚尖峰','小晚尖','午夜'],  # hour_type
            'is_holiday':['0', '1'],            # is_holiday
            'dayofweek':['1', '2', '3', '4', '5', '6', '7']  # dayofweek
            }

        # Encode other features with OrdinalEncoder using explicit categories
        ordinal_encoder = OrdinalEncoder(categories=[categories['hour_type'], categories['is_holiday'], categories['dayofweek']])
        ordinal_encoded=ordinal_encoder.fit_transform(df_combined[['hour_type', 'is_holiday', 'dayofweek']])
        encoded_df = pd.DataFrame(ordinal_encoded, columns=['hour_type', 'is_holiday', 'dayofweek'])


        # # Combine encoded features
        encoded_df['is_end_address'] = df_combined['is_end_address'].reset_index(drop=True)

        # 分割train and predict data
        train_data = encoded_df[encoded_df['is_end_address'].notna()]
        predict_data = encoded_df[encoded_df['is_end_address'].isna()]

        # 訓練模型
        model = CategoricalNB()
        model.fit(train_data.loc[:,['hour_type', 'is_holiday','dayofweek']], train_data['is_end_address'])


        unique_classes = train_data['is_end_address'].nunique() # 確認model的training data的y的class有幾個

        if unique_classes > 1:
          probability_predictions = model.predict_proba(predict_data[[ 'hour_type', 'is_holiday', 'dayofweek']])[:,1]
        else:
             # Handle the single-class case, for example, by setting a default probability
          probability_predictions = np.ones(len(predict_data))

          # 存取结果
        temp_result = pd.DataFrame({
            'end_latlng': [end_location],
            'prob': probability_predictions
        })
        final_result = pd.concat([final_result, temp_result], ignore_index=True)


    # 按照概率排序取前5個結果
    final_result = final_result.sort_values(by='prob', ascending=False).reset_index(drop=True)
    top_result = final_result.head(5)

  except Exception as e:
    print(f"An error occurred during modeling(no lat/lng version)",e) # 當模型運算出現錯誤時，回傳錯誤訊息
    return 'An unexpected error occurred during modeling', 500
  
  try:
    # Perform mapping
    mapping_result = pd.merge(top_result, mapping_data, how='left', on='end_latlng')
    # Filter out rows where 'end_address' is blank or null
    results = mapping_result[mapping_result['end_address'].notna() & (mapping_result['end_address'] != '')]
  
  except Exception as e:
    print(f"An error occurred during mapping result(no lat/lng version)",e) # 當mapping出現錯誤時，回傳錯誤訊息
    return 'An unexpected error occurred during mapping result', 500

  return results



def getRecommandedAddress(request):
    # Record the start time
    start_time = datetime.now()

    # Ensure the pool is created
    pool_connection=create_pool()

    # Get client IP from request headers
    allowed_ips = ["3.113.179.176"]
    client_ip = request.headers.get('X-Forwarded-For', request.remote_addr)
    client_ips = client_ip.split(',')

    # Check if any of the client IPs are in the allowed list
    allow_access = any(ip.strip() in allowed_ips for ip in client_ips)

    # Uncomment this block to enforce IP restriction
    # if not allow_access:
    #     return ('Forbidden', 403)

    # Get the parameters from the query string
    uid = int(request.args.get('uid'))
    if request.args.get('lat')=='' or request.args.get('lat') is None:
      lat=999
    else: 
      lat=float(request.args.get('lat'))

    if request.args.get('lng')=='' or request.args.get('lng') is None:
      lng=999
    else: 
      lng=float(request.args.get('lng'))

    # Ensure that the required parameters are provided
    if not uid:
        return ('Missing uid parameter', 400)

    taipei_tz = pytz.timezone('Asia/Taipei')     # Define the Asia/Taipei timezone
    created_at = datetime.now(taipei_tz) # Get the current date and time
    hour = created_at.hour  # Extract the hour
    weekday = created_at.weekday()+1  # Extract the day of the week (Monday = 0+1, Sunday = 6+1)
    hour_type = { # Hour to hour_type
        (7 <= hour <= 9): '早尖峰',
        (10 <= hour <= 12): '早離峰',
        (13 <= hour <= 16): '午離峰',
        (17 <= hour <= 19): '晚尖峰',
        (20 <= hour <= 22): '小晚尖',
        (2 <= hour <= 6): '凌晨'
      }.get(True, '午夜')
    is_holiday = '0' if 1 <= weekday <= 5 else '1'  # is hoilday
   
    start_lat_rounded = round(lat, 2)  # Round start_lat to 2 decimal places
    start_lng_rounded = round(lng, 2)  # Round start_lng to 2 decimal places
    start_address = f"{round(lat, 2)},{round(lng, 2)}"   # concat the start_lat and start_lng together

    try:

        # Get a connection from the pool
        conn = pool_connection.get_connection()
        # set up cursor
        cursor = conn.cursor(dictionary=True) #creates a cursor object that allows you to execute SQL queries, ensures that the results are returned as dictionaries, where the column names are the keys.
        # Prepare the query1 # Fetch results1 - table 1
        cursor.execute("SELECT * FROM address_v2_training_data_test WHERE uid = %s", (uid,))
        result_1=cursor.fetchall()
        # Prepare the query2    # Fetch results2 - table 2
        cursor.execute("SELECT * FROM address_v2_suggestion_test WHERE uid = %s", (uid,))
        result_2=cursor.fetchall()
        cursor.close()
        conn.close()

    except mysql.connector.Error as err:
        print("Error:", err)
        return ('Error connecting to the database or executing query', 500) #if error occurred while connecting to the database or executing query


    try:
       
      raw_result=pd.DataFrame(result_1) # Get the raw result
      map_result=pd.DataFrame(result_2) # Get the map result

      if raw_result.empty: 
         return "No history data"
      
      if map_result.empty:
         return "No mapping data"
      
    except Exception as e:
        print("Unexpected error after fetching data:", e) #if error occurred while fetching data
        return 'An unexpected error occurred after fetching data', 500

    if start_lat_rounded!=999.00 and start_lng_rounded!=999.00: 
       top_recommend_df=process_has_lat_lng_data(raw_result, start_lat_rounded, start_lng_rounded,start_address, hour_type, is_holiday,str(weekday),map_result)
    else:
       top_recommend_df=process_no_lat_lng_data(raw_result, start_lat_rounded, start_lng_rounded,start_address, hour_type, is_holiday,str(weekday),map_result)

    
    top_recommend_result=top_recommend_df.to_dict("records") # Convert the DataFrame to a list of dictionaries for the response data

    end_time = datetime.now() # Record the end time
    execution_time = end_time - start_time # Calculate the execution time
    execution_time_seconds = execution_time.total_seconds()  # Convert the execution time to seconds
    execution_time_str = f"{execution_time_seconds:.6f} seconds" # Convert to a string
    response_data=json.dumps({"data": top_recommend_result,"execution_time":execution_time_str}, ensure_ascii=False) #output data in JSON format

    return response_data