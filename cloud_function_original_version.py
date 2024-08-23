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

def process_location_data(df_raw, target_lat, target_lon):
  if target_lat!=999.00 and target_lon!=999.00:
    # 分割 start_latlng 为 latitude 和 longitude
    df_raw[['latitude', 'longitude']] = df_raw['start_lating'].str.split(',', expand=True)
    df_raw['latitude'] = df_raw['latitude'].astype(float)
    df_raw['longitude'] = df_raw['longitude'].astype(float)

    # 计算目标经纬度与历史上车点的距离
    df_raw['distance_km'] = df_raw.apply(lambda row: haversine(row['latitude'], row['longitude'], target_lat, target_lon), axis=1)

    # 过滤距离小于 30 km 的数据
    df_raw = df_raw[df_raw['distance_km'] < 30]

    # 计算下车点记录次数
    df_raw['count'] = df_raw.groupby('end_lating')['end_lating'].transform('count')

    # 正规化时间字段
    df_raw['created_at'] = pd.to_datetime(df_raw['created_at'])

    # 根据下车点记录次数以及时间由大到小排序
    df_sorted = df_raw.sort_values(by=['count', 'created_at'], ascending=[False, False])

    # 去除重复的下车点，仅保留前 20 笔记录
    df_unique = df_sorted.drop_duplicates(subset='end_lating', keep='first')
    top_20_unique = df_unique['end_lating'].head(20)

    # 撈取原始数据中的 top 20 下车点
    df_filtered = df_raw[df_raw['end_lating'].isin(top_20_unique)]

    # 返回所需字段
    result = df_filtered.loc[:,['start_lating', 'end_lating', 'hour_type', 'is_holiday','dayofweek']]
  
  else: 

    # 计算下车点记录次数
    df_raw['count'] = df_raw.groupby('end_lating')['end_lating'].transform('count')

    # 正规化时间字段
    df_raw['created_at'] = pd.to_datetime(df_raw['created_at'])

    # 根据下车点记录次数以及时间由大到小排序
    df_sorted = df_raw.sort_values(by=['count', 'created_at'], ascending=[False, False])

    # 去除重复的下车点，仅保留前 20 笔记录
    df_unique = df_sorted.drop_duplicates(subset='end_lating', keep='first')
    top_20_unique = df_unique['end_lating'].head(20)

    # 撈取原始数据中的 top 20 下车点
    df_filtered = df_raw[df_raw['end_lating'].isin(top_20_unique)]

    # 返回所需字段
    result = df_filtered.loc[:,['end_lating', 'hour_type', 'is_holiday','dayofweek']]

  return result


def recommend_list(df_process, start_address, hour_type, is_holiday,dayofweek,mapping_data,target_lat,target_lon):
    # 创建包含用户输入数据的 DataFrame
    data = {
        'start_lating': [start_address],
        'hour_type': [hour_type],
        'is_holiday': [is_holiday],
        'dayofweek':[dayofweek],
        'is_end_address': [np.nan]  # 预测值未知，因此设置为 NaN
    }
    df_test = pd.DataFrame(data)

    final_result = pd.DataFrame(columns=['end_lating', 'prob'])  # 初始化结果 DataFrame

    # 循环遍历每一个独特的 end_latlng
    c_vars = ['start_lating', 'hour_type', 'is_holiday','dayofweek']
    distinct_end_latlng = df_process['end_lating'].unique().tolist()


    if target_lat!=999.00 or target_lon!=999.00:

      for end_location in distinct_end_latlng:
        
        # Change the data type of 'dayofweek' from int to string
        df_process['dayofweek'] = df_process['dayofweek'].astype(str)
        df_process['is_holiday'] = df_process['is_holiday'].astype(str)

        # 创建目标变量 is_end_address
        df_process['is_end_address'] = df_process['end_lating'].apply(lambda x: 1 if x == end_location else 0)

        # 合并处理数据和测试数据
        df_combined = pd.concat([df_process.loc[:,['start_lating', 'hour_type', 'is_holiday','dayofweek', 'is_end_address']], df_test], ignore_index=True)

        # 编码类别特征
        encoded_array = OrdinalEncoder().fit_transform(df_combined.loc[:,['start_lating', 'hour_type', 'is_holiday','dayofweek']])
        encoded_df_p = pd.DataFrame(encoded_array, columns=['start_lating', 'hour_type', 'is_holiday','dayofweek'])
        encoded_df = pd.concat([encoded_df_p, df_combined[['is_end_address']].reset_index(drop=True)], axis=1)

        # 分割训练数据和预测数据
        train_data = encoded_df[encoded_df['is_end_address'].notna()]
        predict_data = encoded_df[encoded_df['is_end_address'].isna()]

        # 训练模型
        model = CategoricalNB()
        model.fit(train_data.loc[:,['start_lating', 'hour_type', 'is_holiday','dayofweek']], train_data['is_end_address'])

        # 预测概率
        probability_predictions = model.predict_proba(predict_data.loc[:,['start_lating', 'hour_type', 'is_holiday','dayofweek']])[:, 1]

        # 存储结果
        temp_result = pd.DataFrame({
            'end_lating': [end_location],
            'prob': probability_predictions
        })
        final_result = pd.concat([final_result, temp_result], ignore_index=True)
    
    else:
      for end_location in distinct_end_latlng:
        # Change the data type of 'dayofweek' from int to string
        df_process['dayofweek'] = df_process['dayofweek'].astype(str)
        df_process['is_holiday'] = df_process['is_holiday'].astype(str)

        # 创建目标变量 is_end_address
        df_process['is_end_address'] = df_process['end_lating'].apply(lambda x: 1 if x == end_location else 0)

        # 合并处理数据和测试数据
        df_combined = pd.concat([df_process.loc[:,[ 'hour_type', 'is_holiday','dayofweek', 'is_end_address']], df_test.loc[:,[ 'hour_type', 'is_holiday','dayofweek', 'is_end_address']]], ignore_index=True)

        # 编码类别特征
        encoded_array = OrdinalEncoder().fit_transform(df_combined.loc[:,[ 'hour_type', 'is_holiday','dayofweek']])
        encoded_df_p = pd.DataFrame(encoded_array, columns=[ 'hour_type', 'is_holiday','dayofweek'])
        encoded_df = pd.concat([encoded_df_p, df_combined[['is_end_address']].reset_index(drop=True)], axis=1)

        # 分割训练数据和预测数据
        train_data = encoded_df[encoded_df['is_end_address'].notna()]
        predict_data = encoded_df[encoded_df['is_end_address'].isna()]

        # 训练模型
        model = CategoricalNB()
        model.fit(train_data.loc[:,['hour_type', 'is_holiday','dayofweek']], train_data['is_end_address'])

        # 预测概率
        probability_predictions = model.predict_proba(predict_data.loc[:,[ 'hour_type', 'is_holiday','dayofweek']])[:, 1]

        # 存储结果
        temp_result = pd.DataFrame({
            'end_lating': [end_location],
            'prob': probability_predictions
        })
        final_result = pd.concat([final_result, temp_result], ignore_index=True)


    # 按概率排序并获取前 5 个结果
    final_result = final_result.sort_values(by='prob', ascending=False).reset_index(drop=True)
    top_result = final_result.head(5)

    # Perform mapping
    mapping_result = pd.merge(top_result, mapping_data, how='left', on='end_lating')
    # Filter out rows where 'end_address' is blank or null
    results = mapping_result[mapping_result['end_address'].notna() & (mapping_result['end_address'] != '')]

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

    
    # Define the Asia/Taipei timezone
    taipei_tz = pytz.timezone('Asia/Taipei')
    # Get the current date and time
    created_at = datetime.now(taipei_tz)
    # Extract the hour
    hour = created_at.hour
    # Extract the day of the week (Monday = 0+1, Sunday = 6+1)
    weekday = created_at.weekday()+1
    # Hour to hour_type
    hour_type = {
        (7 <= hour <= 9): '早尖峰',
        (10 <= hour <= 12): '早離峰',
        (13 <= hour <= 16): '午離峰',
        (17 <= hour <= 19): '晚尖峰',
        (20 <= hour <= 22): '小晚尖',
        (2 <= hour <= 6): '凌晨'
    }.get(True, '午夜')
    # is hoilday
    is_holiday = '0' if 1 <= weekday <= 5 else '1'
    # Round to 2 decimal places
    start_lat_rounded = round(lat, 2)
    start_lng_rounded = round(lng, 2)
    # concat the start_lat and start_lng together
    start_address = f"{round(lat, 2)},{round(lng, 2)}"


    try:

        # Get a connection from the pool
        conn = pool_connection.get_connection()

        # set up cursor
        cursor = conn.cursor(dictionary=True) #creates a cursor object that allows you to execute SQL queries, ensures that the results are returned as dictionaries, where the column names are the keys.
         # Prepare the query1 # Fetch results1 - table 1
        cursor.execute("SELECT * FROM address_v2_training_data WHERE uid = %s", (uid,))
        result_1=cursor.fetchall()

        # Prepare the query2    # Fetch results2 - table 2
        cursor.execute("SELECT * FROM address_v2_suggestion WHERE uid = %s", (uid,))
        result_2=cursor.fetchall()

        cursor.close()
        conn.close()

    except mysql.connector.Error as err:
        print("Error:", err)
        return ('Error connecting to the database or executing query', 500)

    raw_result=pd.DataFrame(result_1)
    map_result=pd.DataFrame(result_2)
    processed_data=process_location_data(raw_result,start_lat_rounded,start_lng_rounded)
    top_recommend_df=recommend_list(processed_data, start_address, hour_type, is_holiday,str(weekday),map_result,start_lat_rounded,start_lng_rounded)
    top_recommend_result=top_recommend_df.to_dict("records")

    # Record the end time
    end_time = datetime.now()
    # Calculate the execution time
    execution_time = end_time - start_time
    # Convert the execution time to seconds
    execution_time_seconds = execution_time.total_seconds() 
    # Convert to a string
    execution_time_str = f"{execution_time_seconds:.6f} seconds"
    #output 
    response_data=json.dumps({"data": top_recommend_result,"execution_time":execution_time_str}, ensure_ascii=False)
    # # Return using jsonify with the appropriate content type
    # response = jsonify(json.loads(response_data))
    # response.headers.set('Content-Type', 'application/json; charset=utf-8')

    return response_data