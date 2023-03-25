#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='whitegrid', font_scale=1.5)
sns.set_palette('Set2', n_colors=10)
plt.rc('font', family='AppleGothic')
plt.rc('axes', unicode_minus=False)

import streamlit as st
from datetime import date


df = pd.read_csv('data/trans_raw.csv', encoding='euc-kr')
lat = pd.read_csv('data/lat.csv', encoding='euc-kr')

df['구매일자'] = pd.to_datetime(df['구매일자'])
# print(df.head())

st.set_page_config(page_title='Transaction Dashboard', 
                   page_icon='🐋', layout='wide')
st.title("Data App Dashboard")

if st.button('새로고침'):
    st.experimental_rerun()


my_df = df.copy()
st.sidebar.title("조건 필터")
st.sidebar.header("날짜 조건")
col1, col2 = st.sidebar.columns(2)
with col1:
    start_date = st.date_input("시작일시", date(2021, 1, 1),
                                       min_value=date(2021,1,1),
                                       max_value=date(2021,12,30))
with col2:
    end_date = st.date_input("종료일시", date(2021, 12, 31),
                                     min_value=date(2021,1,2),
                                     max_value=date(2021,12,31))

# df['구매일자'] = pd.to_datetime(df['구매일자'], errors='coerce')
my_df = my_df[my_df['구매일자'].dt.date.between(start_date, end_date)]

option01 = st.sidebar.checkbox('취소거래제외', value=False)
if option01:
    my_df = my_df[my_df['취소여부']!=1]
    
st.sidebar.header('상품분류선택')
option02 = st.sidebar.multiselect('상품대분류', (my_df.상품대분류명.unique()), default=(my_df.상품대분류명.unique()))
my_df = my_df[my_df.상품대분류명.isin(option02)]
option03 = st.sidebar.multiselect('상품중분류', (my_df.상품중분류명.unique()), default=(my_df.상품중분류명.unique()))
my_df = my_df[my_df.상품중분류명.isin(option03)]


st.header('0. Overview')

col1, col2, col3 = st.columns(3)
col1.metric(label = "평균 판매액(단위:만원)", 
            value = round(my_df['구매금액'].mean() / 10000,3), 
            delta=round(my_df['구매금액'].mean() / 10000 - df['구매금액'].mean() / 10000, 3))

col2.metric(label = "구매 고객수", value = my_df['ID'].nunique(),
            delta=my_df['ID'].nunique() - df['ID'].nunique())

col3.metric(label = "고객 평균 연령", value = round(my_df.groupby('ID')['연령'].mean().mean(),3),
            delta = round(my_df.groupby('ID')['연령'].mean().mean() - df.groupby('ID')['연령'].mean().mean(),3))

st.header('1. 매출현황분석')

st.subheader('전체')
time_frame = st.selectbox("월별/주별/요일별", ("month", "week","weekday"))
whole_values = my_df.groupby(time_frame)[['구매금액']].sum()
whole_values.index.name = 'index'
st.download_button('Download',whole_values.to_csv(encoding='euc-kr'), '매출현황분석.csv')
st.area_chart(whole_values, use_container_width=True)

st.subheader('지역별 비교')

city_range = st.radio(label="범위선택", options=("시단위", "구단위"), index=0)

if city_range=='시단위':
    city_range='구매지역_대분류'
    small_region=False
else:
    city_range='구매지역_소분류'
    small_region = st.multiselect("구선택", (my_df.구매지역_소분류.unique()), (my_df.구매지역_소분류.unique()))

if small_region==False:
    city_values = my_df
else:
    city_values = my_df[my_df['구매지역_소분류'].isin(small_region)]
    
city_values = pd.pivot_table(city_values, index=time_frame, columns=city_range, 
                             values='구매금액', aggfunc='sum',fill_value=0)
city_values.index.name = None
city_values.columns = list(city_values.columns)

st.line_chart(city_values, use_container_width=True)

st.subheader('Top5 비교')

def top5(col_name, top=5):
    my_agg = (my_df.groupby(col_name)['구매금액'].sum()/1000000).reset_index().sort_values('구매금액', ascending=False).head(top)
    my_agg[col_name] = my_agg[col_name].astype('str')
    fig = plt.figure(figsize=(15,10))
    ax = sns.barplot(x='구매금액', y=col_name, data=my_agg)
    ax.bar_label(ax.containers[0], label_type='center', color='white')
    return fig

col1, col2, col3 = st.columns(3)
with col1:
    st.write('Top5 구매지역(단위:백만원)')
    st.pyplot(top5('구매지역_소분류'))
with col2:
    st.write('Top5 구매시간(단위:백만원)')
    st.pyplot(top5('구매시간'))
with col3:
    st.write('Top5 구매상품(단위:백만원)')
    st.pyplot(top5('상품중분류명'))


st.header('2. 고객현황분석')

st.subheader('성별 현황')
st.write('성별 구매건수')
gender_count = my_df.groupby([time_frame, '성별'])['구매수량'].sum().unstack()
gender_count.columns = ['남성','여성']
st.bar_chart(data=gender_count, use_container_width=True)

st.subheader('연령분포')
age_frame = st.selectbox("조건화선택", ("전체", "성별","취소여부","구매지역_대분류"))
if age_frame=='전체':
    fig = sns.displot(x='연령', data=my_df, height=7, rug=True, kde=True)
else:
    fig = sns.displot(x='연령', data=my_df, height=7, rug=True, hue=age_frame, kde=True)
st.pyplot(fig)

st.subheader('지역별분포')
lat = lat.rename(columns={'지역':'거주지역'})
map_lat = my_df[['거주지역']].merge(lat)
jit = np.random.randn(len(map_lat), 2)
jit_ratio = 0.01
map_lat[['lat','lon']] = map_lat[['lat','lon']] + jit*jit_ratio
st.map(map_lat)

