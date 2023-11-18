import pymysql
import pandas as pd

#DB INSERT 함수
def insert_row(cursor, data, table):
    col = ', '.join(data.index)
    place_holders = ', '.join(['%s']*len(data.index))
    key_holders = ', '.join([k+'=%s' for k in data.index]) 
    que = 'INSERT INTO {} ({}) VALUES ({}) ON DUPLICATE KEY UPDATE {}'.format(table, col, place_holders, key_holders)
    cursor.execute(que, list(data.values)*2)


def main(data_file='artist_info.csv'):
    #Data Loading
    artist_info = pd.read_csv(data_file)
    print('데이터로딩완료')
    #DB 접속
    host = 'moondb.crucpgwi9avw.ap-northeast-2.rds.amazonaws.com'
    port = 3306
    username = 'moonsql'
    database = 'production'
    pw = '123456789'

    conn = pymysql.connect(host=host, user=username, passwd=pw,
                        db=database, port=port,
                        use_unicode=True, charset='utf8')
    cursor = conn.cursor()
    print('DB접속성공')

    # 데이터 입력
    for i, row in artist_info.iterrows():
        insert_row(cursor, row, 'Artists')
    conn.commit()
    print('DB업데이트완료')

if __name__=='__main__':
    main()
