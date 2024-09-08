import datetime
import pandas as pd
#from FlagEmbedding import FlagReranker
import numpy as np
import time
import pymysql



class bgeSearch():

    def __init__(self):
        # 数据库连接
        try:
            self.connection = pymysql.connect(
                host='192.168.8.6',
                port=3306,
                user='root',
                passwd='Zhanxin@1234',
                db='zhanxin',
                charset='utf8'
            )
            print("connection succeess")
        except pymysql.Error as e:
            print("Error %s" % str(e))
            exit()
        self.cursor = self.connection.cursor()  # 创建游标对象


    # SELECT DATA
    def selectFromDatabase(self, query):
        try:
            self.cursor.execute(query)
            res = self.cursor.fetchall()
            return res
        except Exception as e:
            return "Error %s" % str(e)
        

    # 关闭游标和数据库的连接
    def disconnectAll(self):
        self.cursor.close()
        self.connection.close()

    # 把数据库的热词给转成list of str
    def findReci(self, connection, query):
        reci = connection.selectFromDatabase(query)
        res = []
        for each in reci:
            res.append(list(each)[0])
        return res
        


    # 当热词格式为 -> xxx: 0.4, xx:0.6 这种的时候，用这个function去提取纯词语
    def extractCleanWords(self, cell): # 这里拿的是一个单独的cell
        if cell == None or cell == []:
            return None, None    ######################333 --------------------> 返回产业名称 要改
        print("cell: ", cell)
        words = cell.split(',')
        cleanWords = [word.split(':')[0] for word in words]
        popularIndex = [word.split(':')[1] for word in words]
        return cleanWords, popularIndex




if __name__ == "__main__":
    connection = bgeSearch()





