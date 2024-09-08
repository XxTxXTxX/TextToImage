import base64
import datetime
from io import BytesIO
import io
import os
import shutil
import time
from PIL import Image

from fake_useragent import UserAgent
import pandas as pd
import requests


import retriver
import wordsSimilar








class main():
    
    def __init__(self): 
        self.model = retriver.hfLoading()       # Loading HuggingFace, 不用管这个
        self.dataBase = wordsSimilar.bgeSearch()        # 这个就是数据库连接，不要被名字误导了
        self.category = "xinnengyuan"    # 后面继续加，这是新闻类别
        self.chanye = "新能源"
        self.date = datetime.datetime.now().strftime('%Y-%m-%d') # 自动生成日期
        self.pathDate = self.date.split(' ')[0].replace('-', '') # 处理日期
        self.csvFile = "/root/zhanxin/code/autocraw/recommend/{}/{}/origin_excelFile_新闻_{}_加属性.csv".format(self.pathDate, self.chanye, self.pathDate) # 新闻路径
        
        self.thumbFilePath = "/root/zhanxin/code/autocraw/recommend/{}/{}/thumbFiles/".format(self.pathDate, self.chanye)
        self.thumbImgDownloadPath = "/root/zhanxin/code/autocraw/recommend/{}/{}/downloaded_images_thumb/".format(self.pathDate, self.chanye)
        self.mdImgDownloadPath = "/root/zhanxin/code/autocraw/recommend/{}/{}/downloaded_images_md/".format(self.pathDate, self.chanye)
        self.mdFilePath = "/root/zhanxin/code/autocraw/recommend/{}/{}/mdFiles/".format(self.pathDate, self.chanye)
        self.similarDic = {}

        self.bm25_retriever = None      # 下面几个是数据库向量化的数据，getVectorizedData()会自行处理
        self.vectordb = None
        self.ranktokenizer = None
        self.rankmodel = None
        self.vectorizedCategory = None

        




    def thumbFileSearch(self, id):              # 在thumbfile里面查找新闻有没有图片，通过新闻id
        folderPath = self.thumbFilePath
        print("查找{}.txt图片中".format(id))

        if folderPath is None:
            raise NotADirectoryError("thumbFiles文件夹为空或者路径错误")
        file_names = os.listdir(folderPath)
        if ("{}.txt".format(id)) not in file_names:
            print("图片不在这里！")
            return False
        print("图片在这里！")
        return True
    
    def sorter(self, popularIndex, reciList):    # --> 把热词们按照大到小的顺序排序然后输出
        temp = {}
        for i in range(len(popularIndex)):
            temp[reciList[i]] = popularIndex[i]
        tempSorted = sorted(temp.items(), key = lambda kv:(kv[1], kv[0]))
        idx = len(popularIndex) - 1
        res = []
        while idx >= 0:
            res.append(tempSorted[idx][0])
            idx -= 1
        print(res)
        print(tempSorted)
        return res
    


    ###############################    这一块负责拿到图片链接并且将图片给转换成base64格式然后再resize宽度高度     #########################################


    def getImageContent(self, url):
        
        response = requests.get(url, headers={'headers': UserAgent().random}, verify=False)
        imageContent = response.content

        return imageContent


    
    def image_to_base64(self, image_data):
        base64_data = base64.b64encode(image_data)
        base64_string = base64_data.decode("utf-8")
        return base64_string
    

    def save_base64(self, image_path, base64path):    
        with open(image_path, "rb") as image_file:
            image_data = image_file.read()           
        image_base64 = self.image_to_base64(image_data)
        text = '''data:image/{};base64,{}'''.format(image_path.split('.')[-1],image_base64)
        with open(base64path,'w') as f:
            f.write(text)
            print("base64写入成功，到{}".format(base64path))



        
    def resize_image_1000(self, id, imageB64_data):
        self.downloadToThumb(id, imageB64_data)

        with Image.open(self.mdImgDownloadPath + "{}.png".format(id)) as image:
            width = image.size[0]
            height = image.size[1]

            scale = 1000/width
            n_img = image.resize((int(width*scale), int(height*scale)), Image.ANTIALIAS)
            return n_img
        
    def downloadToThumb(self, id, imageB64_data):
        file_path = self.mdImgDownloadPath + "{}.png".format(id)
        imgdata = base64.b64decode(imageB64_data)
        file = open(file_path, 'wb')
        file.write(imgdata)
        file.close()

    def downloadToMd(self, id, imageB64_data):
        file_path = self.mdImgDownloadPath + "{}.png".format(id)
        imgdata = base64.b64decode(imageB64_data)
        file = open(file_path, 'wb')
        file.write(imgdata)
        file.close()
            

    def resize_image_300H_200W(self, id, imageB64_data):
        self.downloadToMd(id, imageB64_data)
        

        with Image.open(self.mdImgDownloadPath + "{}.png".format(id)) as image:
            n_img = image.resize((300, 200), Image.ANTIALIAS)
            return n_img
        

    def save_img_to_downloaded_images(self, id, n_img):
        if os.path.exists(self.thumbImgDownloadPath + "{}.png".format(id)):
            os.remove(self.thumbImgDownloadPath + "{}.png".format(id))
        n_img.save(fp = self.thumbImgDownloadPath + "{}.png".format(id))
        print("Resize之后的thumbfile保存成功! ")

    def save_img_to_md(self, id, n_img):
        if os.path.exists(self.mdImgDownloadPath + "{}.png".format(id)):    # 笨办法
            os.remove(self.mdImgDownloadPath + "{}.png".format(id))
        n_img.save(fp = self.mdImgDownloadPath + "{}.png".format(id))
        print("Resize之后的md保存成功! ")


    def insert_into_mdFile(self, id, mdImage):
        self.save_img_to_md(id, mdImage)
        with open(self.mdImgDownloadPath + "{}.png".format(id), "rb") as image_file:
            image_data = image_file.read()           
        image_base64 = self.image_to_base64(image_data)
        imagetext = '![image](data:image/{};base64,{})'
        image_base64text = imagetext.format("png",image_base64)  ## ----> 这上面把图片转换为base64之后然后弄好插入的格式，等会儿直接搜索文章插入到头部

        with open(self.mdFilePath + "{}.txt".format(id), "r") as f:
            data = f.readlines()
        data.insert(0, image_base64text + '\n\n')

        if os.path.exists(self.mdFilePath + "{}.txt".format(id)) and (data != None or data != []):
            os.remove(self.mdFilePath + "{}.txt".format(id))
        
        with open(self.mdFilePath + "{}.txt".format(id), "w") as f:
            for eachLine in data:
                f.write(eachLine)




    def insert_into_thumbFile(self, id, thumbImage):
        self.save_img_to_downloaded_images(id, thumbImage)
        imagePath = self.thumbImgDownloadPath + "{}.png".format(id)
        base64Path = self.thumbFilePath + "{}.txt".format(id)
        self.save_base64(imagePath, base64Path)




    def processImage(self, id, imageUrl):
        url = 'http://' + imageUrl
        # import pdb
        # pdb.set_trace()
        imageContent = self.getImageContent(url)
        imageB64 = self.image_to_base64(imageContent)
        mdImage = self.resize_image_1000(id, imageB64)
        thumbImage = self.resize_image_300H_200W(id, imageB64)
        return mdImage, thumbImage

    ###############################                                结束                                    #########################################


    def removeExtraDir(self):              ### 删除多余文件夹
        if os.path.exists('/root/zhanxin/code/autocraw/recommend/{}/{}/downloaded_images_thumb'.format(search.pathDate, search.chanye)):
            shutil.rmtree('/root/zhanxin/code/autocraw/recommend/{}/{}/downloaded_images_thumb'.format(search.pathDate, search.chanye))
        if os.path.exists('/root/zhanxin/code/autocraw/recommend/{}/{}/downloaded_images_md'.format(search.pathDate, search.chanye)):
            shutil.rmtree('/root/zhanxin/code/autocraw/recommend/{}/{}/downloaded_images_md'.format(search.pathDate, search.chanye))



    def findNewsWithoutImg(self):           # 找到没有图片的新闻
        d = pd.read_csv(self.csvFile, usecols = ['序号', '热词'])
        print(d)                ### 拿到所有新闻
        newsList = []
        for i in range(len(d)):
            index = d['序号'][i]            # 新闻序号
            if not self.thumbFileSearch(index):
                newsList.append(index)
        
        return newsList
    

    def getVectorizedData(self):        # 这里会自动根据self.category判断新闻类别（新能源，人工智能等），然后自动从数据库里拿到关键词进行向量化
        if self.vectorizedCategory != self.category:        # 不用管这里，只需要修改self.category和self.chanye就行
            dataSet = self.dataBase.findReci(self.dataBase, "SELECT Keyword from {}".format(self.category))       # 拿到数据库里的热词准备对比
            print("数据库向量化中")
            start = time.time()
            self.bm25_retriever, self.vectordb, self.ranktokenizer, self.rankmodel = retriver.build_index(self.model, dataSet)  # 向量化
            end = time.time()
            print("向量化完毕, 用时：", end - start)
            self.vectorizedCategory = self.category




    def findSimilarImageUrl(self, id):            # ----> 返回相似词的图片url

        d = pd.read_csv(self.csvFile, usecols = ['序号', '热词'])
        self.getVectorizedData()        # 每次跑的时候判断我们拿到的向量化数据是否对应我们现在的新闻类别，如果不是的话会重新向量化

        imgFound = False
        index = d['序号'][id-1]            # 新闻序号
        hotwords = d['热词'][id-1]             # 新闻热词
        #print(index,hotwords)

        reciList, popularIndex = self.dataBase.extractCleanWords(hotwords)
        if reciList == None and popularIndex == None:
            reciList = [self.chanye]         
        else:
            print("热词列表： ", reciList)
            print("热点数: ", popularIndex)
            reciList = self.sorter(popularIndex, reciList)
            print("排序后的热词列表： ", reciList)

        hotestWord = reciList[0]  # ----> 如果说所有关键词的相似词图片都被用过了，那我们就用热度最高的关键词去找使用最久的图片

        for word in reciList:
            print("词语： ", word)
            bestMatch = retriver.search_index(self.bm25_retriever, self.vectordb, self.ranktokenizer, self.rankmodel, word) # 检索相似词
            print("相似词： ", bestMatch)

            query = "SELECT * from {} where Keyword = '{}';".format(self.category, bestMatch)
            keywordRows = self.dataBase.selectFromDatabase(query)
            for image in keywordRows:
                print(image)
                if image[5] == 0:
                    print("图片链接： ", image[2])
                    try:
                        mdImage, thumbImage = self.processImage(index, image[2])  # mdImage resize to 1000Width, thumb resize to 300*200
                        self.insert_into_mdFile(index, mdImage)
                        self.insert_into_thumbFile(index, thumbImage)
                        imgFound = True
                        break
                    except Exception as e:
                        print(str(e) + ", 图片文件插入失败")
            if imgFound == True:   ## 找到图片了就可以break掉loop，去下一个新闻找了
                print("图片已经生成并插入mdfile和thumbfile")
                break
            print("所有图片都被使用过了")  ## 如果没找到没有用过的图片，自动进入下面的if里面找使用最久的图片


        if not(imgFound):
            minimumFrequencyImage = "Select ImgFileUrl from {} A where Keyword = '{}' and not exists (select 1 from {} where Keyword = A.Keyword and offlineusetime < A.offlineusetime)".format(self.category, hotestWord, self.category)
            keyword = self.dataBase.selectFromDatabase(minimumFrequencyImage)
            try:
                mdImage, thumbImage = self.processImage(keyword[0], keyword[2])  # mdImage resize to 1000Width, thumb resize to 300*200
                self.insert_into_mdFile(index, mdImage)
                self.insert_into_thumbFile(index, thumbImage)
            except Exception as e:
                print(str(e) + ", 使用最久的图片插入失败")
            print("Returning: ", minimumFrequencyImage)
                                                    


if __name__ == "__main__":
    search = main()
    os.makedirs('/root/zhanxin/code/autocraw/recommend/{}/{}/downloaded_images_thumb'.format(search.pathDate, search.chanye), exist_ok=True)
    os.makedirs('/root/zhanxin/code/autocraw/recommend/{}/{}/downloaded_images_md'.format(search.pathDate, search.chanye), exist_ok=True)
    
    
    ############################## 这里是测试的代码，真正跑起来的时候请务必删掉哈！！！！！！！！！！！！ #########################################
    lst = search.findNewsWithoutImg()
    for id in lst:
        search.findSimilarImageUrl(id)
    search.removeExtraDir()
    ############################## 这里是测试的代码，真正跑起来的时候请务必删掉哈！！！！！！！！！！！！ #########################################

