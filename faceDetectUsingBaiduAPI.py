import requests
import base64
import json

# 这是一个使用百度官方api的方法

def get_token():
    # client_id 为官网获取的AK， client_secret 为官网获取的SK
    host = 'https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id=8KqRe4zsSVsZz6kISueguHPL&client_secret=OE7Pat1VRFGa8o4lf1vvYvT2W07uwjuK'
    response = requests.get(host)
    access_token = eval(response.text)['access_token']
    content = "https://aip.baidubce.com/rest/2.0/face/v3/match" + "?access_token=" + access_token
    if (content):
        print(content)
    return content


def read_img(img1,img2):
    with open(img1,'rb') as f:
        pic1=base64.b64encode(f.read())
    with open(img2,'rb') as f:
        pic2=base64.b64encode(f.read())
    params=json.dumps([
        {"image":str(pic1,"utf-8"),"image_type":'BASE64',"face_type":"LIVE"},
        {"image":str(pic2,"utf-8"),"image_type":'BASE64',"face_type":"IDCARD"}
    ])
    return params

# this is a url which we can use
api="https://aip.baidubce.com/rest/2.0/face/v3/match?access_token=24.0fe973ec910ef4518dadc824d81862ae.2592000.1562742611.282335-16470780"


# 4，发起请求拿到对比结果
def analyse_img(file1,file2):
    params=read_img(file1,file2)
    api=get_token()
    content=requests.post(api,params).text
    # print(content)
    score=eval(content)['result']['score']
    if score>80:
        print('图片识别相似度度为'+str(score)+',是同一人')
    else:
        print('图片识别相似度度为'+str(score)+',不是同一人')


if __name__ == '__main__':
    # 这是一个人脸对比的example
    analyse_img("testdata/3a.jpg","testdata/3b.jpg")