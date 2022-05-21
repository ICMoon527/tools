import base64
f=open('D:\Study\PHD4\ASD\ASD\Doc\shared space结构.png','rb') #二进制方式打开图文件
ls_f=base64.b64encode(f.read()) #读取文件内容，转换为base64编码
f.close()
print(ls_f)

# # 将base转化为图片
# import base64
# bs='iVBORw0KGgoAAAANSUhEUg....' # 太长了省略
# imgdata=base64.b64decode(bs)
# file=open('2.jpg','wb')
# file.write(imgdata)
# file.close()