# 全景图片拼接

```
import imageio
# 读取拼接图片
imageA = imageio.imread("11.jpg")
imageB = imageio.imread("22.jpg")

imageA = cv2.resize(imageA, (256, 256))
imageB = cv2.resize(imageB, (256, 256))

tools.ShowPict(imageA)
tools.ShowPict(imageB)

# 把图片拼接成全景图
stitcher = Stitcher()
(result, vis) = stitcher.stitch([imageA, imageB], showMatches=True)

tools.ShowPict(vis)
tools.ShowPict(result)
```

Python实现（包括卷积缩放等），拼接用opencv，DUT图像处理基础大作业


一 SIFT概述	2

1 SIFT 算法的特点	2

2 SIFT 算法可以解决的问题	2

3 SIFT 算法步骤	2

二 SIFT详细流程及编程思路	4

1 尺度空间极值检测	4

1.1 高斯模糊	4

1.2 尺度空间理论	7

1.3 空间极值点检测(关键点的初步探查）	16

2 关键点定位	17

2.1 关键点的精确定位	17

2.2 消除边缘响应	23

3 关键点的方向分配	24

3.1 幅值和角度	24

3.2 拟合二次曲线过程	25

3.3 关键点结果展示	26

3.4 消除重复的关键点	28

4 关键点特征描述	29

4.1确定计算描述子所需的图像区域	29

4.2 将坐标轴旋转为关键点的方向（旋转不变性）	30

4.3 梯度直方图的生成	30

4.4 三线性插值	31

4.5 归一化特征向量	33

5 SIFT特点总结	33

6 SIFT参考资料	34

三 图片拼接	35

1 关键点匹配	35

1.1 Lowe's算法	35

1.2 RANSAC算法	35

1.3 匹配结果	36

2 图像配准	37

3 配准结果	37

四 代码	40

tools.py代码	41

sift.py代码	49

splice.py代码	60

