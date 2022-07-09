# 不调用api，代码实现sift算子，并进行全景图片拼接
流程与opencv的sift.cpp大致一样

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

# 报告内容

Python实现（包括卷积缩放等），拼接用opencv，DUT图像处理基础大作业


## SIFT概述
- SIFT 算法的特点
- SIFT 算法可以解决的问题
- SIFT 算法步骤

## SIFT详细流程及编程思路

- 尺度空间极值检测
  - 高斯模糊
  - 尺度空间理论
  - 空间极值点检测(关键点的初步探查）
  
- 关键点定位
  - 关键点的精确定位
  - 消除边缘响应

- 关键点的方向分配
  - 幅值和角度
  - 拟合二次曲线过程
  - 关键点结果展示
  - 消除重复的关键点

- 关键点特征描述
  - 确定计算描述子所需的图像区域
  - 将坐标轴旋转为关键点的方向（旋转不变性）
  - 梯度直方图的生成（HOG）
  - 三线性插值
  - 归一化特征向量
 
- SIFT特点总结

- SIFT参考资料

## 图片拼接

- 关键点匹配
  - Lowe's算法
  - RANSAC算法
  - 匹配结果

- 图像配准
- 配准结果

