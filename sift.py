from cv2 import KeyPoint
import numpy as np
from tools import *
float_tolerance = 1e-7

@Printsplit
def SIFT(Img, Sigma0=1.6, NumOfIntervals=3, AssumedBlur=0.5, ImgBorderWidth=5):
    """
    SIFT，流程和OpenCV一样
    @param Img: 输入灰度图
    @param Sigma0: SIFT里的Sigma0
    @param NumOfIntervals: SIFT里的大S
    @param AssumedBlur: 高斯模糊偏差，用于预先模糊2倍输入灰度图
    @param ImgBorderWidth: 抛弃的图片边缘的长度，便于计算，网上说一般为5
    @return: 关键点，特征
        关键点用OpenCV里的KeyPoint存储
        关键点的特征为128维
    """
    print('SIFT start!')
    GaussianPyramid = GenerateGaussianPyramid(Img, NumOfIntervals, AssumedBlur, Sigma0)
    DoGPyramid = GenerateDoGPyramid(GaussianPyramid)

    KeyPoints = FindKeyPoints(GaussianPyramid, DoGPyramid, NumOfIntervals, Sigma0, ImgBorderWidth)
    KeyPoints = RemoveDuplicateKeypoints(KeyPoints)

    Descriptors = GenerateDescriptors(KeyPoints, GaussianPyramid)
    print('SIFT Accomplish!')
    return KeyPoints, Descriptors

def GenerateGaussianPyramid(Img, NumOfIntervals=3, AssumedBlur=0.5, Sigma0=1.6):
    """
    生成灰度图的高斯金字塔
    @param Img: 输入灰度图
    @param NumOfIntervals: SIFT里的大S
    @param AssumedBlur: 高斯模糊偏差，用于SIFT里预先模糊2倍输入灰度图
    @param Sigma0: SIFT里的Sigma0（Octave0的SIgma0）
    @return: 高斯金字塔，用列表存储
    """
    # 图片准备
    print('Run GenerateGaussianPyramid')

    Img.astype(np.float64)
    ImgHeight, ImgWidth = len(Img), len(Img[0])

    DoubleShapeImg = Resize(Img, (2 * ImgWidth, 2 * ImgHeight))
    SigmaDiff = np.sqrt(max((Sigma0 ** 2) - ((2 * AssumedBlur) ** 2), 0.01))
    GaussianKernelSize = int((6 * SigmaDiff) // 2 * 2 + 1)
    Kernal_1XN, Kernal_NX1 = GenerateGaussianKernel1D(GaussianKernelSize, SigmaDiff)
    # # 分离高斯卷积
    BaseImg = Convolve(Convolve(DoubleShapeImg, Kernal_1XN), Kernal_NX1)
    print('---Shape:({}, {})'.format(2 * ImgHeight, 2 * ImgWidth))
    # 计算高斯金字塔组数
    ImgShape = np.shape(BaseImg)
    NumOfOctaves = np.int(np.round(np.log2(min(ImgShape)) - 1))
    print('---Number of Octaves is', NumOfOctaves)
    # 第0组Sigma准备
    print('---SigmasOfOctave0:{:.2f}  '.format(Sigma0), end='')
    NumPerOctave = NumOfIntervals + 3
    k = 2 ** (1. / NumOfIntervals)

    GaussianKernelsSigmas = [Sigma0]

    for ImgIndex in range(1, NumPerOctave):
        SigmaPrevious = (k ** (ImgIndex - 1)) * Sigma0
        SigmaTotal = k * SigmaPrevious
        Sigma = np.sqrt(SigmaTotal ** 2 - SigmaPrevious ** 2)
        print('{:.2f}  '.format(Sigma), end='')
        GaussianKernelsSigmas.append(Sigma)
    print(' ')
    # 计算高斯金字塔
    GaussianPyramid = []
    Img = BaseImg

    for OctaveIndex in range(NumOfOctaves):
        GaussianImgInOctave = []
        GaussianImgInOctave.append(Img)
        for GaussianKernelSigma in GaussianKernelsSigmas[1:]:

            GaussianKernelSize = int((6 * GaussianKernelSigma) // 2 * 2 + 1)
            Kernal_1XN, Kernal_NX1 = GenerateGaussianKernel1D(GaussianKernelSize, GaussianKernelSigma)

            Img = Convolve(Convolve(Img, Kernal_1XN), Kernal_NX1)
            GaussianImgInOctave.append(Img)

        GaussianPyramid.append(GaussianImgInOctave)
        NextOctaveBaseImg = GaussianImgInOctave[-3]
        Img = DownSampling(NextOctaveBaseImg)
        print('\r---GaussianPyramidOctave{} Accomplish'.format(OctaveIndex - 1), end='', flush=True)
    print('')
    return np.array(GaussianPyramid)
def GenerateDoGPyramid(GaussianPyramid):
    """
    差分高斯金字塔
    @param GaussianPyramid: 高斯金字塔
    @return: 差分高斯金字塔
    """
    print('Run GenerateDoGPyramid')
    DoGPyramid = []
    DoGOctaveIndex = 0

    for GaussianImgInOctave in GaussianPyramid:
        DoGImgInOctave = []
        for FirstImg, SecondImg in zip(GaussianImgInOctave, GaussianImgInOctave[1:]):
            DoGImgInOctave.append(np.array(SecondImg - FirstImg))

        DoGPyramid.append(DoGImgInOctave)
        print('\r---DoGOctave{} Accomplish'.format(DoGOctaveIndex - 1), end='', flush=True)
        DoGOctaveIndex += 1
    print('')
    return np.array(DoGPyramid)

def FindKeyPoints(GaussianPyramid, DoGPyramid, NumOfIntervals, Sigma0, ImgBorderWidth, ContrastThreshold=0.04):
    """
    找到关键点
    @param GaussianPyramid: 灰度图的高斯金字塔，找关键点方向时要用
    @param DoGPyramid: 灰度图的差分高斯金字塔，找关键点要用
    @param NumOfIntervals: SIFT里的s
    @param Sigma0: SIFT里的Sigma0
    @param ImgBorderWidth: 抛弃的图片边缘的长度，便于计算，网上说一般为5
    @param ContrastThreshold: SIFT里用于消除噪音点
    @return: 关键点列表，里面的关键点用OpenCV的KeyPoint存储
    """
    print('Run FindKeyPoints')
    Threshold = np.floor(0.5 * ContrastThreshold / NumOfIntervals * 255) # 与OpenCV一致
    KeyPoints = []
    SumOfKeyPoints = 0 # 关键点总数
    OctaveStatistic = np.zeros(len(DoGPyramid)) # 各个Octave内的关键点数统计

    for OctaveIndex, DoGImgInOctave in enumerate(DoGPyramid):
        for ImgIndex in range(1, len(DoGImgInOctave)-1):
            DoGImgs = [DoGImgInOctave[ImgIndex - 1], DoGImgInOctave[ImgIndex], DoGImgInOctave[ImgIndex + 1]] # 差分金字塔内的三张图片
            ImgHeight, ImgWidth = np.shape(DoGImgInOctave[ImgIndex])
            for i in range(ImgBorderWidth, ImgHeight - ImgBorderWidth):
                for j in range(ImgBorderWidth, ImgWidth - ImgBorderWidth):
                    DoGArray = np.array([DoGImg[i - 1:i + 2, j - 1:j + 2] for DoGImg in DoGImgs]) # DoG数组（3X3X3）
                    if IsExtremum(DoGArray, Threshold):
                        Keypoint = DetecteKeyPoint(i, j, ImgIndex, OctaveIndex, NumOfIntervals, DoGImgInOctave, Sigma0, ContrastThreshold, ImgBorderWidth)
                        if Keypoint is not None:
                            Keypoint, LocalizedImgIndex = Keypoint # 关键点
                            KeyPointWithOrientations = ComputeKeypointsWithOrientations(Keypoint, OctaveIndex,GaussianPyramid[OctaveIndex][LocalizedImgIndex])
                            for KeyPointWithOrientation in KeyPointWithOrientations: # 带方向的关键点
                                SumOfKeyPoints += 1
                                OctaveStatistic[OctaveIndex] += 1
                                print('\r---Add Octave{}({}, {}), Sum:{}'.format(OctaveIndex, i, j, SumOfKeyPoints),
                                      end='', flush=True)
                                KeyPoints.append(KeyPointWithOrientation)
    print(' ')
    for i in range(len(OctaveStatistic)):
        print('---Octave{}，Sum:{}'.format(i, OctaveStatistic[i]))
    return KeyPoints
def IsExtremum(DoGArray, Threshold=None):
    """
    判断是否是极值点
    @param DoGArray: DoG数组（3X3X3）
    @param Threshold: 阈值，消除噪音点
    @return: 返回True或False
    """
    if Threshold is not None:
        if abs(DoGArray[1, 1, 1]) <= Threshold:
            return False
    Max = DoGArray[1, 1, 1]
    MaxIndex = [1, 1, 1]
    Min = DoGArray[1, 1, 1]
    MinIndex = [1, 1, 1]
    for i in range(3):
        for j in range(3):
            for k in range(3):
                if DoGArray[i, j, k] > Max:
                    Max = DoGArray[i, j, k]
                    MaxIndex = [i, j, k]
                elif DoGArray[i, j, k] < Min:
                    Min = DoGArray[i, j, k]
                    MinIndex = [i, j, k]
    if MaxIndex == [1, 1, 1] or MinIndex == [1, 1, 1]:
        return True
    else:
        return False
def DetecteKeyPoint(i, j, ImgIndex, OctaveIndex, NumOfIntervals, DoGImgInOctave, Sigma0, ContrastThreshold,
                    ImgBorderWidth, EigenvalueRatio=10, NumOfAttempts=5):
    """
    探测极值点是否是关键点
    @param i: 极值点坐标i
    @param j: 极值点坐标j
    @param ImgIndex: 图片的层数s
    @param OctaveIndex: 图片所在组数O
    @param NumOfIntervals: SIFT里的大S
    @param DoGImgInOctave: 差分高斯金字塔的一组图片，探测时可能移动层数
    @param Sigma0: SIFT里的Sigma0
    @param ContrastThreshold: SIFT里用于消除噪音点
    @param ImgBorderWidth: 抛弃的图片边缘的长度，便于计算，网上说一般为5
    @param EigenvalueRatio: 消除边缘响应的阈值里的r
    @param NumOfAttempts: 关键点的探测次数
    @return: 如果探测到关键点就返回KeyPoint，探测不到就返回None
    """
    ImgShape = (len(DoGImgInOctave[0]), len(DoGImgInOctave[0][0]))
    for AttemptIndex in range(NumOfAttempts):
        FirstImg, SecondImg, ThirdImg = DoGImgInOctave[ImgIndex - 1:ImgIndex + 2]
        DoGArray = np.array([FirstImg[i - 1:i + 2, j - 1:j + 2],
                             SecondImg[i - 1:i + 2, j - 1:j + 2],
                             ThirdImg[i - 1:i + 2, j - 1:j + 2]]).astype(np.float) / 255 # Lowe要求归一化，为了跟阈值比较，消除噪音点

        Gradient = ComputeGradientAtCenterPixel(DoGArray)
        Hessian = ComputeHessianAtCenterPixel(DoGArray)
        CoordinatesUpdateValueOfKeyPoint = -1 * np.dot(ComputeInv3(Hessian), np.array([Gradient]).reshape((3, 1)))
        CoordinatesUpdateValueOfKeyPoint.reshape(-1) # 偏移量

        if abs(CoordinatesUpdateValueOfKeyPoint[0]) < 0.5 and abs(CoordinatesUpdateValueOfKeyPoint[1]) < 0.5 and abs(
                CoordinatesUpdateValueOfKeyPoint[2]) < 0.5:
            break
        j += int(np.round(CoordinatesUpdateValueOfKeyPoint[0]))
        i += int(np.round(CoordinatesUpdateValueOfKeyPoint[1]))
        ImgIndex += int(np.round(CoordinatesUpdateValueOfKeyPoint[2]))
        if i < ImgBorderWidth or i >= ImgShape[0] - ImgBorderWidth or j < ImgBorderWidth or j >= ImgShape[1] - ImgBorderWidth or ImgIndex < 1 or ImgIndex > NumOfIntervals:
            return None
    if AttemptIndex >= NumOfAttempts - 1:
        return None

    NewKeyPointValue = DoGArray[1, 1, 1] + 0.5 * np.dot(Gradient, CoordinatesUpdateValueOfKeyPoint)

    if abs(NewKeyPointValue) * NumOfIntervals >= ContrastThreshold:
        xy_Hessian = Hessian[:2, :2]
        Tr = xy_Hessian[0, 0] + xy_Hessian[1, 1]
        Det = xy_Hessian[0, 0] * xy_Hessian[1, 1] - (xy_Hessian[0, 1] ** 2)
        if Det > 0 and EigenvalueRatio * (Tr ** 2) < ((EigenvalueRatio + 1) ** 2) * Det:
            Keypoint = KeyPoint()
            Keypoint.pt = ((j + CoordinatesUpdateValueOfKeyPoint[0]) * (2 ** OctaveIndex),
                           (i + CoordinatesUpdateValueOfKeyPoint[1]) * (2 ** OctaveIndex))
            # Keypoint.pt 要以初始图片大小为准，这里的初始图片是扩大二倍后的图片
            Keypoint.octave = OctaveIndex + ImgIndex * (2 ** 8) + np.int(np.round((CoordinatesUpdateValueOfKeyPoint[2]) * 255)) * (2 ** 16)
            # Keypoint.octave 0-7位保存Octave，8-15位保存ImgIndex， 16-23位保存sigma，防止在后面用Sigma返过来求s时，s直接取整会出错
            Keypoint.size = Sigma0 * (2 ** ((ImgIndex + CoordinatesUpdateValueOfKeyPoint[2]) / np.float32(NumOfIntervals))) * (2 ** (OctaveIndex + 1))
            # Keypoint.size 这里的初始图片是扩大二倍后的图片，各个Sigma要多乘一个2, 所以OctaveIndex + 1
            Keypoint.response = abs(NewKeyPointValue)
            # 为什么不直接从初始图片开始算？SIFT关键点主要在-1组，具体在报告中介绍

            return Keypoint, int(ImgIndex + CoordinatesUpdateValueOfKeyPoint[2])
    return None
def ComputeKeypointsWithOrientations(Keypoint, OctaveIndex, GaussianImg, LenOfHistogram=36, PeakRatio=0.8):
    """
    计算关键点方向
    @param Keypoint: 关键点
    @param OctaveIndex: 关键点所在组数，用于求Sigma_oct
    @param GaussianImg: 关键点所在高斯金字塔的图片
    @param LenOfHistogram: 梯度直方图的长度这里是36，一个柱10度
    @param PeakRatio: 大于主方向的PeakRatio倍，也要存储，Lowe推荐0.8
    @return: 带方向的极值点KeyPoint
    """
    KeyPointWithOrientations = []
    ImgShape = np.shape(GaussianImg)

    Sigma_oct = 1.5 * Keypoint.size / np.float32(2 ** (OctaveIndex + 1))
    Radius = int(np.round(3 * Sigma_oct))
    WeightFactor = -0.5 / (Sigma_oct ** 2)
    Histogram = np.zeros(LenOfHistogram)
    SmoothHistogram = np.zeros(LenOfHistogram)

    for i in range(-Radius, Radius + 1):
        Region_y = int(np.round(Keypoint.pt[1] / np.float32(2 ** OctaveIndex))) + i
        if Region_y > 0 and Region_y < ImgShape[0] - 1:
            for j in range(-Radius, Radius + 1):
                Region_x = int(np.round(Keypoint.pt[0] / np.float32(2 ** OctaveIndex))) + j
                if Region_x > 0 and Region_x < ImgShape[1] - 1:
                    if np.sqrt(i**2 + j**2) < Radius: # 圆形区域内
                        dx = GaussianImg[Region_y, Region_x + 1] - GaussianImg[Region_y, Region_x - 1]
                        dy = GaussianImg[Region_y - 1, Region_x] - GaussianImg[Region_y + 1, Region_x]
                        Magnitude = np.sqrt(dx * dx + dy * dy)
                        Orientation = np.rad2deg(np.arctan2(dy, dx))  # 1-360

                        Weight = np.exp(WeightFactor * (i ** 2 + j ** 2))
                        HistogramIndex = int(np.round(Orientation * LenOfHistogram / 360.)) % LenOfHistogram
                        Histogram[HistogramIndex] = Histogram[HistogramIndex] + Weight * Magnitude
    # 在直方图统计时，每相邻三个像素点采用高斯加权，根据Lowe的建议，模板采用[0.25,0.5,0.25],并且连续加权两次.
    for i in range(2):
        for n in range(LenOfHistogram):
            if n == 0:
                SmoothHistogram[n] = 0.5 * Histogram[n] + 0.25 * (Histogram[LenOfHistogram - 1] + Histogram[n + 1])
            if n == LenOfHistogram - 1:
                SmoothHistogram[n] = 0.5 * Histogram[n] + 0.25 * (Histogram[n - 1] + Histogram[0])
            else:
                SmoothHistogram[n] = 0.5 * Histogram[n] + 0.25 * (Histogram[n - 1] + Histogram[n + 1])
        Histogram = SmoothHistogram

    OrientationMax = max(SmoothHistogram)
    SmoothHistogramPeaks = np.where(
        np.logical_and(SmoothHistogram > np.roll(SmoothHistogram, 1), SmoothHistogram > np.roll(SmoothHistogram, -1)))[
        0] # 找平滑后的直方图的极值点

    for PeakIndex in SmoothHistogramPeaks:
        PeakValue = SmoothHistogram[PeakIndex]
        if PeakValue >= PeakRatio * OrientationMax: # 统计主方向和大于0.8倍主方向的方向
            LeftValue = SmoothHistogram[(PeakIndex - 1) % LenOfHistogram]
            RightValue = SmoothHistogram[(PeakIndex + 1) % LenOfHistogram]

            InterpolatedPeakIndex = 0.5 * (LeftValue - RightValue) / (LeftValue - 2 * PeakValue + RightValue)
            Orientation = (PeakIndex + InterpolatedPeakIndex) * (360 / LenOfHistogram)
            Orientation = 360 - Orientation
            NewkeyPoint = KeyPoint(*tuple(0.5 * np.array(Keypoint.pt)), 0.5 * Keypoint.size, Orientation, Keypoint.response, Keypoint.octave)
            # 这里的KeyPoint的极值点就全部检测完成了，为回到初始图片的大小所以乘以0.5
            KeyPointWithOrientations.append(NewkeyPoint)
    return KeyPointWithOrientations

def RemoveDuplicateKeypoints(KeyPoints):
    """
    消除重复的极值点
    @param KeyPoints: 关键点列表
    @return: 去除重复关键点的列表
    """
    print('Run RemoveDuplicateKeypoints')
    if len(KeyPoints) < 2:
        return KeyPoints
    NeedDel = []
    Unique = []
    LenOfKeyPoints = len(KeyPoints)
    for i in range(LenOfKeyPoints):
        for j in range(i+1, LenOfKeyPoints):
            if KeyPoints[i].pt[0] == KeyPoints[j].pt[0] and KeyPoints[i].pt[1] == KeyPoints[j].pt[1] and KeyPoints[i].angle == KeyPoints[j].angle:
                NeedDel.append(i)
                print('\r---del(x, y):({:.2f}, {:.2f})'.format(*KeyPoints[i].pt), end='', flush=True)
    print(' ')
    for i in range(LenOfKeyPoints):
        if i not in NeedDel:
            Unique.append(KeyPoints[i])
    if len(Unique) == LenOfKeyPoints:
        print('---No points to delete')
    else:
        print('---Del {},Sum {}'.format(LenOfKeyPoints - len(Unique), len(Unique)))
    return Unique
def GenerateDescriptors(KeyPoints, GaussianPyramid, WindowWidth=4, LenOfHistogram=8, ScaleMultiplier=3, DescriptorMaxValue=0.2):
    """
    计算关键点特征
    @param KeyPoints: 关键点列表
    @param GaussianPyramid: 高斯金字塔，计算特征在高斯金字塔上算
    @param Sigma0: SIFT里的Sigma0
    @param NumOfIntervals: SIFT里的大S
    @param WindowWidth: 描述子所需区域的划分边长，Lowe推荐为4
    @param LenOfHistogram: 梯度直方图的长度，这里为8，一个柱45度
    @param ScaleMultiplier: 描述子所需区域的半径的系数
    @param DescriptorMaxValue: 消除光照变化的影响的阈值
    @return: 128维特征
    """
    print('Run GenerateDescriptors')
    Descriptors = []

    for Keypoint in KeyPoints:
        Octave = Keypoint.octave & 255
        if Octave >= 128:
            Octave = Octave | -128
        Layer = (Keypoint.octave >> 8) & 255

        GaussianImg = GaussianPyramid[Octave + 1][Layer]
        ImgHeight, ImgWidth = np.shape(GaussianImg)

        Point = (np.round(np.array(Keypoint.pt) / (np.float32(2 ** Octave) if Octave >= 0 else np.float32(2 ** (-Octave))))).astype(np.int) # 从原始图片尺寸回到取样高斯图片尺寸上
        Angle = 360. - Keypoint.angle
        cos = np.cos(np.deg2rad(Angle))
        sin = np.sin(np.deg2rad(Angle))
        KeyPointInfomations = [] # 关键点信息，包括（4X4方格内的位置(x’’,y’’)，旋转后的相对坐标(x’,y’)，加权后的幅值，方向的直方图坐标）
        HistogramTensor = np.zeros((WindowWidth + 2, WindowWidth + 2, LenOfHistogram))

        ThreeSigma_oct = ScaleMultiplier * 0.5  * Keypoint.size / (np.float32(2 ** Octave) if Octave >= 0 else np.float32(2 ** (-Octave)))
        Radius = int(np.round(ThreeSigma_oct * np.sqrt(2) * (WindowWidth + 1) * 0.5))
        for i in range(-Radius, Radius + 1):
            for j in range(-Radius, Radius + 1):
                yi = j * sin + i * cos
                xi = j * cos - i * sin
                yii = (yi / ThreeSigma_oct) + 0.5 * WindowWidth - 0.5
                xii = (xi / ThreeSigma_oct) + 0.5 * WindowWidth - 0.5
                if yii > -1 and yii < WindowWidth and xii > -1 and xii < WindowWidth:
                    WindowRow = int(np.round(Point[1] + i))
                    WindowCol = int(np.round(Point[0] + j))
                    if WindowRow > 0 and WindowRow < ImgHeight - 1 and WindowCol > 0 and WindowCol < ImgWidth - 1:
                        dx = GaussianImg[WindowRow, WindowCol + 1] - GaussianImg[WindowRow, WindowCol - 1]
                        dy = GaussianImg[WindowRow - 1, WindowCol] - GaussianImg[WindowRow + 1, WindowCol]
                        Magnitude = np.sqrt(dx * dx + dy * dy)
                        Orientation = np.rad2deg(np.arctan2(dy, dx)) % 360
                        Weight = np.exp(-0.5 / ((0.5 * WindowWidth) ** 2) * ((yi / ThreeSigma_oct) ** 2 + (xi / ThreeSigma_oct) ** 2))
                        KeyPointInfomations.append((yii, xii, Weight * Magnitude, (Orientation - Angle) * (LenOfHistogram / 360.)))

        for yii, xii, Magnitude, OrientationBin in KeyPointInfomations:
            yii_floor, xii_floor, o_floor = np.floor([yii, xii, OrientationBin]).astype(np.int)
            dr, dc, do = yii - yii_floor, xii - xii_floor, OrientationBin - o_floor
            if o_floor < 0:
                o_floor += LenOfHistogram
            if o_floor >= LenOfHistogram:
                o_floor -= LenOfHistogram

            HistogramTensor[yii_floor + 1, xii_floor + 1, o_floor % LenOfHistogram] += Magnitude * (1 - dr) * (1 - dc) * (1 - do)
            HistogramTensor[yii_floor + 1, xii_floor + 1, (o_floor + 1) % LenOfHistogram] += Magnitude * (1 - dr) * (1 - dc) * do
            HistogramTensor[yii_floor + 1, xii_floor + 2, o_floor % LenOfHistogram] += Magnitude * (1 - dr) * dc * (1 - do)
            HistogramTensor[yii_floor + 1, xii_floor + 2, (o_floor + 1) % LenOfHistogram] += Magnitude * (1 - dr) * dc * do
            HistogramTensor[yii_floor + 2, xii_floor + 1, o_floor % LenOfHistogram] += Magnitude * dr * (1 - dc) * (1 - do)
            HistogramTensor[yii_floor + 2, xii_floor + 1, (o_floor + 1) % LenOfHistogram] += Magnitude * dr * (1 - dc) * do
            HistogramTensor[yii_floor + 2, xii_floor + 2, o_floor % LenOfHistogram] += Magnitude * dr * dc * (1 - do)
            HistogramTensor[yii_floor + 2, xii_floor + 2, (o_floor + 1) % LenOfHistogram] += Magnitude * dr * dc * do

        DescriptorVector = HistogramTensor[1:-1, 1:-1, :].flatten()
        # 消除光照影响
        Threshold = np.linalg.norm(DescriptorVector) * DescriptorMaxValue
        DescriptorVector[DescriptorVector > Threshold] = Threshold
        DescriptorVector /= max(np.linalg.norm(DescriptorVector), float_tolerance) # 与OpenCV一致
        # 仿照OpenCV
        DescriptorVector = np.round(512 * DescriptorVector)
        DescriptorVector[DescriptorVector < 0] = 0
        DescriptorVector[DescriptorVector > 255] = 255
        Descriptors.append(DescriptorVector)
    return np.array(Descriptors, dtype='float32')

