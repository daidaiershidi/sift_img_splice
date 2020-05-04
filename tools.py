import numpy as np
def Timing(f):
    """
    函数装饰器，计时器，输出函数运行时间
    @param f: 被计时函数
    @return: 被计时函数运行结果
    """
    def rf(*args, **kwargs):
        import time
        start_time = time.time()
        result = f(*args, **kwargs)
        end_time = time.time()
        print('用时：{}ms'.format((end_time - start_time) * 1000))
        return result
    return rf
def ShowPict(pict, Gray=False):
    """
    函数装饰器，显示图片，可选择是否按照灰度图显示
    @param pict: 要显示的图片
    @param Gray: 是否按照灰度图显示
    """
    import matplotlib.pyplot as plt
    if Gray:
        plt.imshow(pict.astype(np.float), cmap='gray')
    else:
        plt.imshow(pict.astype(np.int))
    plt.axis('off')
    plt.show()
def Printsplit(f):
    """
    函数装饰器，对函数输出进行分隔
    ---------------------------------
        （中间是函数自身的输出结果）
    ---------------------------------
    @param f: 被装饰的函数
    @return: 被装饰的函数的运行结果
    """
    def rf(*args, **kwargs):
        print('-' * 30)
        result = f(*args, **kwargs)
        print('-'*30)
        return result
    return rf

def Resize(OriginalImage, TargetSize):
    """
    单通道图片进行缩放
    @param OriginalImage: 原始图片
    @param TargetSize: 目标图片的大小，格式为（宽，高）
    @return: 缩放后的单通道图片
    """
    OriginalImageWidth, OriginalImageHeight = len(OriginalImage[0]), len(OriginalImage)
    TargetWidth, TargetHeight = TargetSize[0], TargetSize[1]
    WidthScale = OriginalImageWidth / TargetWidth
    HeightScale = OriginalImageHeight / TargetHeight
    TargetImage = np.zeros((TargetHeight, TargetWidth), dtype=np.uint8)
    for i in range(TargetHeight):
        for j in range(TargetWidth):
            x, y = (i + 0.5) * HeightScale - 0.5, (j + 0.5) * WidthScale - 0.5
            x1, y1 = int(x), int(y)
            x2, y2 = min(x1 + 1, OriginalImageHeight - 1), min(y1 + 1, OriginalImageWidth - 1)
            E1 = (x - x1) * (int(OriginalImage[x2][y1]) - int(OriginalImage[x1][y1])) + int(OriginalImage[x1][y1])
            E2 = (x - x1) * (int(OriginalImage[x2][y2]) - int(OriginalImage[x1][y2])) + int(OriginalImage[x1][y2])
            f = (y - y1) * (E2 - E1) + E1
            TargetImage[i][j] = f
    return TargetImage
def GenerateGaussianKernel2D(KernelSize=3, Sigma=0):
    """
    生成2维高斯卷积核
    @param KernelSize:高斯核边长
    @param Sigma:Sigma
    @return: 2维高斯卷积核
    """
    Kernel = np.zeros([KernelSize, KernelSize])
    Center = KernelSize // 2
    if Sigma == 0:
        Sigma = ((KernelSize - 1) * 0.5 - 1) * 0.3 + 0.8
    s = 2 * (Sigma ** 2)
    SumValue = 0
    for i in range(0, KernelSize):
        for j in range(0, KernelSize):
            x = i - Center
            y = j - Center
            Kernel[i, j] = np.exp(-(x ** 2 + y ** 2) / s)
            SumValue += Kernel[i, j]
    SumValue = 1 / SumValue
    return Kernel * SumValue
def GenerateGaussianKernel1D(KernelSize=3, Sigma=0):
    """
    生成两个分离高斯卷积核
    @param KernelSize:高斯核边长
    @param Sigma:Sigma
    @return: 两个分离高斯卷积核，大小为1XN和NX1
    """
    Kernel = np.zeros([1,KernelSize])
    Center = KernelSize // 2
    if Sigma == 0:
        Sigma = ((KernelSize - 1) * 0.5 - 1) * 0.3 + 0.8
    s = 2 * (Sigma ** 2)
    SumValue = 0
    for i in range(0, KernelSize):
            x = i - Center
            Kernel[0, i] = np.exp(-(x ** 2) / s)
            SumValue += Kernel[0, i]
    SumValue = 1 / SumValue
    Kernal1XN = Kernel * SumValue
    KernalNX1 = Kernal1XN.T
    return Kernal1XN, KernalNX1

def ComputeGradientAtCenterPixel(DoGArray):
    """
    求DoG矩阵（3X3X3）中心点的梯度
    @param DoGArray: DoG矩阵（3X3X3）
    @return: 中心点的梯度[dx, dy, ds]
    """
    dx = 0.5 * (DoGArray[1, 1, 2] - DoGArray[1, 1, 0])
    dy = 0.5 * (DoGArray[1, 2, 1] - DoGArray[1, 0, 1])
    ds = 0.5 * (DoGArray[2, 1, 1] - DoGArray[0, 1, 1])
    return np.array([dx, dy, ds])
def ComputeHessianAtCenterPixel(DoGArray):
    """
    求DoG矩阵（3X3X3）中心点的hessian阵
    @param DoGArray: DoG矩阵（3X3X3）
    @return: 中心点的hessian阵 [[dxx, dxy, dxs],
                               [dxy, dyy, dys],
                               [dxs, dys, dss]]
    """
    CenterPixelValue = DoGArray[1, 1, 1]
    dxx = DoGArray[1, 1, 2] - 2 * CenterPixelValue + DoGArray[1, 1, 0]
    dyy = DoGArray[1, 2, 1] - 2 * CenterPixelValue + DoGArray[1, 0, 1]
    dss = DoGArray[2, 1, 1] - 2 * CenterPixelValue + DoGArray[0, 1, 1]
    dxy = 0.25 * (DoGArray[1, 2, 2] - DoGArray[1, 2, 0] - DoGArray[1, 0, 2] + DoGArray[1, 0, 0])
    dxs = 0.25 * (DoGArray[2, 1, 2] - DoGArray[2, 1, 0] - DoGArray[0, 1, 2] + DoGArray[0, 1, 0])
    dys = 0.25 * (DoGArray[2, 2, 1] - DoGArray[2, 0, 1] - DoGArray[0, 2, 1] + DoGArray[0, 0, 1])
    return np.array([[dxx, dxy, dxs],
                  [dxy, dyy, dys],
                  [dxs, dys, dss]])

def ComputeDet3(Arr):
    """
    求3X3矩阵的行列式
    @param Arr: 3X3矩阵
    @return: 3X3矩阵的行列式
    """
    return Arr[0,0]*Arr[1,1]*Arr[2,2] + Arr[1,0]*Arr[2,1]*Arr[0,2] + Arr[2,0]*Arr[0,1]*Arr[1,2] - Arr[0,0]*Arr[2,1]*Arr[1,2] - Arr[2,0]*Arr[1,1]*Arr[0,2] - Arr[1,0]*Arr[0,1]*Arr[2,2]
def ComputeInv3(Arr):
    """
    求3X3矩阵的逆矩阵
    @param Arr: 3X3矩阵
    @return: 3X3矩阵的逆矩阵
    """
    det = ComputeDet3(Arr)
    return np.array([[Arr[1,1]*Arr[2,2]-Arr[1,2]*Arr[2,1], Arr[0,2]*Arr[2,1]-Arr[0,1]*Arr[2,2], Arr[0,1]*Arr[1,2]-Arr[0,2]*Arr[1,1]],
                     [Arr[1,2]*Arr[2,0]-Arr[1,0]*Arr[2,2], Arr[0,0]*Arr[2,2]-Arr[0,2]*Arr[2,0], Arr[0,2]*Arr[1,0]-Arr[0,0]*Arr[1,2]],
                     [Arr[1,0]*Arr[2,1]-Arr[1,1]*Arr[2,0], Arr[0,1]*Arr[2,0]-Arr[0,0]*Arr[2,1], Arr[0,0]*Arr[1,1]-Arr[0,1]*Arr[1,0]]])/det

def GenerateGrayImg(Img):
    """
    将3通道图片转换成灰度图
    @param Img: 3通道或1通道图片
    @return: 输入图片的灰度图
    """
    ImgShape = np.shape(Img)
    if len(ImgShape) == 3:
        r, g, b = [Img[:, :, i] for i in range(3)]
        GrayImg = r * 0.299 + g * 0.587 + b * 0.114
    if len(ImgShape) == 2:
        GrayImg = Img
    return GrayImg
def Convolve(Img, Filter, Mode='AdjustWeight'):
    """
    对1通道或3通道图片进行卷积
    @param Img: 图片数组
    @param Filter: 卷积核，边长必须为大于0的单数，必须是二维
    @param Mode: 边缘处理方式（Symmetry,AdjustWeight）
    @return: 输出卷积后的图片，大小不变
    """
    ImgShape = np.shape(Img)
    Filter = np.array(Filter, dtype=np.float)
    ConvImg = np.zeros_like(Img, dtype=np.float)
    if len(ImgShape) == 3:
        for i in range(3):
            ConvImg[:, :, i] = _Convolve(Img[:, :, i], Filter, Mode)
    if len(ImgShape) == 2:
        ConvImg = _Convolve(Img, Filter, Mode)
    return ConvImg
def _Convolve(Img, Filter, Mode):
    """
    单通道图片卷积
    @param Img: 图片数组
    @param Filter: 卷积核
    @param Mode: 边缘处理方式（Symmetry,AdjustWeight）
    @return: 卷积后的图片
    """
    FilterHeight = Filter.shape[0]
    FilterWidth = Filter.shape[1]
    ConvImgHeight = Img.shape[0]
    ConvImgWidth = Img.shape[1]
    ConvImg = np.zeros((len(Img), len(Img[0])), dtype=np.float)
    if Mode == 'Symmetry':
        Height = Filter.shape[0] // 2
        Width = Filter.shape[1] // 2
        PadImg = np.zeros((len(Img) + 2 * Height, len(Img[0]) + 2 * Width), dtype=np.float)
        PadImg[Height:Height+len(Img), Width:Width+len(Img[0])] = Img
        PadImg[0:Height, :] = PadImg[Height:2*Height, :]
        PadImg[len(Img) + Height:len(Img) + 2 * Height, :] = PadImg[len(Img):len(Img)+Height, :]
        PadImg[:, 0:Width] = PadImg[:, Width:2*Width]
        PadImg[:, len(Img[0]) + Width:len(Img[0]) + 2 * Width] = PadImg[:, len(Img[0]):len(Img[0]) + Width]
        Img = PadImg
        for i in range(ConvImgHeight):
            for j in range(ConvImgWidth):
                ConvImg[i][j] = (Img[i:i + FilterHeight, j:j + FilterWidth] * Filter).sum()

    if Mode == 'AdjustWeight':
        Height = Filter.shape[0] // 2
        Width = Filter.shape[1] // 2
        PadImg = np.zeros((len(Img) + 2 * Height, len(Img[0]) + 2 * Width), dtype=np.float)
        PadImg[Height:Height + len(Img), Width:Width + len(Img[0])] = Img
        Img = PadImg
        IsPointInTheSet0Img = lambda x, y:(x>=FilterHeight//2 and x<len(Img)-3*(FilterHeight//2)) and (y>=FilterWidth//2 and y<len(Img[0])-3*(FilterWidth//2))
        IsPointOutsideTheSet0Img = lambda x, y:(x<FilterHeight//2 or x>=len(Img)-FilterHeight//2) or (y<FilterWidth//2 or y>=len(Img[0])-FilterWidth//2)
        for i in range(ConvImgHeight):
            for j in range(ConvImgWidth):
                # print('*'*20)
                # print(Img)
                # print('卷积核坐上角坐标 i,j：',i,j,IsPointInTheSet0Img(i, j))
                if IsPointInTheSet0Img(i, j):
                    ConvImg[i][j] = (Img[i:i + FilterHeight, j:j + FilterWidth] * Filter).sum()
                else:
                    y, x = i+FilterHeight//2, j+FilterWidth//2
                    # print('卷积中心坐标 x,y：',x,y)
                    AdjancetX = [x+ix for ix in range(-FilterHeight//2+1,FilterHeight//2+1)]*FilterWidth
                    AdjancetY_ = [y+iy for iy in range(-FilterWidth//2+1,FilterWidth//2+1)]
                    AdjancetY = []
                    for k in range(FilterWidth):
                        for z in range(FilterHeight):
                            AdjancetY.append(AdjancetY_[k])
                    OutsidesPointLogical = [not(IsPointOutsideTheSet0Img(a, b)) for a,b in zip(AdjancetX, AdjancetY)]
                    FilterSum = 0
                    # print(AdjancetX)
                    # print(AdjancetY)
                    # print(OutsidesPointLogical)
                    for Logical,AX,AY in zip(OutsidesPointLogical,AdjancetX,AdjancetY):
                        # print('Logical,AX,AY:',Logical,AX-x+FilterHeight//2,AY-y+FilterWidth//2)
                        if Logical:
                            FilterSum += Filter[AX-x+FilterHeight//2][AY-y+FilterWidth//2]
                    # print('卷积核：', Filter)
                    # print('卷积区域:', Img[i:i + FilterHeight, j:j + FilterWidth])
                    ConvSum = (Img[i:i + FilterHeight, j:j + FilterWidth] * Filter).sum()
                    ConvImg[i][j] = ConvSum/FilterSum if FilterSum!=0 else ConvSum
                    # print('结果：',ConvSum/FilterSum if FilterSum!=0 else ConvSum)
    return ConvImg
def DownSampling(Img):
    """
    对单通道图片进行步长为2的下采样（隔点采样）
    @param Img: 图片
    @return: 采样后的图片
    """
    Height, Width = len(Img), len(Img[0])
    NewImg = []
    for x in range(0,Height,2):
        Row = []
        for y in range(0,Width,2):
            Row.append(Img[x][y])
        NewImg.append(Row)
    return np.array(NewImg)
def RGB2HSI(Img):
    """
    rgb转hsi
    @param Img: rgb图片
    @return: hsi图片
    """
    rows = int(Img.shape[0])
    cols = int(Img.shape[1])
    r, g, b = [Img[:, :, i] for i in range(3)]
    # 归一化到[0,1]
    b = b / 255.0
    g = g / 255.0
    r = r / 255.0
    RawHSI = Img.copy()
    H, S, I = [Img[:, :, i] for i in range(3)]
    for i in range(rows):
        for j in range(cols):
            num = 0.5 * ((r[i, j]-g[i, j])+(r[i, j]-b[i, j]))
            den = np.sqrt((r[i, j]-g[i, j])**2+(r[i, j]-b[i, j])*(g[i, j]-b[i, j]))
            theta = np.float(np.arccos(num/den))

            if den == 0:
                    H = 0
            elif b[i, j] <= g[i, j]:
                H = theta
            else:
                H = 2*3.14169265 - theta

            min_RGB = min(min(b[i, j], g[i, j]), r[i, j])
            sum = b[i, j]+g[i, j]+r[i, j]
            if sum == 0:
                S = 0
            else:
                S = 1 - 3*min_RGB/sum

            H = H/(2*3.14159265)
            I = sum/3.0
            # 输出HSI图像，扩充到255以方便显示，一般H分量在[0,2pi]之间，S和I在[0,1]之间
            RawHSI[i, j, 0] = H*255
            RawHSI[i, j, 1] = S*255
            RawHSI[i, j, 2] = I*255
    return RawHSI


if __name__ == '__main__':
    import cv2
    import imageio
    Img = imageio.imread('11.jpg')
    RawHSI = RGB2HSI(Img)

    cv2.imshow('Img', Img)
    cv2.imshow('RawHSI', RawHSI)

    key = cv2.waitKey(0) & 0xFF
    if key == ord('q'):
        cv2.destroyAllWindows()


