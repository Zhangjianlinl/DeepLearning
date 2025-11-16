import os
from pathlib import Path
import gdown
import tensorflow as tf
from retinaface.commons.logger import Logger

logger = Logger(module="retinaface/model/retinaface_model.py")

# pylint: disable=too-many-statements, no-name-in-module

# configurations

# 逐词解释：
# tf_version        -> 变量名：用于保存 TensorFlow 的主版本号（major version）
# =                 -> 赋值运算符：将右侧计算结果赋给左侧变量
# int(              -> 将括号内的字符串转换为整数类型
# tf.__version__    -> TensorFlow 暴露的版本字符串，例如 "2.13.0"、"1.15.5"
# .split(".",       -> 对版本字符串按点号进行分割
#         maxsplit=1)-> 最多只分割一次，结果形如 ["2", "13.0"] 或 ["1", "15.5"]
# )[0]              -> 取分割结果列表的第 0 个元素，即主版本号的字符串部分（如 "2" 或 "1"）
# )                 -> 结束 int( 的括号，将主版本号字符串转换为整数（如 2 或 1）
tf_version = int(tf.__version__.split(".", maxsplit=1)[0])

# 第 24 行：判断 TensorFlow 主版本号是否为 1
# if            -> 条件判断关键字：如果满足条件则执行缩进代码块
# tf_version    -> 前面定义的变量：保存 TensorFlow 主版本号（1 或 2）
# ==            -> 相等比较运算符：判断左右两侧值是否相等
# 1:            -> 整数 1，冒号表示条件判断结束，后续为执行代码块
if tf_version == 1:
    # 第 25 行：从独立的 Keras 包导入 Model 类
    # from          -> 导入关键字：指定从哪个模块导入
    # keras.models  -> Keras 的模型模块（TensorFlow 1.x 时代 Keras 是独立包）
    # import        -> 导入关键字：指定要导入的对象
    # Model         -> 神经网络模型的基类，用于构建和管理模型
    from keras.models import Model
    # 第 26-37 行：从独立的 Keras 包导入各种神经网络层
    # from keras.layers -> 从 Keras 的层模块导入
    # import (...)      -> 导入多个类，用括号包裹实现多行导入
    from keras.layers import (
        # Input              -> 输入层：定义模型的输入张量形状
        Input,
        # BatchNormalization -> 批归一化层：对每个批次数据进行归一化，加速训练和提高稳定性
        BatchNormalization,
        # ZeroPadding2D      -> 零填充层：在图像周围填充零值，保持卷积后的尺寸
        ZeroPadding2D,
        # Conv2D             -> 二维卷积层：用于提取图像特征的核心层
        Conv2D,
        # ReLU               -> ReLU 激活函数层：修正线性单元，f(x)=max(0,x)
        ReLU,
        # MaxPool2D          -> 最大池化层：对特征图进行下采样，减小尺寸
        MaxPool2D,
        # Add                -> 加法层：将多个张量逐元素相加（用于残差连接）
        Add,
        # UpSampling2D       -> 上采样层：放大特征图尺寸（用于特征金字塔）
        UpSampling2D,
        # concatenate        -> 拼接函数：沿指定轴将多个张量拼接在一起
        concatenate,
        # Softmax            -> Softmax 激活层：将输出转换为概率分布
        Softmax,
    )

# 如果 TensorFlow 版本不是 1（即版本 2 或更高）
# else:         -> 条件判断的否则分支：当 if 条件不满足时执行
else:
    # 第 40 行：从 TensorFlow 集成的 Keras 模块导入 Model 类
    # from tensorflow.keras.models -> TensorFlow 2.x 中 Keras 已集成为子模块
    # import Model                 -> 导入模型基类（功能与上面相同，只是路径不同）
    from tensorflow.keras.models import Model
    # 第 41-52 行：从 TensorFlow 集成的 Keras 导入各种神经网络层
    # from tensorflow.keras.layers -> 从 TensorFlow 的 Keras 层模块导入
    # import (...)                 -> 导入的层与上面完全相同，只是导入路径不同
    from tensorflow.keras.layers import (
        # Input              -> 输入层：定义模型的输入张量形状
        Input,
        # BatchNormalization -> 批归一化层：对每个批次数据进行归一化
        BatchNormalization,
        # ZeroPadding2D      -> 零填充层：在图像周围填充零值
        ZeroPadding2D,
        # Conv2D             -> 二维卷积层：用于提取图像特征
        Conv2D,
        # ReLU               -> ReLU 激活函数层：修正线性单元
        ReLU,
        # MaxPool2D          -> 最大池化层：对特征图进行下采样
        MaxPool2D,
        # Add                -> 加法层：将多个张量逐元素相加
        Add,
        # UpSampling2D       -> 上采样层：放大特征图尺寸
        UpSampling2D,
        # concatenate        -> 拼接函数：沿指定轴将多个张量拼接
        concatenate,
        # Softmax            -> Softmax 激活层：将输出转换为概率分布
        Softmax,
    )


# 定义加载预训练权重的函数
# def             -> 定义函数的关键字
# load_weights    -> 函数名：加载权重
# (model: Model): -> 参数列表，model 是参数名，Model 是类型注解（表示这是一个 Keras 模型对象）
def load_weights(model: Model):
    """
    Loading pre-trained weights for the RetinaFace model
    Args:
        model (Model): retinaface model structure with randon weights
    Returns:
        model (Model): retinaface model with its structure and pre-trained weights

    """
    # 获取 DeepFace 的主目录路径
    # home            -> 变量名：保存用户主目录路径
    # =               -> 赋值运算符
    # str(            -> 将结果转换为字符串类型
    # os.getenv(      -> 获取环境变量的函数
    # "DEEPFACE_HOME" -> 环境变量名：DeepFace 的自定义主目录
    # , default=      -> 如果环境变量不存在，使用默认值
    # str(Path.home())-> 获取系统用户主目录并转换为字符串（如 C:\Users\Administrator）
    # ))              -> 两层括号结束
    home = str(os.getenv("DEEPFACE_HOME", default=str(Path.home())))

    # 拼接权重文件的完整路径
    # exact_file -> 变量名：精确的文件路径
    # =          -> 赋值运算符
    # home +     -> 主目录路径字符串
    # "/.deepface/weights/retinaface.h5" -> 相对路径，权重文件存储位置
    exact_file = home + "/.deepface/weights/retinaface.h5"
    # 定义权重文件的下载 URL
    # url -> 变量名：统一资源定位符
    # =   -> 赋值运算符
    # "https://..." -> GitHub 发布页面的权重文件下载地址
    url = "https://github.com/serengil/deepface_models/releases/download/v1.0/retinaface.h5"

    # -----------------------------

    # 检查 .deepface 目录是否存在
    # if                -> 条件判断关键字
    # not               -> 逻辑非运算符：取反
    # os.path.exists(   -> 检查路径是否存在的函数
    # home + "/.deepface" -> 拼接目录路径
    # ):                -> 条件结束，如果目录不存在则执行下面代码
    if not os.path.exists(home + "/.deepface"):
        # 创建 .deepface 目录
        # os.mkdir(              -> 创建单层目录的函数
        # home + "/.deepface")   -> 要创建的目录路径
        os.mkdir(home + "/.deepface")
        # 记录目录创建信息到日志
        # logger.info(           -> 日志记录器的信息级别方法
        # f"Directory {home}/.deepface created" -> f-string 格式化字符串，嵌入 home 变量值
        # )                      -> 函数调用结束
        logger.info(f"Directory {home}/.deepface created")

    # 检查 weights 子目录是否存在
    # if not os.path.exists( -> 判断路径是否不存在
    # home + "/.deepface/weights" -> weights 子目录的完整路径
    # ):                     -> 条件结束
    if not os.path.exists(home + "/.deepface/weights"):
        # 创建 weights 子目录
        # os.mkdir(home + "/.deepface/weights") -> 创建 weights 目录
        os.mkdir(home + "/.deepface/weights")
        # 记录 weights 目录创建信息
        # logger.info(f"Directory {home}/.deepface/weights created") -> 输出日志
        logger.info(f"Directory {home}/.deepface/weights created")

    # -----------------------------

    # 检查权重文件是否不存在
    # if                    -> 条件判断
    # os.path.isfile(       -> 检查路径是否为文件的函数
    # exact_file)           -> 权重文件的完整路径
    # is not True:          -> 判断结果是否不为 True（即文件不存在或不是文件）
    if os.path.isfile(exact_file) is not True:
        # 记录即将下载权重文件的信息
        # logger.info(          -> 输出信息级别日志
        # f"retinaface.h5 will be downloaded from the url {url}") -> 提示下载来源
        logger.info(f"retinaface.h5 will be downloaded from the url {url}")
        # 使用 gdown 下载权重文件
        # gdown.download(       -> gdown 库的下载函数（专门用于下载 Google Drive 和 GitHub 文件）
        # url,                  -> 下载链接
        # exact_file,           -> 保存到的本地文件路径
        # quiet=False)          -> 不使用静默模式，显示下载进度
        gdown.download(url, exact_file, quiet=False)

    # -----------------------------

    # gdown should download the pretrained weights here.
    # If it does not still exist, then throw an exception.
    # 再次检查文件是否存在，如果下载失败则抛出异常
    # if os.path.isfile(exact_file) is not True: -> 文件仍然不存在
    if os.path.isfile(exact_file) is not True:
        # 抛出值错误异常，提示手动下载
        # raise          -> 抛出异常的关键字
        # ValueError(    -> 值错误异常类型
        # "..."          -> 错误信息字符串，使用 + 拼接多行
        # + url          -> 嵌入下载 URL
        # + " and copy it to the ", -> 提示复制到
        # exact_file,    -> 目标文件路径
        # "manually.",   -> 手动操作提示
        # )              -> 异常构造结束
        raise ValueError(
            "Pre-trained weight could not be loaded!"
            + " You might try to download the pre-trained weights from the url "
            + url
            + " and copy it to the ",
            exact_file,
            "manually.",
        )

    # 加载权重文件到模型
    # model.load_weights( -> 调用模型对象的加载权重方法
    # exact_file)         -> 传入权重文件路径
    model.load_weights(exact_file)

    # 返回加载了权重的模型
    # return -> 返回关键字
    # model  -> 返回已加载权重的模型对象
    return model


def build_model() -> Model:
    """
    Build RetinaFace model
    
    这个函数定义了 RetinaFace 人脸检测网络的完整结构
    
    网络架构：
    1. 输入层：接收任意尺寸的 RGB 图片
    2. ResNet-50 主干网络：提取图像特征（共4个stage）
    3. FPN（特征金字塔网络）：融合多尺度特征
    4. 检测头：输出人脸框、关键点、置信度
    
    编写方式：使用 Keras 函数式 API
    语法：output = Layer(参数)(input)
    
    Returns:
        Model: 未加载权重的 RetinaFace 模型（需要调用 load_weights 加载权重）
    """
    
    # ==================== 第一部分：输入层 ====================
    # 定义输入层：接收任意尺寸的 RGB 图片
    # dtype=tf.float32  -> 数据类型为 32 位浮点数
    # shape=(None, None, 3) -> (高度, 宽度, 通道数)
    #   - None: 可以接受任意尺寸的图片（动态尺寸）
    #   - 3: RGB 三通道彩色图片
    # name="data" -> 层的名称，方便调试和可视化
    data = Input(dtype=tf.float32, shape=(None, None, 3), name="data")

    # ==================== 第二部分：数据预处理 ====================
    # 批归一化（Batch Normalization）：对输入数据进行标准化
    # epsilon -> 防止除零的小常数
    # trainable=False -> 推理时不更新该层的参数（使用训练时统计的均值和方差）
    # (data) -> 将 data 作为输入，形成连接
    bn_data = BatchNormalization(epsilon=1.9999999494757503e-05, name="bn_data", trainable=False)(
        data
    )

    # ==================== 第三部分：第一个卷积块（Conv0） ====================
    # 零填充：在图像周围填充 3 圈零值像素
    # 作用：保证卷积后的特征图不会太小
    # padding=tuple([3, 3]) -> 上下左右各填充 3 个像素
    conv0_pad = ZeroPadding2D(padding=tuple([3, 3]))(bn_data)

    # 第一个卷积层：大卷积核提取初始特征
    # filters=64 -> 使用 64 个卷积核（输出 64 个通道）
    # kernel_size=(7, 7) -> 每个卷积核的尺寸是 7×7
    # strides=[2, 2] -> 步长为 2，图像尺寸缩小为原来的 1/2
    # padding="VALID" -> 不自动填充（因为前面手动填充了）
    # use_bias=False -> 不使用偏置项（因为后面有 BN 层会处理偏移）
    # (conv0_pad) -> 将填充后的图像作为输入
    conv0 = Conv2D(
        filters=64,
        kernel_size=(7, 7),
        name="conv0",
        strides=[2, 2],
        padding="VALID",
        use_bias=False,
    )(conv0_pad)

    # 批归一化：标准化卷积层的输出，加速训练和提高稳定性
    bn0 = BatchNormalization(epsilon=1.9999999494757503e-05, name="bn0", trainable=False)(conv0)

    # ReLU 激活函数：增加非线性，f(x) = max(0, x)
    # 作用：将负值变为 0，保留正值
    relu0 = ReLU(name="relu0")(bn0)

    # 零填充：为池化层准备
    pooling0_pad = ZeroPadding2D(padding=tuple([1, 1]))(relu0)

    # 最大池化层：降低特征图的空间分辨率
    # (3, 3) -> 池化窗口大小 3×3
    # (2, 2) -> 步长为 2，图像尺寸再次缩小为原来的 1/2
    # 作用：减少计算量，增加感受野
    # 现在图像尺寸变为原始的 1/4（经过两次步长为2的操作）
    pooling0 = MaxPool2D((3, 3), (2, 2), padding="valid", name="pooling0")(pooling0_pad)

    # ==================== 第四部分：ResNet 残差块 ====================
    # Stage 1, Unit 1 - 第一个残差单元
    # 
    # 残差块的结构：
    #   输入 x
    #   ├──────────────────────┐  (捷径路径 shortcut)
    #   │                      │
    #   ↓                      │
    #   [主路径：3个卷积]      │
    #   ↓                      │
    #   相加 ←─────────────────┘
    #   ↓
    #   输出
    #
    # 残差学习的优势：
    # 1. 缓解梯度消失问题，可以训练更深的网络
    # 2. 学习残差（变化量）而不是完整映射，更容易优化
    # 3. 可以通过捷径路径直接传递梯度
    
    # 批归一化
    stage1_unit1_bn1 = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="stage1_unit1_bn1", trainable=False
    )(pooling0)

    # ReLU 激活（这个激活的输出会分成两路）
    stage1_unit1_relu1 = ReLU(name="stage1_unit1_relu1")(stage1_unit1_bn1)

    # 【主路径 - 分支1】第一个卷积：1×1 卷积降维
    # 作用：减少通道数，降低计算量（瓶颈结构 bottleneck）
    stage1_unit1_conv1 = Conv2D(
        filters=64,
        kernel_size=(1, 1),
        name="stage1_unit1_conv1",
        strides=[1, 1],
        padding="VALID",
        use_bias=False,
    )(stage1_unit1_relu1)

    # 【捷径路径 - 分支2】shortcut connection
    # 用 1×1 卷积调整通道数，使其能与主路径相加
    # 注意：输入是 64 通道，输出是 256 通道（需要匹配主路径最后的输出）
    stage1_unit1_sc = Conv2D(
        filters=256,
        kernel_size=(1, 1),
        name="stage1_unit1_sc",
        strides=[1, 1],
        padding="VALID",
        use_bias=False,
    )(stage1_unit1_relu1)

    # 【主路径继续】批归一化
    stage1_unit1_bn2 = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="stage1_unit1_bn2", trainable=False
    )(stage1_unit1_conv1)

    # ReLU 激活
    stage1_unit1_relu2 = ReLU(name="stage1_unit1_relu2")(stage1_unit1_bn2)

    # 零填充：为 3×3 卷积准备
    stage1_unit1_conv2_pad = ZeroPadding2D(padding=tuple([1, 1]))(stage1_unit1_relu2)

    # 【主路径】第二个卷积：3×3 卷积提取特征
    # 作用：在降维后提取空间特征（这是瓶颈结构的核心）
    stage1_unit1_conv2 = Conv2D(
        filters=64,
        kernel_size=(3, 3),
        name="stage1_unit1_conv2",
        strides=[1, 1],
        padding="VALID",
        use_bias=False,
    )(stage1_unit1_conv2_pad)

    # 批归一化
    stage1_unit1_bn3 = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="stage1_unit1_bn3", trainable=False
    )(stage1_unit1_conv2)

    # ReLU 激活
    stage1_unit1_relu3 = ReLU(name="stage1_unit1_relu3")(stage1_unit1_bn3)

    # 【主路径】第三个卷积：1×1 卷积升维
    # 作用：恢复通道数到 256（与捷径路径匹配）
    # 整个瓶颈结构：256→64→64→256（先降维，提取特征，再升维）
    stage1_unit1_conv3 = Conv2D(
        filters=256,
        kernel_size=(1, 1),
        name="stage1_unit1_conv3",
        strides=[1, 1],
        padding="VALID",
        use_bias=False,
    )(stage1_unit1_relu3)

    # 【残差连接】将主路径和捷径路径相加
    # Add() 是一个层，[list] 中是要相加的两个张量
    # stage1_unit1_conv3: 主路径的输出（256 通道）
    # stage1_unit1_sc: 捷径路径的输出（256 通道）
    # 结果：逐元素相加，形状不变
    # 这就是 ResNet 的核心：y = F(x) + x
    plus0_v1 = Add()([stage1_unit1_conv3, stage1_unit1_sc])

    stage1_unit2_bn1 = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="stage1_unit2_bn1", trainable=False
    )(plus0_v1)

    stage1_unit2_relu1 = ReLU(name="stage1_unit2_relu1")(stage1_unit2_bn1)

    stage1_unit2_conv1 = Conv2D(
        filters=64,
        kernel_size=(1, 1),
        name="stage1_unit2_conv1",
        strides=[1, 1],
        padding="VALID",
        use_bias=False,
    )(stage1_unit2_relu1)

    stage1_unit2_bn2 = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="stage1_unit2_bn2", trainable=False
    )(stage1_unit2_conv1)

    stage1_unit2_relu2 = ReLU(name="stage1_unit2_relu2")(stage1_unit2_bn2)

    stage1_unit2_conv2_pad = ZeroPadding2D(padding=tuple([1, 1]))(stage1_unit2_relu2)

    stage1_unit2_conv2 = Conv2D(
        filters=64,
        kernel_size=(3, 3),
        name="stage1_unit2_conv2",
        strides=[1, 1],
        padding="VALID",
        use_bias=False,
    )(stage1_unit2_conv2_pad)

    stage1_unit2_bn3 = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="stage1_unit2_bn3", trainable=False
    )(stage1_unit2_conv2)

    stage1_unit2_relu3 = ReLU(name="stage1_unit2_relu3")(stage1_unit2_bn3)

    stage1_unit2_conv3 = Conv2D(
        filters=256,
        kernel_size=(1, 1),
        name="stage1_unit2_conv3",
        strides=[1, 1],
        padding="VALID",
        use_bias=False,
    )(stage1_unit2_relu3)

    plus1_v2 = Add()([stage1_unit2_conv3, plus0_v1])

    stage1_unit3_bn1 = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="stage1_unit3_bn1", trainable=False
    )(plus1_v2)

    stage1_unit3_relu1 = ReLU(name="stage1_unit3_relu1")(stage1_unit3_bn1)

    stage1_unit3_conv1 = Conv2D(
        filters=64,
        kernel_size=(1, 1),
        name="stage1_unit3_conv1",
        strides=[1, 1],
        padding="VALID",
        use_bias=False,
    )(stage1_unit3_relu1)

    stage1_unit3_bn2 = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="stage1_unit3_bn2", trainable=False
    )(stage1_unit3_conv1)

    stage1_unit3_relu2 = ReLU(name="stage1_unit3_relu2")(stage1_unit3_bn2)

    stage1_unit3_conv2_pad = ZeroPadding2D(padding=tuple([1, 1]))(stage1_unit3_relu2)

    stage1_unit3_conv2 = Conv2D(
        filters=64,
        kernel_size=(3, 3),
        name="stage1_unit3_conv2",
        strides=[1, 1],
        padding="VALID",
        use_bias=False,
    )(stage1_unit3_conv2_pad)

    stage1_unit3_bn3 = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="stage1_unit3_bn3", trainable=False
    )(stage1_unit3_conv2)

    stage1_unit3_relu3 = ReLU(name="stage1_unit3_relu3")(stage1_unit3_bn3)

    stage1_unit3_conv3 = Conv2D(
        filters=256,
        kernel_size=(1, 1),
        name="stage1_unit3_conv3",
        strides=[1, 1],
        padding="VALID",
        use_bias=False,
    )(stage1_unit3_relu3)

    plus2 = Add()([stage1_unit3_conv3, plus1_v2])

    stage2_unit1_bn1 = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="stage2_unit1_bn1", trainable=False
    )(plus2)

    stage2_unit1_relu1 = ReLU(name="stage2_unit1_relu1")(stage2_unit1_bn1)

    stage2_unit1_conv1 = Conv2D(
        filters=128,
        kernel_size=(1, 1),
        name="stage2_unit1_conv1",
        strides=[1, 1],
        padding="VALID",
        use_bias=False,
    )(stage2_unit1_relu1)

    stage2_unit1_sc = Conv2D(
        filters=512,
        kernel_size=(1, 1),
        name="stage2_unit1_sc",
        strides=[2, 2],
        padding="VALID",
        use_bias=False,
    )(stage2_unit1_relu1)

    stage2_unit1_bn2 = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="stage2_unit1_bn2", trainable=False
    )(stage2_unit1_conv1)

    stage2_unit1_relu2 = ReLU(name="stage2_unit1_relu2")(stage2_unit1_bn2)

    stage2_unit1_conv2_pad = ZeroPadding2D(padding=tuple([1, 1]))(stage2_unit1_relu2)

    stage2_unit1_conv2 = Conv2D(
        filters=128,
        kernel_size=(3, 3),
        name="stage2_unit1_conv2",
        strides=[2, 2],
        padding="VALID",
        use_bias=False,
    )(stage2_unit1_conv2_pad)

    stage2_unit1_bn3 = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="stage2_unit1_bn3", trainable=False
    )(stage2_unit1_conv2)

    stage2_unit1_relu3 = ReLU(name="stage2_unit1_relu3")(stage2_unit1_bn3)

    stage2_unit1_conv3 = Conv2D(
        filters=512,
        kernel_size=(1, 1),
        name="stage2_unit1_conv3",
        strides=[1, 1],
        padding="VALID",
        use_bias=False,
    )(stage2_unit1_relu3)

    plus3 = Add()([stage2_unit1_conv3, stage2_unit1_sc])

    stage2_unit2_bn1 = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="stage2_unit2_bn1", trainable=False
    )(plus3)

    stage2_unit2_relu1 = ReLU(name="stage2_unit2_relu1")(stage2_unit2_bn1)

    stage2_unit2_conv1 = Conv2D(
        filters=128,
        kernel_size=(1, 1),
        name="stage2_unit2_conv1",
        strides=[1, 1],
        padding="VALID",
        use_bias=False,
    )(stage2_unit2_relu1)

    stage2_unit2_bn2 = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="stage2_unit2_bn2", trainable=False
    )(stage2_unit2_conv1)

    stage2_unit2_relu2 = ReLU(name="stage2_unit2_relu2")(stage2_unit2_bn2)

    stage2_unit2_conv2_pad = ZeroPadding2D(padding=tuple([1, 1]))(stage2_unit2_relu2)

    stage2_unit2_conv2 = Conv2D(
        filters=128,
        kernel_size=(3, 3),
        name="stage2_unit2_conv2",
        strides=[1, 1],
        padding="VALID",
        use_bias=False,
    )(stage2_unit2_conv2_pad)

    stage2_unit2_bn3 = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="stage2_unit2_bn3", trainable=False
    )(stage2_unit2_conv2)

    stage2_unit2_relu3 = ReLU(name="stage2_unit2_relu3")(stage2_unit2_bn3)

    stage2_unit2_conv3 = Conv2D(
        filters=512,
        kernel_size=(1, 1),
        name="stage2_unit2_conv3",
        strides=[1, 1],
        padding="VALID",
        use_bias=False,
    )(stage2_unit2_relu3)

    plus4 = Add()([stage2_unit2_conv3, plus3])

    stage2_unit3_bn1 = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="stage2_unit3_bn1", trainable=False
    )(plus4)

    stage2_unit3_relu1 = ReLU(name="stage2_unit3_relu1")(stage2_unit3_bn1)

    stage2_unit3_conv1 = Conv2D(
        filters=128,
        kernel_size=(1, 1),
        name="stage2_unit3_conv1",
        strides=[1, 1],
        padding="VALID",
        use_bias=False,
    )(stage2_unit3_relu1)

    stage2_unit3_bn2 = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="stage2_unit3_bn2", trainable=False
    )(stage2_unit3_conv1)

    stage2_unit3_relu2 = ReLU(name="stage2_unit3_relu2")(stage2_unit3_bn2)

    stage2_unit3_conv2_pad = ZeroPadding2D(padding=tuple([1, 1]))(stage2_unit3_relu2)

    stage2_unit3_conv2 = Conv2D(
        filters=128,
        kernel_size=(3, 3),
        name="stage2_unit3_conv2",
        strides=[1, 1],
        padding="VALID",
        use_bias=False,
    )(stage2_unit3_conv2_pad)

    stage2_unit3_bn3 = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="stage2_unit3_bn3", trainable=False
    )(stage2_unit3_conv2)

    stage2_unit3_relu3 = ReLU(name="stage2_unit3_relu3")(stage2_unit3_bn3)

    stage2_unit3_conv3 = Conv2D(
        filters=512,
        kernel_size=(1, 1),
        name="stage2_unit3_conv3",
        strides=[1, 1],
        padding="VALID",
        use_bias=False,
    )(stage2_unit3_relu3)

    plus5 = Add()([stage2_unit3_conv3, plus4])

    stage2_unit4_bn1 = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="stage2_unit4_bn1", trainable=False
    )(plus5)

    stage2_unit4_relu1 = ReLU(name="stage2_unit4_relu1")(stage2_unit4_bn1)

    stage2_unit4_conv1 = Conv2D(
        filters=128,
        kernel_size=(1, 1),
        name="stage2_unit4_conv1",
        strides=[1, 1],
        padding="VALID",
        use_bias=False,
    )(stage2_unit4_relu1)

    stage2_unit4_bn2 = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="stage2_unit4_bn2", trainable=False
    )(stage2_unit4_conv1)

    stage2_unit4_relu2 = ReLU(name="stage2_unit4_relu2")(stage2_unit4_bn2)

    stage2_unit4_conv2_pad = ZeroPadding2D(padding=tuple([1, 1]))(stage2_unit4_relu2)

    stage2_unit4_conv2 = Conv2D(
        filters=128,
        kernel_size=(3, 3),
        name="stage2_unit4_conv2",
        strides=[1, 1],
        padding="VALID",
        use_bias=False,
    )(stage2_unit4_conv2_pad)

    stage2_unit4_bn3 = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="stage2_unit4_bn3", trainable=False
    )(stage2_unit4_conv2)

    stage2_unit4_relu3 = ReLU(name="stage2_unit4_relu3")(stage2_unit4_bn3)

    stage2_unit4_conv3 = Conv2D(
        filters=512,
        kernel_size=(1, 1),
        name="stage2_unit4_conv3",
        strides=[1, 1],
        padding="VALID",
        use_bias=False,
    )(stage2_unit4_relu3)

    plus6 = Add()([stage2_unit4_conv3, plus5])

    stage3_unit1_bn1 = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="stage3_unit1_bn1", trainable=False
    )(plus6)

    stage3_unit1_relu1 = ReLU(name="stage3_unit1_relu1")(stage3_unit1_bn1)

    stage3_unit1_conv1 = Conv2D(
        filters=256,
        kernel_size=(1, 1),
        name="stage3_unit1_conv1",
        strides=[1, 1],
        padding="VALID",
        use_bias=False,
    )(stage3_unit1_relu1)

    stage3_unit1_sc = Conv2D(
        filters=1024,
        kernel_size=(1, 1),
        name="stage3_unit1_sc",
        strides=[2, 2],
        padding="VALID",
        use_bias=False,
    )(stage3_unit1_relu1)

    stage3_unit1_bn2 = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="stage3_unit1_bn2", trainable=False
    )(stage3_unit1_conv1)

    stage3_unit1_relu2 = ReLU(name="stage3_unit1_relu2")(stage3_unit1_bn2)

    stage3_unit1_conv2_pad = ZeroPadding2D(padding=tuple([1, 1]))(stage3_unit1_relu2)

    stage3_unit1_conv2 = Conv2D(
        filters=256,
        kernel_size=(3, 3),
        name="stage3_unit1_conv2",
        strides=[2, 2],
        padding="VALID",
        use_bias=False,
    )(stage3_unit1_conv2_pad)

    ssh_m1_red_conv = Conv2D(
        filters=256,
        kernel_size=(1, 1),
        name="ssh_m1_red_conv",
        strides=[1, 1],
        padding="VALID",
        use_bias=True,
    )(stage3_unit1_relu2)

    stage3_unit1_bn3 = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="stage3_unit1_bn3", trainable=False
    )(stage3_unit1_conv2)

    ssh_m1_red_conv_bn = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="ssh_m1_red_conv_bn", trainable=False
    )(ssh_m1_red_conv)

    stage3_unit1_relu3 = ReLU(name="stage3_unit1_relu3")(stage3_unit1_bn3)

    ssh_m1_red_conv_relu = ReLU(name="ssh_m1_red_conv_relu")(ssh_m1_red_conv_bn)

    stage3_unit1_conv3 = Conv2D(
        filters=1024,
        kernel_size=(1, 1),
        name="stage3_unit1_conv3",
        strides=[1, 1],
        padding="VALID",
        use_bias=False,
    )(stage3_unit1_relu3)

    plus7 = Add()([stage3_unit1_conv3, stage3_unit1_sc])

    stage3_unit2_bn1 = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="stage3_unit2_bn1", trainable=False
    )(plus7)

    stage3_unit2_relu1 = ReLU(name="stage3_unit2_relu1")(stage3_unit2_bn1)

    stage3_unit2_conv1 = Conv2D(
        filters=256,
        kernel_size=(1, 1),
        name="stage3_unit2_conv1",
        strides=[1, 1],
        padding="VALID",
        use_bias=False,
    )(stage3_unit2_relu1)

    stage3_unit2_bn2 = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="stage3_unit2_bn2", trainable=False
    )(stage3_unit2_conv1)

    stage3_unit2_relu2 = ReLU(name="stage3_unit2_relu2")(stage3_unit2_bn2)

    stage3_unit2_conv2_pad = ZeroPadding2D(padding=tuple([1, 1]))(stage3_unit2_relu2)

    stage3_unit2_conv2 = Conv2D(
        filters=256,
        kernel_size=(3, 3),
        name="stage3_unit2_conv2",
        strides=[1, 1],
        padding="VALID",
        use_bias=False,
    )(stage3_unit2_conv2_pad)

    stage3_unit2_bn3 = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="stage3_unit2_bn3", trainable=False
    )(stage3_unit2_conv2)

    stage3_unit2_relu3 = ReLU(name="stage3_unit2_relu3")(stage3_unit2_bn3)

    stage3_unit2_conv3 = Conv2D(
        filters=1024,
        kernel_size=(1, 1),
        name="stage3_unit2_conv3",
        strides=[1, 1],
        padding="VALID",
        use_bias=False,
    )(stage3_unit2_relu3)

    plus8 = Add()([stage3_unit2_conv3, plus7])

    stage3_unit3_bn1 = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="stage3_unit3_bn1", trainable=False
    )(plus8)

    stage3_unit3_relu1 = ReLU(name="stage3_unit3_relu1")(stage3_unit3_bn1)

    stage3_unit3_conv1 = Conv2D(
        filters=256,
        kernel_size=(1, 1),
        name="stage3_unit3_conv1",
        strides=[1, 1],
        padding="VALID",
        use_bias=False,
    )(stage3_unit3_relu1)

    stage3_unit3_bn2 = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="stage3_unit3_bn2", trainable=False
    )(stage3_unit3_conv1)

    stage3_unit3_relu2 = ReLU(name="stage3_unit3_relu2")(stage3_unit3_bn2)

    stage3_unit3_conv2_pad = ZeroPadding2D(padding=tuple([1, 1]))(stage3_unit3_relu2)

    stage3_unit3_conv2 = Conv2D(
        filters=256,
        kernel_size=(3, 3),
        name="stage3_unit3_conv2",
        strides=[1, 1],
        padding="VALID",
        use_bias=False,
    )(stage3_unit3_conv2_pad)

    stage3_unit3_bn3 = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="stage3_unit3_bn3", trainable=False
    )(stage3_unit3_conv2)

    stage3_unit3_relu3 = ReLU(name="stage3_unit3_relu3")(stage3_unit3_bn3)

    stage3_unit3_conv3 = Conv2D(
        filters=1024,
        kernel_size=(1, 1),
        name="stage3_unit3_conv3",
        strides=[1, 1],
        padding="VALID",
        use_bias=False,
    )(stage3_unit3_relu3)

    plus9 = Add()([stage3_unit3_conv3, plus8])

    stage3_unit4_bn1 = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="stage3_unit4_bn1", trainable=False
    )(plus9)

    stage3_unit4_relu1 = ReLU(name="stage3_unit4_relu1")(stage3_unit4_bn1)

    stage3_unit4_conv1 = Conv2D(
        filters=256,
        kernel_size=(1, 1),
        name="stage3_unit4_conv1",
        strides=[1, 1],
        padding="VALID",
        use_bias=False,
    )(stage3_unit4_relu1)

    stage3_unit4_bn2 = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="stage3_unit4_bn2", trainable=False
    )(stage3_unit4_conv1)

    stage3_unit4_relu2 = ReLU(name="stage3_unit4_relu2")(stage3_unit4_bn2)

    stage3_unit4_conv2_pad = ZeroPadding2D(padding=tuple([1, 1]))(stage3_unit4_relu2)

    stage3_unit4_conv2 = Conv2D(
        filters=256,
        kernel_size=(3, 3),
        name="stage3_unit4_conv2",
        strides=[1, 1],
        padding="VALID",
        use_bias=False,
    )(stage3_unit4_conv2_pad)

    stage3_unit4_bn3 = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="stage3_unit4_bn3", trainable=False
    )(stage3_unit4_conv2)

    stage3_unit4_relu3 = ReLU(name="stage3_unit4_relu3")(stage3_unit4_bn3)

    stage3_unit4_conv3 = Conv2D(
        filters=1024,
        kernel_size=(1, 1),
        name="stage3_unit4_conv3",
        strides=[1, 1],
        padding="VALID",
        use_bias=False,
    )(stage3_unit4_relu3)

    plus10 = Add()([stage3_unit4_conv3, plus9])

    stage3_unit5_bn1 = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="stage3_unit5_bn1", trainable=False
    )(plus10)

    stage3_unit5_relu1 = ReLU(name="stage3_unit5_relu1")(stage3_unit5_bn1)

    stage3_unit5_conv1 = Conv2D(
        filters=256,
        kernel_size=(1, 1),
        name="stage3_unit5_conv1",
        strides=[1, 1],
        padding="VALID",
        use_bias=False,
    )(stage3_unit5_relu1)

    stage3_unit5_bn2 = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="stage3_unit5_bn2", trainable=False
    )(stage3_unit5_conv1)

    stage3_unit5_relu2 = ReLU(name="stage3_unit5_relu2")(stage3_unit5_bn2)

    stage3_unit5_conv2_pad = ZeroPadding2D(padding=tuple([1, 1]))(stage3_unit5_relu2)

    stage3_unit5_conv2 = Conv2D(
        filters=256,
        kernel_size=(3, 3),
        name="stage3_unit5_conv2",
        strides=[1, 1],
        padding="VALID",
        use_bias=False,
    )(stage3_unit5_conv2_pad)

    stage3_unit5_bn3 = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="stage3_unit5_bn3", trainable=False
    )(stage3_unit5_conv2)

    stage3_unit5_relu3 = ReLU(name="stage3_unit5_relu3")(stage3_unit5_bn3)

    stage3_unit5_conv3 = Conv2D(
        filters=1024,
        kernel_size=(1, 1),
        name="stage3_unit5_conv3",
        strides=[1, 1],
        padding="VALID",
        use_bias=False,
    )(stage3_unit5_relu3)

    plus11 = Add()([stage3_unit5_conv3, plus10])

    stage3_unit6_bn1 = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="stage3_unit6_bn1", trainable=False
    )(plus11)

    stage3_unit6_relu1 = ReLU(name="stage3_unit6_relu1")(stage3_unit6_bn1)

    stage3_unit6_conv1 = Conv2D(
        filters=256,
        kernel_size=(1, 1),
        name="stage3_unit6_conv1",
        strides=[1, 1],
        padding="VALID",
        use_bias=False,
    )(stage3_unit6_relu1)

    stage3_unit6_bn2 = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="stage3_unit6_bn2", trainable=False
    )(stage3_unit6_conv1)

    stage3_unit6_relu2 = ReLU(name="stage3_unit6_relu2")(stage3_unit6_bn2)

    stage3_unit6_conv2_pad = ZeroPadding2D(padding=tuple([1, 1]))(stage3_unit6_relu2)

    stage3_unit6_conv2 = Conv2D(
        filters=256,
        kernel_size=(3, 3),
        name="stage3_unit6_conv2",
        strides=[1, 1],
        padding="VALID",
        use_bias=False,
    )(stage3_unit6_conv2_pad)

    stage3_unit6_bn3 = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="stage3_unit6_bn3", trainable=False
    )(stage3_unit6_conv2)

    stage3_unit6_relu3 = ReLU(name="stage3_unit6_relu3")(stage3_unit6_bn3)

    stage3_unit6_conv3 = Conv2D(
        filters=1024,
        kernel_size=(1, 1),
        name="stage3_unit6_conv3",
        strides=[1, 1],
        padding="VALID",
        use_bias=False,
    )(stage3_unit6_relu3)

    plus12 = Add()([stage3_unit6_conv3, plus11])

    stage4_unit1_bn1 = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="stage4_unit1_bn1", trainable=False
    )(plus12)

    stage4_unit1_relu1 = ReLU(name="stage4_unit1_relu1")(stage4_unit1_bn1)

    stage4_unit1_conv1 = Conv2D(
        filters=512,
        kernel_size=(1, 1),
        name="stage4_unit1_conv1",
        strides=[1, 1],
        padding="VALID",
        use_bias=False,
    )(stage4_unit1_relu1)

    stage4_unit1_sc = Conv2D(
        filters=2048,
        kernel_size=(1, 1),
        name="stage4_unit1_sc",
        strides=[2, 2],
        padding="VALID",
        use_bias=False,
    )(stage4_unit1_relu1)

    stage4_unit1_bn2 = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="stage4_unit1_bn2", trainable=False
    )(stage4_unit1_conv1)

    stage4_unit1_relu2 = ReLU(name="stage4_unit1_relu2")(stage4_unit1_bn2)

    stage4_unit1_conv2_pad = ZeroPadding2D(padding=tuple([1, 1]))(stage4_unit1_relu2)

    stage4_unit1_conv2 = Conv2D(
        filters=512,
        kernel_size=(3, 3),
        name="stage4_unit1_conv2",
        strides=[2, 2],
        padding="VALID",
        use_bias=False,
    )(stage4_unit1_conv2_pad)

    ssh_c2_lateral = Conv2D(
        filters=256,
        kernel_size=(1, 1),
        name="ssh_c2_lateral",
        strides=[1, 1],
        padding="VALID",
        use_bias=True,
    )(stage4_unit1_relu2)

    stage4_unit1_bn3 = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="stage4_unit1_bn3", trainable=False
    )(stage4_unit1_conv2)

    ssh_c2_lateral_bn = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="ssh_c2_lateral_bn", trainable=False
    )(ssh_c2_lateral)

    stage4_unit1_relu3 = ReLU(name="stage4_unit1_relu3")(stage4_unit1_bn3)

    ssh_c2_lateral_relu = ReLU(name="ssh_c2_lateral_relu")(ssh_c2_lateral_bn)

    stage4_unit1_conv3 = Conv2D(
        filters=2048,
        kernel_size=(1, 1),
        name="stage4_unit1_conv3",
        strides=[1, 1],
        padding="VALID",
        use_bias=False,
    )(stage4_unit1_relu3)

    plus13 = Add()([stage4_unit1_conv3, stage4_unit1_sc])

    stage4_unit2_bn1 = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="stage4_unit2_bn1", trainable=False
    )(plus13)

    stage4_unit2_relu1 = ReLU(name="stage4_unit2_relu1")(stage4_unit2_bn1)

    stage4_unit2_conv1 = Conv2D(
        filters=512,
        kernel_size=(1, 1),
        name="stage4_unit2_conv1",
        strides=[1, 1],
        padding="VALID",
        use_bias=False,
    )(stage4_unit2_relu1)

    stage4_unit2_bn2 = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="stage4_unit2_bn2", trainable=False
    )(stage4_unit2_conv1)

    stage4_unit2_relu2 = ReLU(name="stage4_unit2_relu2")(stage4_unit2_bn2)

    stage4_unit2_conv2_pad = ZeroPadding2D(padding=tuple([1, 1]))(stage4_unit2_relu2)

    stage4_unit2_conv2 = Conv2D(
        filters=512,
        kernel_size=(3, 3),
        name="stage4_unit2_conv2",
        strides=[1, 1],
        padding="VALID",
        use_bias=False,
    )(stage4_unit2_conv2_pad)

    stage4_unit2_bn3 = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="stage4_unit2_bn3", trainable=False
    )(stage4_unit2_conv2)

    stage4_unit2_relu3 = ReLU(name="stage4_unit2_relu3")(stage4_unit2_bn3)

    stage4_unit2_conv3 = Conv2D(
        filters=2048,
        kernel_size=(1, 1),
        name="stage4_unit2_conv3",
        strides=[1, 1],
        padding="VALID",
        use_bias=False,
    )(stage4_unit2_relu3)

    plus14 = Add()([stage4_unit2_conv3, plus13])

    stage4_unit3_bn1 = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="stage4_unit3_bn1", trainable=False
    )(plus14)

    stage4_unit3_relu1 = ReLU(name="stage4_unit3_relu1")(stage4_unit3_bn1)

    stage4_unit3_conv1 = Conv2D(
        filters=512,
        kernel_size=(1, 1),
        name="stage4_unit3_conv1",
        strides=[1, 1],
        padding="VALID",
        use_bias=False,
    )(stage4_unit3_relu1)

    stage4_unit3_bn2 = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="stage4_unit3_bn2", trainable=False
    )(stage4_unit3_conv1)

    stage4_unit3_relu2 = ReLU(name="stage4_unit3_relu2")(stage4_unit3_bn2)

    stage4_unit3_conv2_pad = ZeroPadding2D(padding=tuple([1, 1]))(stage4_unit3_relu2)

    stage4_unit3_conv2 = Conv2D(
        filters=512,
        kernel_size=(3, 3),
        name="stage4_unit3_conv2",
        strides=[1, 1],
        padding="VALID",
        use_bias=False,
    )(stage4_unit3_conv2_pad)

    stage4_unit3_bn3 = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="stage4_unit3_bn3", trainable=False
    )(stage4_unit3_conv2)

    stage4_unit3_relu3 = ReLU(name="stage4_unit3_relu3")(stage4_unit3_bn3)

    stage4_unit3_conv3 = Conv2D(
        filters=2048,
        kernel_size=(1, 1),
        name="stage4_unit3_conv3",
        strides=[1, 1],
        padding="VALID",
        use_bias=False,
    )(stage4_unit3_relu3)

    plus15 = Add()([stage4_unit3_conv3, plus14])

    bn1 = BatchNormalization(epsilon=1.9999999494757503e-05, name="bn1", trainable=False)(plus15)

    relu1 = ReLU(name="relu1")(bn1)

    ssh_c3_lateral = Conv2D(
        filters=256,
        kernel_size=(1, 1),
        name="ssh_c3_lateral",
        strides=[1, 1],
        padding="VALID",
        use_bias=True,
    )(relu1)

    ssh_c3_lateral_bn = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="ssh_c3_lateral_bn", trainable=False
    )(ssh_c3_lateral)

    ssh_c3_lateral_relu = ReLU(name="ssh_c3_lateral_relu")(ssh_c3_lateral_bn)

    ssh_m3_det_conv1_pad = ZeroPadding2D(padding=tuple([1, 1]))(ssh_c3_lateral_relu)

    ssh_m3_det_conv1 = Conv2D(
        filters=256,
        kernel_size=(3, 3),
        name="ssh_m3_det_conv1",
        strides=[1, 1],
        padding="VALID",
        use_bias=True,
    )(ssh_m3_det_conv1_pad)

    ssh_m3_det_context_conv1_pad = ZeroPadding2D(padding=tuple([1, 1]))(ssh_c3_lateral_relu)

    ssh_m3_det_context_conv1 = Conv2D(
        filters=128,
        kernel_size=(3, 3),
        name="ssh_m3_det_context_conv1",
        strides=[1, 1],
        padding="VALID",
        use_bias=True,
    )(ssh_m3_det_context_conv1_pad)

    ssh_c3_up = UpSampling2D(size=(2, 2), interpolation="nearest", name="ssh_c3_up")(
        ssh_c3_lateral_relu
    )

    ssh_m3_det_conv1_bn = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="ssh_m3_det_conv1_bn", trainable=False
    )(ssh_m3_det_conv1)

    ssh_m3_det_context_conv1_bn = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="ssh_m3_det_context_conv1_bn", trainable=False
    )(ssh_m3_det_context_conv1)

    x1_shape = tf.shape(ssh_c3_up)
    x2_shape = tf.shape(ssh_c2_lateral_relu)
    offsets = [0, (x1_shape[1] - x2_shape[1]) // 2, (x1_shape[2] - x2_shape[2]) // 2, 0]
    size = [-1, x2_shape[1], x2_shape[2], -1]
    crop0 = tf.slice(ssh_c3_up, offsets, size, "crop0")

    ssh_m3_det_context_conv1_relu = ReLU(name="ssh_m3_det_context_conv1_relu")(
        ssh_m3_det_context_conv1_bn
    )

    plus0_v2 = Add()([ssh_c2_lateral_relu, crop0])

    ssh_m3_det_context_conv2_pad = ZeroPadding2D(padding=tuple([1, 1]))(
        ssh_m3_det_context_conv1_relu
    )

    ssh_m3_det_context_conv2 = Conv2D(
        filters=128,
        kernel_size=(3, 3),
        name="ssh_m3_det_context_conv2",
        strides=[1, 1],
        padding="VALID",
        use_bias=True,
    )(ssh_m3_det_context_conv2_pad)

    ssh_m3_det_context_conv3_1_pad = ZeroPadding2D(padding=tuple([1, 1]))(
        ssh_m3_det_context_conv1_relu
    )

    ssh_m3_det_context_conv3_1 = Conv2D(
        filters=128,
        kernel_size=(3, 3),
        name="ssh_m3_det_context_conv3_1",
        strides=[1, 1],
        padding="VALID",
        use_bias=True,
    )(ssh_m3_det_context_conv3_1_pad)

    ssh_c2_aggr_pad = ZeroPadding2D(padding=tuple([1, 1]))(plus0_v2)

    ssh_c2_aggr = Conv2D(
        filters=256,
        kernel_size=(3, 3),
        name="ssh_c2_aggr",
        strides=[1, 1],
        padding="VALID",
        use_bias=True,
    )(ssh_c2_aggr_pad)

    ssh_m3_det_context_conv2_bn = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="ssh_m3_det_context_conv2_bn", trainable=False
    )(ssh_m3_det_context_conv2)

    ssh_m3_det_context_conv3_1_bn = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="ssh_m3_det_context_conv3_1_bn", trainable=False
    )(ssh_m3_det_context_conv3_1)

    ssh_c2_aggr_bn = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="ssh_c2_aggr_bn", trainable=False
    )(ssh_c2_aggr)

    ssh_m3_det_context_conv3_1_relu = ReLU(name="ssh_m3_det_context_conv3_1_relu")(
        ssh_m3_det_context_conv3_1_bn
    )

    ssh_c2_aggr_relu = ReLU(name="ssh_c2_aggr_relu")(ssh_c2_aggr_bn)

    ssh_m3_det_context_conv3_2_pad = ZeroPadding2D(padding=tuple([1, 1]))(
        ssh_m3_det_context_conv3_1_relu
    )

    ssh_m3_det_context_conv3_2 = Conv2D(
        filters=128,
        kernel_size=(3, 3),
        name="ssh_m3_det_context_conv3_2",
        strides=[1, 1],
        padding="VALID",
        use_bias=True,
    )(ssh_m3_det_context_conv3_2_pad)

    ssh_m2_det_conv1_pad = ZeroPadding2D(padding=tuple([1, 1]))(ssh_c2_aggr_relu)

    ssh_m2_det_conv1 = Conv2D(
        filters=256,
        kernel_size=(3, 3),
        name="ssh_m2_det_conv1",
        strides=[1, 1],
        padding="VALID",
        use_bias=True,
    )(ssh_m2_det_conv1_pad)

    ssh_m2_det_context_conv1_pad = ZeroPadding2D(padding=tuple([1, 1]))(ssh_c2_aggr_relu)

    ssh_m2_det_context_conv1 = Conv2D(
        filters=128,
        kernel_size=(3, 3),
        name="ssh_m2_det_context_conv1",
        strides=[1, 1],
        padding="VALID",
        use_bias=True,
    )(ssh_m2_det_context_conv1_pad)

    ssh_m2_red_up = UpSampling2D(size=(2, 2), interpolation="nearest", name="ssh_m2_red_up")(
        ssh_c2_aggr_relu
    )

    ssh_m3_det_context_conv3_2_bn = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="ssh_m3_det_context_conv3_2_bn", trainable=False
    )(ssh_m3_det_context_conv3_2)

    ssh_m2_det_conv1_bn = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="ssh_m2_det_conv1_bn", trainable=False
    )(ssh_m2_det_conv1)

    ssh_m2_det_context_conv1_bn = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="ssh_m2_det_context_conv1_bn", trainable=False
    )(ssh_m2_det_context_conv1)

    x1_shape = tf.shape(ssh_m2_red_up)
    x2_shape = tf.shape(ssh_m1_red_conv_relu)
    offsets = [0, (x1_shape[1] - x2_shape[1]) // 2, (x1_shape[2] - x2_shape[2]) // 2, 0]
    size = [-1, x2_shape[1], x2_shape[2], -1]
    crop1 = tf.slice(ssh_m2_red_up, offsets, size, "crop1")

    ssh_m3_det_concat = concatenate(
        [ssh_m3_det_conv1_bn, ssh_m3_det_context_conv2_bn, ssh_m3_det_context_conv3_2_bn],
        3,
        name="ssh_m3_det_concat",
    )

    ssh_m2_det_context_conv1_relu = ReLU(name="ssh_m2_det_context_conv1_relu")(
        ssh_m2_det_context_conv1_bn
    )

    plus1_v1 = Add()([ssh_m1_red_conv_relu, crop1])

    ssh_m3_det_concat_relu = ReLU(name="ssh_m3_det_concat_relu")(ssh_m3_det_concat)

    ssh_m2_det_context_conv2_pad = ZeroPadding2D(padding=tuple([1, 1]))(
        ssh_m2_det_context_conv1_relu
    )

    ssh_m2_det_context_conv2 = Conv2D(
        filters=128,
        kernel_size=(3, 3),
        name="ssh_m2_det_context_conv2",
        strides=[1, 1],
        padding="VALID",
        use_bias=True,
    )(ssh_m2_det_context_conv2_pad)

    ssh_m2_det_context_conv3_1_pad = ZeroPadding2D(padding=tuple([1, 1]))(
        ssh_m2_det_context_conv1_relu
    )

    ssh_m2_det_context_conv3_1 = Conv2D(
        filters=128,
        kernel_size=(3, 3),
        name="ssh_m2_det_context_conv3_1",
        strides=[1, 1],
        padding="VALID",
        use_bias=True,
    )(ssh_m2_det_context_conv3_1_pad)

    ssh_c1_aggr_pad = ZeroPadding2D(padding=tuple([1, 1]))(plus1_v1)

    ssh_c1_aggr = Conv2D(
        filters=256,
        kernel_size=(3, 3),
        name="ssh_c1_aggr",
        strides=[1, 1],
        padding="VALID",
        use_bias=True,
    )(ssh_c1_aggr_pad)

    face_rpn_cls_score_stride32 = Conv2D(
        filters=4,
        kernel_size=(1, 1),
        name="face_rpn_cls_score_stride32",
        strides=[1, 1],
        padding="VALID",
        use_bias=True,
    )(ssh_m3_det_concat_relu)

    inter_1 = concatenate(
        [face_rpn_cls_score_stride32[:, :, :, 0], face_rpn_cls_score_stride32[:, :, :, 1]], axis=1
    )
    inter_2 = concatenate(
        [face_rpn_cls_score_stride32[:, :, :, 2], face_rpn_cls_score_stride32[:, :, :, 3]], axis=1
    )
    final = tf.stack([inter_1, inter_2])
    face_rpn_cls_score_reshape_stride32 = tf.transpose(
        final, (1, 2, 3, 0), name="face_rpn_cls_score_reshape_stride32"
    )

    face_rpn_bbox_pred_stride32 = Conv2D(
        filters=8,
        kernel_size=(1, 1),
        name="face_rpn_bbox_pred_stride32",
        strides=[1, 1],
        padding="VALID",
        use_bias=True,
    )(ssh_m3_det_concat_relu)

    face_rpn_landmark_pred_stride32 = Conv2D(
        filters=20,
        kernel_size=(1, 1),
        name="face_rpn_landmark_pred_stride32",
        strides=[1, 1],
        padding="VALID",
        use_bias=True,
    )(ssh_m3_det_concat_relu)

    ssh_m2_det_context_conv2_bn = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="ssh_m2_det_context_conv2_bn", trainable=False
    )(ssh_m2_det_context_conv2)

    ssh_m2_det_context_conv3_1_bn = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="ssh_m2_det_context_conv3_1_bn", trainable=False
    )(ssh_m2_det_context_conv3_1)

    ssh_c1_aggr_bn = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="ssh_c1_aggr_bn", trainable=False
    )(ssh_c1_aggr)

    ssh_m2_det_context_conv3_1_relu = ReLU(name="ssh_m2_det_context_conv3_1_relu")(
        ssh_m2_det_context_conv3_1_bn
    )

    ssh_c1_aggr_relu = ReLU(name="ssh_c1_aggr_relu")(ssh_c1_aggr_bn)

    face_rpn_cls_prob_stride32 = Softmax(name="face_rpn_cls_prob_stride32")(
        face_rpn_cls_score_reshape_stride32
    )

    input_shape = [tf.shape(face_rpn_cls_prob_stride32)[k] for k in range(4)]
    sz = tf.dtypes.cast(input_shape[1] / 2, dtype=tf.int32)
    inter_1 = face_rpn_cls_prob_stride32[:, 0:sz, :, 0]
    inter_2 = face_rpn_cls_prob_stride32[:, 0:sz, :, 1]
    inter_3 = face_rpn_cls_prob_stride32[:, sz:, :, 0]
    inter_4 = face_rpn_cls_prob_stride32[:, sz:, :, 1]
    final = tf.stack([inter_1, inter_3, inter_2, inter_4])
    face_rpn_cls_prob_reshape_stride32 = tf.transpose(
        final, (1, 2, 3, 0), name="face_rpn_cls_prob_reshape_stride32"
    )

    ssh_m2_det_context_conv3_2_pad = ZeroPadding2D(padding=tuple([1, 1]))(
        ssh_m2_det_context_conv3_1_relu
    )

    ssh_m2_det_context_conv3_2 = Conv2D(
        filters=128,
        kernel_size=(3, 3),
        name="ssh_m2_det_context_conv3_2",
        strides=[1, 1],
        padding="VALID",
        use_bias=True,
    )(ssh_m2_det_context_conv3_2_pad)

    ssh_m1_det_conv1_pad = ZeroPadding2D(padding=tuple([1, 1]))(ssh_c1_aggr_relu)

    ssh_m1_det_conv1 = Conv2D(
        filters=256,
        kernel_size=(3, 3),
        name="ssh_m1_det_conv1",
        strides=[1, 1],
        padding="VALID",
        use_bias=True,
    )(ssh_m1_det_conv1_pad)

    ssh_m1_det_context_conv1_pad = ZeroPadding2D(padding=tuple([1, 1]))(ssh_c1_aggr_relu)

    ssh_m1_det_context_conv1 = Conv2D(
        filters=128,
        kernel_size=(3, 3),
        name="ssh_m1_det_context_conv1",
        strides=[1, 1],
        padding="VALID",
        use_bias=True,
    )(ssh_m1_det_context_conv1_pad)

    ssh_m2_det_context_conv3_2_bn = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="ssh_m2_det_context_conv3_2_bn", trainable=False
    )(ssh_m2_det_context_conv3_2)

    ssh_m1_det_conv1_bn = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="ssh_m1_det_conv1_bn", trainable=False
    )(ssh_m1_det_conv1)

    ssh_m1_det_context_conv1_bn = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="ssh_m1_det_context_conv1_bn", trainable=False
    )(ssh_m1_det_context_conv1)

    ssh_m2_det_concat = concatenate(
        [ssh_m2_det_conv1_bn, ssh_m2_det_context_conv2_bn, ssh_m2_det_context_conv3_2_bn],
        3,
        name="ssh_m2_det_concat",
    )

    ssh_m1_det_context_conv1_relu = ReLU(name="ssh_m1_det_context_conv1_relu")(
        ssh_m1_det_context_conv1_bn
    )

    ssh_m2_det_concat_relu = ReLU(name="ssh_m2_det_concat_relu")(ssh_m2_det_concat)

    ssh_m1_det_context_conv2_pad = ZeroPadding2D(padding=tuple([1, 1]))(
        ssh_m1_det_context_conv1_relu
    )

    ssh_m1_det_context_conv2 = Conv2D(
        filters=128,
        kernel_size=(3, 3),
        name="ssh_m1_det_context_conv2",
        strides=[1, 1],
        padding="VALID",
        use_bias=True,
    )(ssh_m1_det_context_conv2_pad)

    ssh_m1_det_context_conv3_1_pad = ZeroPadding2D(padding=tuple([1, 1]))(
        ssh_m1_det_context_conv1_relu
    )

    ssh_m1_det_context_conv3_1 = Conv2D(
        filters=128,
        kernel_size=(3, 3),
        name="ssh_m1_det_context_conv3_1",
        strides=[1, 1],
        padding="VALID",
        use_bias=True,
    )(ssh_m1_det_context_conv3_1_pad)

    face_rpn_cls_score_stride16 = Conv2D(
        filters=4,
        kernel_size=(1, 1),
        name="face_rpn_cls_score_stride16",
        strides=[1, 1],
        padding="VALID",
        use_bias=True,
    )(ssh_m2_det_concat_relu)

    inter_1 = concatenate(
        [face_rpn_cls_score_stride16[:, :, :, 0], face_rpn_cls_score_stride16[:, :, :, 1]], axis=1
    )
    inter_2 = concatenate(
        [face_rpn_cls_score_stride16[:, :, :, 2], face_rpn_cls_score_stride16[:, :, :, 3]], axis=1
    )
    final = tf.stack([inter_1, inter_2])
    face_rpn_cls_score_reshape_stride16 = tf.transpose(
        final, (1, 2, 3, 0), name="face_rpn_cls_score_reshape_stride16"
    )

    face_rpn_bbox_pred_stride16 = Conv2D(
        filters=8,
        kernel_size=(1, 1),
        name="face_rpn_bbox_pred_stride16",
        strides=[1, 1],
        padding="VALID",
        use_bias=True,
    )(ssh_m2_det_concat_relu)

    face_rpn_landmark_pred_stride16 = Conv2D(
        filters=20,
        kernel_size=(1, 1),
        name="face_rpn_landmark_pred_stride16",
        strides=[1, 1],
        padding="VALID",
        use_bias=True,
    )(ssh_m2_det_concat_relu)

    ssh_m1_det_context_conv2_bn = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="ssh_m1_det_context_conv2_bn", trainable=False
    )(ssh_m1_det_context_conv2)

    ssh_m1_det_context_conv3_1_bn = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="ssh_m1_det_context_conv3_1_bn", trainable=False
    )(ssh_m1_det_context_conv3_1)

    ssh_m1_det_context_conv3_1_relu = ReLU(name="ssh_m1_det_context_conv3_1_relu")(
        ssh_m1_det_context_conv3_1_bn
    )

    face_rpn_cls_prob_stride16 = Softmax(name="face_rpn_cls_prob_stride16")(
        face_rpn_cls_score_reshape_stride16
    )

    input_shape = [tf.shape(face_rpn_cls_prob_stride16)[k] for k in range(4)]
    sz = tf.dtypes.cast(input_shape[1] / 2, dtype=tf.int32)
    inter_1 = face_rpn_cls_prob_stride16[:, 0:sz, :, 0]
    inter_2 = face_rpn_cls_prob_stride16[:, 0:sz, :, 1]
    inter_3 = face_rpn_cls_prob_stride16[:, sz:, :, 0]
    inter_4 = face_rpn_cls_prob_stride16[:, sz:, :, 1]
    final = tf.stack([inter_1, inter_3, inter_2, inter_4])
    face_rpn_cls_prob_reshape_stride16 = tf.transpose(
        final, (1, 2, 3, 0), name="face_rpn_cls_prob_reshape_stride16"
    )

    ssh_m1_det_context_conv3_2_pad = ZeroPadding2D(padding=tuple([1, 1]))(
        ssh_m1_det_context_conv3_1_relu
    )

    ssh_m1_det_context_conv3_2 = Conv2D(
        filters=128,
        kernel_size=(3, 3),
        name="ssh_m1_det_context_conv3_2",
        strides=[1, 1],
        padding="VALID",
        use_bias=True,
    )(ssh_m1_det_context_conv3_2_pad)

    ssh_m1_det_context_conv3_2_bn = BatchNormalization(
        epsilon=1.9999999494757503e-05, name="ssh_m1_det_context_conv3_2_bn", trainable=False
    )(ssh_m1_det_context_conv3_2)

    ssh_m1_det_concat = concatenate(
        [ssh_m1_det_conv1_bn, ssh_m1_det_context_conv2_bn, ssh_m1_det_context_conv3_2_bn],
        3,
        name="ssh_m1_det_concat",
    )

    ssh_m1_det_concat_relu = ReLU(name="ssh_m1_det_concat_relu")(ssh_m1_det_concat)
    face_rpn_cls_score_stride8 = Conv2D(
        filters=4,
        kernel_size=(1, 1),
        name="face_rpn_cls_score_stride8",
        strides=[1, 1],
        padding="VALID",
        use_bias=True,
    )(ssh_m1_det_concat_relu)

    inter_1 = concatenate(
        [face_rpn_cls_score_stride8[:, :, :, 0], face_rpn_cls_score_stride8[:, :, :, 1]], axis=1
    )
    inter_2 = concatenate(
        [face_rpn_cls_score_stride8[:, :, :, 2], face_rpn_cls_score_stride8[:, :, :, 3]], axis=1
    )
    final = tf.stack([inter_1, inter_2])
    face_rpn_cls_score_reshape_stride8 = tf.transpose(
        final, (1, 2, 3, 0), name="face_rpn_cls_score_reshape_stride8"
    )

    face_rpn_bbox_pred_stride8 = Conv2D(
        filters=8,
        kernel_size=(1, 1),
        name="face_rpn_bbox_pred_stride8",
        strides=[1, 1],
        padding="VALID",
        use_bias=True,
    )(ssh_m1_det_concat_relu)

    face_rpn_landmark_pred_stride8 = Conv2D(
        filters=20,
        kernel_size=(1, 1),
        name="face_rpn_landmark_pred_stride8",
        strides=[1, 1],
        padding="VALID",
        use_bias=True,
    )(ssh_m1_det_concat_relu)

    face_rpn_cls_prob_stride8 = Softmax(name="face_rpn_cls_prob_stride8")(
        face_rpn_cls_score_reshape_stride8
    )

    input_shape = [tf.shape(face_rpn_cls_prob_stride8)[k] for k in range(4)]
    sz = tf.dtypes.cast(input_shape[1] / 2, dtype=tf.int32)
    inter_1 = face_rpn_cls_prob_stride8[:, 0:sz, :, 0]
    inter_2 = face_rpn_cls_prob_stride8[:, 0:sz, :, 1]
    inter_3 = face_rpn_cls_prob_stride8[:, sz:, :, 0]
    inter_4 = face_rpn_cls_prob_stride8[:, sz:, :, 1]
    final = tf.stack([inter_1, inter_3, inter_2, inter_4])
    face_rpn_cls_prob_reshape_stride8 = tf.transpose(
        final, (1, 2, 3, 0), name="face_rpn_cls_prob_reshape_stride8"
    )

    model = Model(
        inputs=data,
        outputs=[
            face_rpn_cls_prob_reshape_stride32,
            face_rpn_bbox_pred_stride32,
            face_rpn_landmark_pred_stride32,
            face_rpn_cls_prob_reshape_stride16,
            face_rpn_bbox_pred_stride16,
            face_rpn_landmark_pred_stride16,
            face_rpn_cls_prob_reshape_stride8,
            face_rpn_bbox_pred_stride8,
            face_rpn_landmark_pred_stride8,
        ],
    )
    model = load_weights(model)

    return model
