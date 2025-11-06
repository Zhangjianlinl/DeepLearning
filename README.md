# 项目快速上手（RetinaFace 本地可运行示例）

本仓库包含：
- 上游 `retinaface` 库源码（位于 `retinaface/`）
- 一个可直接运行的命令行示例 `demo.py`（已添加），便于把“库项目”当作“完整项目”直接使用

## 一、环境准备

建议使用虚拟环境（Windows PowerShell）：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
```

安装依赖（两种方式任选其一）：

- 使用本地源码依赖：
```powershell
pip install -r retinaface/requirements.txt
# 若 TensorFlow=2.16+ 报 Keras3 相关错误，再执行：
pip install tf-keras
```

- 直接使用 PyPI 包：
```powershell
pip install retina-face
```

> 首次运行时会自动下载预训练权重到用户目录 `~/.deepface/weights/retinaface.h5`，需要网络。

## 二、运行示例命令

### 1) 人脸检测（输出 JSON）
```powershell
python demo.py detect --image retinaface/tests/dataset/img3.jpg --threshold 0.9
```

### 2) 检测+对齐+裁剪保存（输出图片）
```powershell
python demo.py extract --image retinaface/tests/dataset/img3.jpg --output outputs --expand 20 --target-size 224x224 --min-max-norm
```

### 3) 图片可视化（在图上画框和关键点）
```powershell
python demo.py visualize --image retinaface/tests/dataset/img3.jpg --output visualized.jpg --threshold 0.9 --show
```

### 4) 摄像头实时检测（按 q 退出）
```powershell
python demo.py webcam --camera 0 --threshold 0.9 --width 640 --height 480
```

## 三、参数说明

- 通用：
  - `--threshold`：检测阈值，默认 `0.9`（越高越严格）
  - `--no-upscale`：禁止对小图上采样（默认允许）

- `detect`：
  - `--image`：输入图片路径（必填）

- `extract`：
  - `--image`：输入图片路径（必填）
  - `--output`：输出目录，默认 `outputs`
  - `--no-align`：不做对齐
  - `--expand`：扩大人脸区域百分比
  - `--target-size`：目标尺寸，格式 `WxH`，如 `224x224`
  - `--min-max-norm`：与 `--target-size` 搭配，将像素归一化到 `[0,1]`

- `visualize`：
  - `--image`：输入图片路径（必填）
  - `--output`：输出图路径，默认 `visualized.jpg`
  - `--show`：弹窗显示结果

- `webcam`：
  - `--camera`：摄像头索引，默认 `0`
  - `--width` / `--height`：捕获分辨率

## 四、直接在代码里调用（可选）

```python
from retinaface import RetinaFace

# 检测
faces = RetinaFace.detect_faces("retinaface/tests/dataset/img3.jpg")
print(faces)

# 提取并对齐
imgs = RetinaFace.extract_faces(
    img_path="retinaface/tests/dataset/img3.jpg",
    align=True, expand_face_area=20, target_size=(224, 224)
)
print(len(imgs), imgs[0].shape)
```

## 五、常见问题

- 运行时报 `tf.keras / Keras 3` 相关错误：
  - 安装 `tf-keras`：`pip install tf-keras`
- 权重下载失败：
  - 检查网络；或手动下载权重放到 `~/.deepface/weights/retinaface.h5`
- oneDNN / TF 日志过多：
  - 可设置环境变量：`$env:TF_ENABLE_ONEDNN_OPTS="0"`

完成！现在你可以把本仓库当作“开箱即用”的人脸检测/对齐/可视化/实时工具来使用。
