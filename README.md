# QRCode-Processor

本项目提供一个工具，用于在图像中自动检测二维码占位符，并用新的二维码图像替换它。

其设计的业务场景是：

1. 后台在活动配置页上传一张有占位符二维码的海报模板
2. 前台用户分享活动海报时，海报上便有用户专属的二维码。

    > 理论上用户专属二维码可以被替换为任意图片

## 功能

- **自动检测**：自动定位图像中的二维码占位符。
- **图像预处理**：对原始图像进行处理，以提高二维码检测的准确性。
- **二维码替换**：将检测到的二维码占位符替换为新的二维码图像。
- **角度和大小调整**：自动调整新的二维码图像的角度和大小以匹配占位符。

## 系统要求

- Python 3.x
- OpenCV
- Numpy
- Logging

## 文件结构

- `logger.py`：定义了日志记录功能，用于记录程序的运行情况。
- `original_poster.png`：示例原始海报图片，包含二维码占位符。
- `placeholder_qrcode.png`：示例占位符QR码图像。
- `qr_code_detector.py`：主要的脚本文件，包含 `Preprocessor`、`Detector` 和 `Replacer` 类以及 `main` 函数。
- `qrcode.png`：示例用的新二维码图像，用于替换原有的占位符。

    > 其实它不一定是个二维码，可以替换成任意像替换的图片，但需要是正方形

- `output.png`：脚本处理后输出的图片。
- `logs`：目录，存放日志文件。

`qrcode_detector.py` 的 UML 类图如下：

<img width="703" alt="UML" src="https://github.com/NayutaNick/QRCode-Processor/assets/117813317/7c1727f8-800a-467d-b245-8548ee6be431">


## 安装指南

确保您已经安装了所有必需的 Python 库。可以使用以下命令安装缺失的库：

```bash
pip install numpy opencv-python
```

## 使用说明

1. 将您的原始海报（一定是有一张二维码的）和您想要替换的二维码图像保存在本地。
    > 可以修改 `main` 函数中的 `image_path` 和 `qrcode_path` 变量，以从指向本地文件更改为指向一个url。
2. 确保 `logger.py` 正确配置。
3. 运行 `qr_code_detector.py`（或作为包导入，调用 `qr_code_detector.main`）。

脚本执行后，输出图像将保存在脚本的运行目录下，文件名为 `output.png`。

## 代码结构

### qr_code_detector.py

- `Preprocessor` 类：负责图像的预处理。
- `Detector` 类：利用 `OpenCV` 检测图像中的QR码占位符。
- `Replacer` 类：处理新的二维码图像的调整和替换工作。
- `main`：主函数，协调上述类的工作流程。

### Logger.py

一个日志记录器，稍微根据个人习惯将 `logging` 库优化了一下。日志保存在 `./logs` 文件夹中，唯一日志文件是 `log.log`

可以针对项目需求进行调整、或直接改用 `logging`。

## 注意事项

- 确保原始图像和二维码图像的质量足够高（至少手机能扫上），以确保二维码可以被正确识别和替换。
- 作为占位符的二维码（本项目的示范用例是`placeholder_qrcode.png`）可随意替换成别的二维码，但为了识别的鲁棒性，**内容编码格式请务必是 `utf-8`，并留有一定的白边**
- 预处理步骤的参数可能需要根据您的特定图像进行调整。
