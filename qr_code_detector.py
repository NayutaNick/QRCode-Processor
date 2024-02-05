import cv2
import numpy as np
from Logger import logger
from math import fabs, sin, radians, cos

class Preprocessor:
    def __init__(self, image_path):
        self.image_path = image_path

    def load_image(self):
        img = cv2.imread(self.image_path)
        if img is not None:
            logger.info(f'Image loaded successfully from {self.image_path}')
        else:
            logger.error(f'Failed to load image from {self.image_path}')
        return img

    def preprocess(self, img):
        # 将图像转换为灰度图
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 高斯模糊的参数
        blur_ksize = (7, 7)  # 高斯核的大小
        blur_sigmaX = 0  # X方向的标准差；0 表示根据核大小自动计算

        # 应用高斯模糊以减少噪声
        # img = cv2.GaussianBlur(img, blur_ksize, blur_sigmaX)

        # 自适应阈值的参数
        max_output_value = 255  # THRESH_BINARY 和 THRESH_BINARY_INV 阈值类型时使用的最大值
        adaptive_method = cv2.ADAPTIVE_THRESH_GAUSSIAN_C  # 使用的自适应阈值算法（高斯）
        threshold_type = cv2.THRESH_BINARY  # 阈值类型（二进制）
        block_size = 11  # 用于计算阈值的像素邻域大小
        subtract_from_mean = 2  # 从平均值或加权平均值中减去的常数

        # 应用自适应阈值来二值化图像
        # img = cv2.adaptiveThreshold(img, max_output_value, adaptive_method, threshold_type, block_size, subtract_from_mean)

        logger.info('图像预处理完成。')
        return img

class Detector:
    def __init__(self):
        self.detector = cv2.QRCodeDetector()

    def _detect_qr_code(self, img):
        # Detect and decode the QR code
        data, points, _ = self.detector.detectAndDecode(img)
        if points is not None:
            logger.info(f"二维码占位符解码结果: {data}; 角点坐标为: {points}")
            return points[0].reshape(-1, 2).astype(int)
        else:
            logger.error(f"找不到二维码占位符")
            return None

    def detect_placeholder(self, preprocessed_img):
        # Call the private method to detect the QR code
        corners = self._detect_qr_code(preprocessed_img)
        if corners is not None:
            return np.intp(corners)
        else:
            return None

class Replacer:
    def __init__(self, original_img, qrcode_image_path):
        self.original_img = original_img
        # Convert the original image to BGRA if it is not already
        if self.original_img.shape[2] == 3:
            self.original_img = cv2.cvtColor(self.original_img, cv2.COLOR_BGR2BGRA)
        self.qrcode_image_path = qrcode_image_path
    
    def replace_qrcode(self, placeholder):
        qr_img = cv2.imread(self.qrcode_image_path, cv2.IMREAD_UNCHANGED)
        if qr_img is not None:
            logger.info(f'QR code image loaded successfully from {self.qrcode_image_path}')
        else:
            logger.error(f'Failed to load QR code image from {self.qrcode_image_path}')
            return self.original_img
        adjusted_qr_img = self._adjust_qrcode_image(qr_img, placeholder)
        result_img = self._place_qrcode_in_image(self.original_img, adjusted_qr_img, placeholder)
        logger.info('QR code replacement completed.')
        return result_img

    def _adjust_qrcode_image(self, qr_img, placeholder):
        # 获取二维码的大小和旋转角度
        rect = cv2.minAreaRect(placeholder)
        (x, y), (width, height), angle = rect
        
        if angle <= 45:
            angle_qr = -angle
        else:
            angle_qr = 90 - angle

        # 调整大小
        qr_img_resized = cv2.resize(qr_img, (int(width), int(height)))
        
        # 调整角度
        qr_img_rotated = self._opencv_rotate(qr_img_resized, angle_qr)
        return qr_img_rotated

    def _opencv_rotate(self, img, angle):
        # If the image is not already 4-channel, convert it to BGRA
        if img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
            
        h, w = img.shape[:2]
        center = (w / 2, h / 2)
        scale = 1.005  # cover placeholder's black edge
        M = cv2.getRotationMatrix2D(center, angle, scale)
        new_H = int(w * fabs(sin(radians(angle))) + h * fabs(cos(radians(angle))))
        new_W = int(h * fabs(sin(radians(angle))) + w * fabs(cos(radians(angle))))
        
        logger.info(f"angle = {angle}, h = {h}, w = {w}, new_H = {new_H}, new_W = {new_W}")
        
        # Check if new_H and new_W are too large
        max_dim = 1500
        if new_W >= max_dim or new_H >= max_dim:
            logger.error(f"Rotated image size is too large: {new_W}x{new_H}.")
            return None
        
        M[0, 2] += (new_W - w) / 2
        M[1, 2] += (new_H - h) / 2
        rotated_img = cv2.warpAffine(img, M, (new_W, new_H), borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))
        
        return rotated_img

    def _place_qrcode_in_image(self, original_img, qr_img, placeholder):
        rect = cv2.minAreaRect(placeholder)
        box = cv2.boxPoints(rect)
        box = np.intp(box)

        # Determine the position of the QR code in the original image
        x, y, w, h = cv2.boundingRect(box)
        
        # Adjust width and height based on the scale factor
        scale_factor = 1.01
        w_scaled, h_scaled = int(w * scale_factor), int(h * scale_factor)
        
        # Resize the QR code image based on the scale factor
        qr_img_resized = cv2.resize(qr_img, (w_scaled, h_scaled))
        
        # Create a mask from the alpha channel of the resized QR code image
        mask = qr_img_resized[:, :, 3]

        # Calculate the position offset due to scaling
        offset_x = (w_scaled - w) // 2
        offset_y = (h_scaled - h) // 2

        # Adjust the position to center the QR code in its destination
        x_adjusted, y_adjusted = x - offset_x, y - offset_y

        # Ensure the adjusted positions are within the original image boundaries
        x_adjusted = max(0, min(original_img.shape[1] - w_scaled, x_adjusted))
        y_adjusted = max(0, min(original_img.shape[0] - h_scaled, y_adjusted))

        # Use the mask to blend the QR code into the original image
        for c in range(0, 3):
            original_img[y_adjusted : y_adjusted + h_scaled, x_adjusted : x_adjusted + w_scaled, c] = (
                original_img[y_adjusted : y_adjusted + h_scaled, x_adjusted : x_adjusted + w_scaled, c] * (1 - mask / 255.0) +
                qr_img_resized[:, :, c] * (mask / 255.0)
            )
        
        return original_img

def main():
    image_path = 'original_poster.png'
    qrcode_path = 'qrcode.png'
    
    # 图像预处理
    processor = Preprocessor(image_path)
    original_img = processor.load_image()
    preprocessed_img = processor.preprocess(original_img)
    
    # 二维码占位图检测
    detector = Detector()
    placeholder = detector.detect_placeholder(preprocessed_img)
    
    if placeholder is not None:
        # 二维码替换
        replacer = Replacer(original_img, qrcode_path)
        output_img = replacer.replace_qrcode(placeholder)
        
        # 结果输出
        cv2.imwrite('output.png', output_img)
        logger.info('Output image saved successfully.')
    else:
        logger.error('QR code replacement process completed with no placeholder detected.')

if __name__ == "__main__":
    main()
    