import cv2
import numpy as np
from typing import Optional, Tuple


class FundusPreprocessor:
    """
    Production-grade fundus image preprocessor.
    Handles the real-world garbage: black borders, uneven illumination,
    camera artifacts, JPEG compression, variable resolution.
    """

    @staticmethod
    def circle_crop(img: np.ndarray, tolerance: int = 15) -> np.ndarray:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        _, mask = cv2.threshold(gray, tolerance, 255, cv2.THRESH_BINARY)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return img

        largest = max(contours, key=cv2.contourArea)
        (cx, cy), radius = cv2.minEnclosingCircle(largest)
        cx, cy, r = int(cx), int(cy), int(radius * 0.95)

        cmask = np.zeros_like(gray)
        cv2.circle(cmask, (cx, cy), r, 255, -1)
        result = cv2.bitwise_and(img, img, mask=cmask)

        y1, y2 = max(0, cy - r), min(img.shape[0], cy + r)
        x1, x2 = max(0, cx - r), min(img.shape[1], cx + r)
        cropped = result[y1:y2, x1:x2]

        h, w = cropped.shape[:2]
        if h != w:
            size = max(h, w)
            square = np.zeros((size, size, 3), dtype=np.uint8)
            y_off, x_off = (size - h) // 2, (size - w) // 2
            square[y_off:y_off+h, x_off:x_off+w] = cropped
            return square
        return cropped

    @staticmethod
    def ben_graham(img: np.ndarray, sigma: Optional[float] = None) -> np.ndarray:
        if sigma is None:
            sigma = max(img.shape[:2]) / 30
        img_f = img.astype(np.float32)
        blur = cv2.GaussianBlur(img_f, (0, 0), sigma)
        result = cv2.addWeighted(img_f, 4, blur, -4, 128)
        return np.clip(result, 0, 255).astype(np.uint8)

    @staticmethod
    def clahe_enhance(img: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        l = clahe.apply(l)
        return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)

    def process(self, img: np.ndarray, target_size: int = 224, apply_ben_graham: bool = True, apply_clahe: bool = True) -> np.ndarray:
        img = self.circle_crop(img)
        if apply_ben_graham:
            img = self.ben_graham(img)
        if apply_clahe:
            img = self.clahe_enhance(img)
        img = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_LANCZOS4)
        return img

    def process_bytes(self, img_bytes: bytes, target_size: int = 224) -> np.ndarray:
        img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return self.process(img, target_size)
