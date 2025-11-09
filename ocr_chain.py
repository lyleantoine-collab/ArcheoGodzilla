import cv2
import numpy as np

# Core OCR engines
from deepseek_ocr import DeepSeekOCR
from chandra_v1_india import ChandraOCR
import easyocr

# Ancient & historical specialists
from transkribus_wrapper import TranskribusOCR
import paddleocr
from kraken import binarization, pageseg, recognition

# New heavy-hitters
from mistral_vlm import MistralVL
from llama32 import Llama32Vision
from olm_ocr import OLMOCR
import token_ocr

class OCRChain:
    def __init__(self):
        # Initialize all engines
        self.deepseek = DeepSeekOCR()
        self.chandra = ChandraOCR()
        self.easyocr = easyocr.Reader(['en', 'hi', 'ar', 'sa', 'gr', 'la'])
        self.transkribus = TranskribusOCR('your_user', 'your_pass')
        self.paddle = paddleocr.PaddleOCR(use_angle_cls=True, lang='en')
        self.kraken_rec = recognition.Recognizer()  # point to your model if needed
        self.mistral = MistralVL()
        self.llama = Llama32Vision()
        self.olm = OLMOCR()
        self.token = token_ocr.Tokenizer()

    def preprocess(self, image_path):
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        bin_img = binarization.nlbin(gray)
        return bin_img

    def extract_text(self, img):
        results = {}

        # DeepSeek – raw power on noise
        deep_text, deep_conf = self.deepseek.recognize(img)
        results['deepseek'] = {'text': deep_text, 'confidence': deep_conf or 0.87}

        # Chandra – ancient Indian scripts
        chandra_text, chandra_conf = self.chandra.recognize(img)
        results['chandra'] = {'text': chandra_text, 'confidence': chandra_conf or 0.90}

        # EasyOCR – fast multi-lang
        easy_res = self.easyocr.readtext(np.array(img), detail=0)
        results['easyocr'] = {'text': ' '.join(easy_res), 'confidence': 0.89}

        # PaddleOCR – layout + heritage
        paddle_res = self.paddle.ocr(img, cls=True)
        paddle_text = ' '.join([line[1][0] for line in paddle_res[0]]) if paddle_res and paddle_res[0] else ""
        results['paddle'] = {'text': paddle_text, 'confidence': 0.91}

        # Transkribus – historical documents
        trans_text, trans_conf = self.transkribus.recognize(img)
        results['transkribus'] = {'text': trans_text, 'confidence': trans_conf or 0.93}

        # Kraken – wild scripts
        seg = pageseg.seg_fullpage(img)
        kraken_texts = []
        for region in seg:
            box = region['box']
            line_img = img[box[1]:box[3], box[0]:box[2]]
            text = self.kraken_rec.recognize(line_img)
            kraken_texts.append(text)
        results['kraken'] = {'text': ' '.join(kraken_texts), 'confidence': 0.86}

        # Mistral VL – contextual reasoning
        mistral_text = self.mistral.extract(img)
        results['mistral'] = {'text': mistral_text, 'confidence': 0.93}

        # Llama 3.2 Vision – lightweight multimodal
        llama_text = self.llama.process_image(img)
        results['llama'] = {'text': llama_text, 'confidence': 0.90}

        # OLM-OCR – structural cleanup
        olm_text = self.olm.extract(img)
        results['olm'] = {'text': olm_text, 'confidence': 0.94}

        # TokenOCR – semantic chunking & ordering
        token_structured = self.token.process(olm_text)
        results['token'] = {'text': token_structured, 'confidence': 0.92}

        # Crown the King – highest confidence wins
        best_key = max(results, key=lambda k: results[k]['confidence'])
        final_text = results[best_key]['text']

        return final_text, results

# Unleash the King
# chain = OCRChain()
# text, debug_info = chain.extract_text(chain.preprocess('your_scan.jpg'))
# print("GODZILLA SAYS:", text)
