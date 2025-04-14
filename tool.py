import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import FunctionTransformer
import pickle
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder


class VietnamesePOSTagger:
    def __init__(self):
        self.model = None
        self.label_encoder = None
        self.tagset = {
            "N": 1,    # Danh từ chung
            "Np": 2,   # Danh từ riêng
            "Nc": 3,   # Danh từ chỉ loại
            "Nu": 4,   # Danh từ đơn vị
            "V": 5,    # Động từ
            "A": 6,    # Tính từ
            "P": 7,    # Đại từ
            "L": 8,    # Định từ
            "M": 9,    # Số từ
            "R": 10,   # Phó từ
            "E": 11,   # Giới từ
            "C": 12,   # Liên từ
            "I": 13,   # Thán từ
            "T": 14,   # Trợ từ
            "U": 15,   # Từ đơn lẻ
            "Y": 16,   # Từ viết tắt
            "X": 17,   # Từ không phân loại
            "CH": 18,  # Dấu câu (thêm vào cho đầy đủ) // khi tool predict thì annotator sẽ sữa lại cho giống guidelines của cô
        }

    def _preprocess_word(self, word):
        """Tiền xử lý từ, loại bỏ dấu câu nếu đính kèm"""
        word = word.strip()
        if len(word) > 1 and word[-1] in ",.?!;:-/()[]{}\"'":
            return word[:-1]
        return word

    def _extract_features(self, word, index=None, words=None):
        """Trích xuất đặc trưng cho một từ và ngữ cảnh của nó"""
        word = self._preprocess_word(word)

        features = {
            'word': word.lower(),
            'is_capitalized': 1 if word and word[0].isupper() else 0,
            'is_all_caps': 1 if word.isupper() else 0,
            'has_underscore': 1 if '_' in word else 0,
            'has_digit': 1 if any(c.isdigit() for c in word) else 0,
            'is_digit': 1 if word.isdigit() else 0,
            'word_len': len(word),
            'prefix_1': word[0] if word else '',
            'prefix_2': word[:2] if len(word) > 1 else '',
            'prefix_3': word[:3] if len(word) > 2 else '',
            'suffix_1': word[-1] if word else '',
            'suffix_2': word[-2:] if len(word) > 1 else '',
            'suffix_3': word[-3:] if len(word) > 2 else '',
            'is_punctuation': 1 if word in ",.?!;:-/()[]{}\"'" else 0,
        }

        if words is not None and index is not None:
            if index > 0:
                prev_word = self._preprocess_word(words[index-1])
                features['prev_word'] = prev_word.lower()
                features['prev_is_cap'] = 1 if prev_word and prev_word[0].isupper(
                ) else 0
                features['prev_pos'] = 'BOS' if index == 0 else ''
            else:
                features['prev_word'] = 'BOS'
                features['prev_is_cap'] = 0
                features['prev_pos'] = 'BOS'

            if index < len(words) - 1:
                next_word = self._preprocess_word(words[index+1])
                features['next_word'] = next_word.lower()
                features['next_is_cap'] = 1 if next_word and next_word[0].isupper(
                ) else 0
                features['next_pos'] = 'EOS' if index == len(words) - 1 else ''
            else:
                features['next_word'] = 'EOS'
                features['next_is_cap'] = 0
                features['next_pos'] = 'EOS'

            if index > 0 and index < len(words) - 1:
                features['bigram'] = words[index-1].lower() + '_' + \
                    words[index+1].lower()

        return features

    def load_data_from_file(self, file_path):
        """Đọc dữ liệu từ file văn bản với định dạng từ/nhãn"""
        sentences = []
        current_sentence = []

        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:  
                    if current_sentence:
                        sentences.append(current_sentence)
                        current_sentence = []
                    continue

                tokens = line.split()
                for token in tokens:
                    parts = token.split('/')
                    if len(parts) >= 2:
                        word = '/'.join(parts[:-1])
                        tag = parts[-1]
                        if word == tag:
                            current_sentence.append((word, "CH")) # thêm vào để đảm bảo đúng với guidelines của cô
                            continue
                        current_sentence.append((word, tag))

               
                if current_sentence:
                    sentences.append(current_sentence)
                    current_sentence = []

        return sentences

    def load_data_from_text(self, text):
        """Đọc dữ liệu từ chuỗi văn bản với định dạng từ/nhãn"""
        sentences = []
        current_sentence = []

        lines = text.strip().split('\n')
        for line in lines:
            line = line.strip()
            if not line: 
                if current_sentence:
                    sentences.append(current_sentence)
                    current_sentence = []
                continue

            tokens = line.split()
            for token in tokens:
                parts = token.rsplit('/', 1) 
                if len(parts) == 2:
                    word, tag = parts
                    current_sentence.append((word, tag))

            
            if current_sentence:
                sentences.append(current_sentence)
                current_sentence = []

        return sentences

    def _prepare_data(self, sentences):
        """Chuẩn bị dữ liệu để huấn luyện mô hình"""
        X = []
        y = []

        for sentence in sentences:
            words = [token[0] for token in sentence]
            tags = [token[1] for token in sentence]

            for i, (word, tag) in enumerate(zip(words, tags)):
                features = self._extract_features(word, i, words)
                X.append(features)
                y.append(tag)

        return X, y

    def train(self, data, test_size=0.5, random_state=42):
    
        X, y = self._prepare_data(data)
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)

        # X_df = pd.DataFrame(X)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=random_state
        )
        self.model = Pipeline([
            ('vectorizer', DictVectorizer()),
            ('classifier', LogisticRegression(max_iter=100, C=1.0, solver='lbfgs',
                                              multi_class='multinomial', verbose=1))
        ])

    
        self.model.fit(X_train, y_train)

      
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Độ chính xác: {accuracy:.4f}")
        return self

    def predict_sentence(self, sentence):
        """
        Dự đoán nhãn POS cho một câu
        
        Parameters:
        sentence: str - Câu cần gán nhãn
        
        Returns:
        List[Tuple(str, str)] - Danh sách các cặp (từ, nhãn POS)
        """
        if self.model is None:
            raise ValueError("Mô hình chưa được huấn luyện!")

    
        words = sentence.split()

    
        X = []
        for i, word in enumerate(words):
            features = self._extract_features(word, i, words)
            X.append(features)
        predictions_encoded = self.model.predict(X)
        predictions = self.label_encoder.inverse_transform(predictions_encoded)

        return list(zip(words, predictions))

    def predict_file(self, input_file, output_file=None):
        """
        Dự đoán nhãn POS cho một tệp văn bản và lưu kết quả vào tệp
        
        Parameters:
        input_file: str - Đường dẫn đến tệp đầu vào
        output_file: str - Đường dẫn đến tệp đầu ra (mặc định: input_file + '.tagged')
        
        Returns:
        None
        """
        if self.model is None:
            raise ValueError("Mô hình chưa được huấn luyện!")

        if output_file is None:
            output_file = input_file + '.tagged'

        with open(input_file, 'r', encoding='utf-8') as f_in, open(output_file, 'w', encoding='utf-8') as f_out:
            for line in f_in:
                line = line.strip()
                if not line:
                    f_out.write('\n')
                    continue

                tagged_tokens = self.predict_sentence(line)
                tagged_line = ' '.join(
                    [f"{word}/{tag}" for word, tag in tagged_tokens])
                f_out.write(tagged_line + '\n')

    def save_model(self, filepath):
        """Lưu mô hình vào file"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'label_encoder': self.label_encoder,
                'tagset': self.tagset
            }, f)
        print(f"Đã lưu mô hình vào {filepath}")

    def load_model(self, filepath):
        """Tải mô hình từ file"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.label_encoder = data['label_encoder']
            self.tagset = data.get('tagset', self.tagset)
        print(f"Đã tải mô hình từ {filepath}")
        return self
if __name__ == "__main__":
    tagger = VietnamesePOSTagger().load_model(
        "/kaggle/working/vietnamese_pos_model.pkl")
    test_sentence = "Sáng nay trời đẹp ? và trong xanh."
    result = tagger.predict_sentence(test_sentence)

    print("\nKết quả gán nhãn POS cho câu mẫu:")
    for word, tag in result:
        print(f"{word}/{tag}")
    test_sentence2 = "Những người lính Thái Mỹ đã chiến đấu anh dũng, họ là niềm tự hào của dân tộc."
    result2 = tagger.predict_sentence(test_sentence2)

    print("\nKết quả gán nhãn POS với mô hình đã tải:")
    for word, tag in result2:
        print(f"{word}/{tag}")
