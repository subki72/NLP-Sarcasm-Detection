"""
Inference module for loading the fine-tuned DistilBERT model 
and performing sarcasm detection on raw text inputs.
"""
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from src.config import MODEL_PATH, MAX_LENGTH, LABEL_MAP

class SarcasmPredictor:
    """
    A predictor class to encapsulate the model and tokenizer initialization,
    ensuring they are loaded only once for efficient inference.
    """
    
    def __init__(self, model_path: str = MODEL_PATH):
        """
        Initializes the SarcasmPredictor with the specified model path.
        
        Args:
            model_path (str): The directory path containing the pre-trained model and tokenizer.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
            self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {model_path}. Error: {e}")

    def predict(self, text: str) -> dict:
        """
        Predicts whether the input text is sarcastic or genuine.
        
        Args:
            text (str): The raw text input to be classified.
            
        Returns:
            dict: A dictionary containing the original text, predicted label,
                  confidence score (percentage), and a boolean flag.
        """
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            padding=True, 
            max_length=MAX_LENGTH
        )
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
        
        probs = F.softmax(outputs.logits, dim=-1)
        confidence, predicted_class_idx = torch.max(probs, dim=-1)
        
        predicted_label = LABEL_MAP[predicted_class_idx.item()]
        confidence_score = confidence.item()

        return {
            "text": text,
            "prediction": predicted_label,
            "confidence": round(confidence_score * 100, 2),
            "is_sarcastic": bool(predicted_class_idx.item() == 1)
        }

if __name__ == "__main__":
    predictor = SarcasmPredictor()
    test_sentence = "Man Finally Finishes Reading Terms and Conditions Agreement"
    result = predictor.predict(test_sentence)
    print(f"Input: {result['text']}")
    print(f"Output: {result['prediction']} ({result['confidence']}%)")