
# Import the libraries
import os
import torch
import nltk
from tqdm import tqdm
from typing import List, Union
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

class EnglishToPolishTranslator:
    """
    A class for translating English text to Polish using a fine-tuned machine translation model.
    """
    
    def __init__(self, model_path: str = "./final_model", use_gpu: bool = True):
        """
        Initialize the translator with the specified model.
        
        Args:
            model_path (str): Path to the fine-tuned model directory
            use_gpu (bool): Whether to use GPU for inference if available
        """
        self.model_path = model_path
        self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, src_lang="eng_Latn", tgt_lang="pol_Latn")
        
        # Load model
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(self.device).half()
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Download nltk sentence tokenizer if not already downloaded
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
    
    def translate(self, text: Union[str, List[str]], 
                max_length: int = 1024,
                num_beams: int = 50, 
                temperature: float = 0.3,
                batch_size: int = 8) -> Union[str, List[str]]:
        """
        Translate English text to Polish with unified progress tracking.
        
        Args:
            text (str or List[str]): Text to translate. Can be a single string or a list of strings.
            max_length (int): Maximum length of the generated translation
            num_beams (int): Number of beams for beam search
            temperature (float): Temperature for sampling
            
        Returns:
            str or List[str]: Translated text in the same format as input (string or list)
        """
        # Handle single string case
        if isinstance(text, str):
            return self._translate_text(text, max_length, num_beams, temperature)
        
        # Handle list of strings case
        elif isinstance(text, list):
            # Initialize a unified progress bar
            total_sentences = len(text)
            with tqdm(total=total_sentences, desc="Translating Sentences to Polish", unit="sentence") as pbar:
                translated_sentences = []
                for i in range(0, total_sentences, batch_size):  # Process in chunks of batch_size (e.g., 16)
                    batch_sentences = text[i:i+batch_size]
                    batch_translations = self._translate_batch(batch_sentences, max_length, num_beams, temperature)
                    translated_sentences.extend(batch_translations)
                    pbar.update(len(batch_sentences))  # Update progress bar by the number of sentences processed
                return translated_sentences
        
        else:
            raise TypeError("Input must be either a string or a list of strings")
    
    def _translate_text(self, text: str, max_length: int, num_beams: int, temperature: float) -> str:
        """
        Translate a single text by breaking it into sentences, translating each separately,
        and then combining the results.
        
        Args:
            text (str): Text to translate
            max_length (int): Maximum length of translation
            num_beams (int): Number of beams for beam search
            temperature (float): Temperature for sampling
            
        Returns:
            str: Translated text
        """
        # Split text into sentences for better translation
        sentences = nltk.sent_tokenize(text)
        
        # Translate each sentence
        translated_sentences = self._translate_batch(sentences, max_length, num_beams, temperature)
        
        # Join sentences back into a coherent text
        return " ".join(translated_sentences)
    
    def _translate_batch(self, sentences: List[str], max_length: int, num_beams: int, temperature: float) -> List[str]:
        """
        Translate a batch of sentences.
        
        Args:
            sentences (List[str]): List of sentences to translate
            max_length (int): Maximum length of translation
            num_beams (int): Number of beams for beam search
            temperature (float): Temperature for sampling
            
        Returns:
            List[str]: List of translated sentences
        """
        # Set tokenizer languages
        self.tokenizer.src_lang, self.tokenizer.tgt_lang = "eng_Latn", "pol_Latn"
        
        # Tokenize input
        inputs = self.tokenizer(sentences, return_tensors="pt", padding=True, 
                                truncation=True, max_length=max_length)
        
        # Move inputs to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate translations
        # with torch.no_grad():
        # with torch.amp.autocast("cuda"):
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                forced_bos_token_id=self.tokenizer.convert_tokens_to_ids("pol_Latn"),
                max_length=max_length,
                num_beams=num_beams,
                temperature=temperature,
                early_stopping=True
            )
        
        # Decode translations
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)




class PolishToEnglishTranslator:
    """
    A class for translating Polish text to English using a fine-tuned machine translation model.
    """
    
    def __init__(self, model_path: str = "./final_model", use_gpu: bool = True):
        """
        Initialize the translator with the specified model.
        
        Args:
            model_path (str): Path to the fine-tuned model directory
            use_gpu (bool): Whether to use GPU for inference if available
        """
        self.model_path = model_path
        self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, src_lang="pol_Latn", tgt_lang="eng_Latn")
        
        # Load model
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(self.device).half()
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Download nltk sentence tokenizer if not already downloaded
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
    
    def translate(self, text: Union[str, List[str]], 
                max_length: int = 1024,
                num_beams: int = 50, 
                temperature: float = 0.3,
                batch_size: int = 8) -> Union[str, List[str]]:
        """
        Translate English text to Polish with unified progress tracking.
        
        Args:
            text (str or List[str]): Text to translate. Can be a single string or a list of strings.
            max_length (int): Maximum length of the generated translation
            num_beams (int): Number of beams for beam search
            temperature (float): Temperature for sampling
            
        Returns:
            str or List[str]: Translated text in the same format as input (string or list)
        """
        # Handle single string case
        if isinstance(text, str):
            return self._translate_text(text, max_length, num_beams, temperature)
        
        # Handle list of strings case
        elif isinstance(text, list):
            # Initialize a unified progress bar
            total_sentences = len(text)
            with tqdm(total=total_sentences, desc="Translating Sentences to English", unit="sentence") as pbar:
                translated_sentences = []
                for i in range(0, total_sentences, batch_size):  # Process in chunks of batch_size (e.g., 16)
                    batch_sentences = text[i:i+batch_size]
                    batch_translations = self._translate_batch(batch_sentences, max_length, num_beams, temperature)
                    translated_sentences.extend(batch_translations)
                    pbar.update(len(batch_sentences))  # Update progress bar by the number of sentences processed
                return translated_sentences
    
    def _translate_text(self, text: str, max_length: int, num_beams: int, temperature: float) -> str:
        """
        Translate a single text by breaking it into sentences, translating each separately,
        and then combining the results.
        
        Args:
            text (str): Text to translate
            max_length (int): Maximum length of translation
            num_beams (int): Number of beams for beam search
            temperature (float): Temperature for sampling
            
        Returns:
            str: Translated text
        """
        # Split text into sentences for better translation
        sentences = nltk.sent_tokenize(text)
        
        # Translate each sentence
        translated_sentences = self._translate_batch(sentences, max_length, num_beams, temperature)
        
        # Join sentences back into a coherent text
        return " ".join(translated_sentences)
    
    def _translate_batch(self, sentences: List[str], max_length: int, num_beams: int, temperature: float) -> List[str]:
        """
        Translate a batch of sentences.
        
        Args:
            sentences (List[str]): List of sentences to translate
            max_length (int): Maximum length of translation
            num_beams (int): Number of beams for beam search
            temperature (float): Temperature for sampling
            
        Returns:
            List[str]: List of translated sentences
        """
        # Set tokenizer languages
        self.tokenizer.src_lang, self.tokenizer.tgt_lang = "pol_Latn", "eng_Latn"
        
        # Tokenize input
        inputs = self.tokenizer(sentences, return_tensors="pt", padding=True, 
                                truncation=True, max_length=max_length)
        
        # Move inputs to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate translations
        # with torch.no_grad():
        # with torch.amp.autocast("cuda"):
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                forced_bos_token_id=self.tokenizer.convert_tokens_to_ids("eng_Latn"),
                max_length=max_length,
                num_beams=num_beams,
                temperature=temperature,
                early_stopping=True
            )
        
        # Decode translations
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

