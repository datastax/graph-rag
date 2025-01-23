import pytest

import sys
import os

from collections.abc import Iterable



# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

# Import the necessary classes
from keybert_metadata_extractor import KeyBERTMetadataExtractor
from base_test import BaseTest

class TestKeyBERTMetadataExtractor(BaseTest):
    
    def test_model_load(self):
        extractor = KeyBERTMetadataExtractor()
        assert extractor._kw_model is not None

    def test_inject_model(self):
        model = KeyBERTMetadataExtractor.load_default_model("all-MiniLM-L6-v2")
        extractor = KeyBERTMetadataExtractor(model=model)
        assert extractor._kw_model is model

    def test_extract_one(self, test_text):
        extractor = KeyBERTMetadataExtractor()
        keywords = extractor.extract_one(test_text)
        assert isinstance(keywords, set)
        assert len(keywords) > 0

    def test_extract_many(self, test_paragraphs):
        
        extractor = KeyBERTMetadataExtractor()
        keywords_iterable = extractor.extract_many(test_paragraphs)
        assert isinstance(keywords_iterable, Iterable)
        assert (len(list(keywords_iterable)) == len(test_paragraphs))

    def test_extract_one_with_kwargs(self, test_text):
        extractor = KeyBERTMetadataExtractor(
            extract_keywords_kwargs={
                'use_maxsum': True,
                'top_n': 10
            }
        )
        keywords = extractor.extract_one(test_text)
        print(keywords)
        assert isinstance(keywords, set)
        assert len(keywords) == 10
        
    def test_extract_many(self, test_documents):
        extractor = KeyBERTMetadataExtractor()
        keywords_iterable = extractor.extract_many(test_documents)
        assert isinstance(keywords_iterable, Iterable)
        assert (len(list(keywords_iterable)) == len(test_documents))