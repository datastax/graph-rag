import pytest
import sys
import os
from unittest.mock import MagicMock, patch
from collections.abc import Iterable

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

# Import the necessary classes
from gliner_metadata_extractor import GLiNERMetadataExtractor
from base_test import BaseTest

class TestGLiNERLinkExtractor(BaseTest):
    
    @pytest.fixture
    def default_labels(self):
        return ["people", "places", "dates", "events"]
        
    @pytest.fixture
    def default_model(self):
        return GLiNERMetadataExtractor.load_default_model("urchade/gliner_mediumv2.1")

    def test_model_load(self, default_labels):
        extractor = GLiNERMetadataExtractor(labels=default_labels)
        assert extractor._gliner_model is not None

    def test_inject_model(self, default_labels, default_model):
        extractor = GLiNERMetadataExtractor(labels=default_labels, model=default_model)
        assert extractor._gliner_model is default_model

    def test_extract_one(self, test_text, default_labels):
        extractor = GLiNERMetadataExtractor(labels=default_labels)
        entities = extractor.extract_one(test_text)
        assert isinstance(entities, set)
        assert len(entities) > 0

    def test_extract_many(self, test_paragraphs, default_labels):
        extractor = GLiNERMetadataExtractor(default_labels)
        entities_iterable = extractor.extract_many(test_paragraphs)
        assert isinstance(entities_iterable, Iterable)
        assert len(list(entities_iterable)) == len(test_paragraphs)

    def test_extract_one_with_kwargs(self, test_text, default_labels, default_model):
        extractor1 = GLiNERMetadataExtractor(
            model=default_model,
            labels=default_labels,
            extract_kwargs={
                'threshold': 0.7,
            }
        )
        
        extractor2 = GLiNERMetadataExtractor(
            model=default_model,
            labels=default_labels,
            extract_kwargs={
                'threshold': 0.3,
            }
        )

        entities1 = extractor1.extract_one(test_text)
        print(entities1)
        entities2 = extractor2.extract_one(test_text)
        assert isinstance(entities1, set)
        assert isinstance(entities2, set)
        assert len(entities1) <= len(entities2)
        assert(extractor1._gliner_model == extractor2._gliner_model)
