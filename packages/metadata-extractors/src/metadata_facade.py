from typing import List, Dict, Any, Optional
from langchain_core.documents import Document

from .gliner_metadata_extractor import GlinerMetadataExtractor
from .html_metadata_extractor import HtmlMetadataExtractor
from .keybert_metadata_extractor import KeyBertMetadataExtractor

class MetadataExtractor:
    """A unified interface for metadata extraction strategies."""

    _models: Dict[str, Any] = {}
    _strategies: Dict[str, Any] = {}

    @staticmethod
    def initialize_strategies():
        """
        Initialize the strategy instances.
        """
        MetadataExtractor._strategies = {
            "gliner": GlinerMetadataExtractor(),
            "html": HtmlMetadataExtractor(),
            "keybert": KeyBertMetadataExtractor(),
        }

    @staticmethod
    def initialize_model(strategy: str):
        """
        Initialize the model for a specific strategy.

        Args:
            strategy (str): The strategy for which to initialize the model.
            embedding_model (Optional[str]): The embedding model to use for the KeyBERT strategy.
        """
        if strategy == "keybert":
            try:
                import keybert
                model = keybert.KeyBERT(model=embedding_model or "all-MiniLM-L6-v2")
                MetadataExtractor._models[strategy] = model
            except ImportError:
                raise ImportError(
                    "keybert is required for KeyBertMetadataExtractor. "
                    "Please install it with `pip install keybert`."
                ) from None
        elif strategy == "gliner":
            # Initialize the model for GlinerMetadataExtractor
            try:
                from gliner import GLiNER
                self._model = GLiNER.from_pretrained(model)
            except ImportError:
                raise ImportError(
                    "gliner is required for GLiNERLinkExtractor. "
                    "Please install it with `pip install gliner`."
                ) from None
                MetadataExtractor._models[strategy] = model
                
        elif strategy == "html":
            # Initialize the model for HtmlMetadataExtractor
            model = HtmlMetadataExtractor.initialize_model()
            MetadataExtractor._models[strategy] = model
        else:
            raise ValueError(f"Unknown strategy '{strategy}'.")



    @staticmethod
    def extract_metadata(strategy: str, document: Document) -> List[str]:
        """
        Extract metadata using the specified strategy.

        Args:
            strategy (str): The extraction strategy to use.
            document (Document): The document to process.

        Returns:
            List[str]: A list of extracted metadata.
        """
        if strategy not in MetadataExtractor._strategies:
            raise ValueError(f"Unknown strategy '{strategy}'. Available strategies: {list(MetadataExtractor._strategies.keys())}")

        if strategy not in MetadataExtractor._models:
            MetadataExtractor.initialize_models({strategy: None})   
        # Get the strategy and the associated model
        strategy_instance = MetadataExtractor._strategies[strategy]
        model = MetadataExtractor._models.get(strategy)

        if not model:
            raise ValueError(f"No model provided for strategy '{strategy}'.")

        # Call the strategy's extract method with the document and model
        return strategy_instance.extract(document, model)

# Initialize the strategies once
MetadataExtractor.initialize_strategies()