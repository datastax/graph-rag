from typing import Any, Dict, Iterable, Optional, Set, Union

from langchain_core.documents import Document

from metadata_extractor import AbstractMetadataExtractor

KeybertInput = Union[str, Document]

class KeyBERTMetadataExtractor(AbstractMetadataExtractor[KeybertInput]):
    def __init__(
        self,
        *,
        kind: str = "kw",
        model: Any = None,  # Make model optional with default value None
        embedding_model: str = "all-MiniLM-L6-v2",
        extract_keywords_kwargs: dict[str, Any] | None = None,
    ):
        self._kind = kind
        self._kw_model = model
        self._extract_keywords_kwargs = extract_keywords_kwargs or {}

        # If no model is provided, load the default model
        if self._kw_model is None:
            self._kw_model = KeyBERTMetadataExtractor.load_default_model(embedding_model)

    @staticmethod
    def load_default_model(embedding_model: str) -> Any:
        try:
            import keybert
            return keybert.KeyBERT(model=embedding_model)
        except ImportError:
            raise ImportError(
                "keybert is required for KeybertLinkExtractor. Please install it with `pip install keybert`."
            ) from None

    def extract_one(self, input: KeybertInput) -> set[Any]:
        keywords = self._kw_model.extract_keywords(
            input if isinstance(input, str) else input.page_content,
            **self._extract_keywords_kwargs,
        )
        return set(keywords)

    def extract_many(
        self,
        inputs: Iterable[KeybertInput],
    ) -> Iterable[set[Any]]:
        inputs = list(inputs)
        if len(inputs) == 1:
            # Even though we pass a list, if it contains one item, keybert will
            # flatten it. This means it's easier to just call the special case
            # for one item.
            yield self.extract_one(inputs[0])
        elif len(inputs) > 1:
            strs = [i if isinstance(i, str) else i.page_content for i in inputs]
            extracted = self._kw_model.extract_keywords(strs, **self._extract_keywords_kwargs)
            for keywords in extracted:
                yield set(keywords)
