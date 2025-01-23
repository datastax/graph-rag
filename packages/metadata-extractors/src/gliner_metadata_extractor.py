from typing import Any, Dict, Iterable, Optional, Set, Union, List

from langchain_core.documents import Document

from metadata_extractor import AbstractMetadataExtractor

GLiNERInput = Union[str, Document]

class GLiNERMetadataExtractor(AbstractMetadataExtractor[GLiNERInput]):
    def __init__(
        self,
        labels: List[str],
        *,
        kind: str = "entity",
        model: Any = None,  # Make model optional with default value None
        embedding_model: str = "urchade/gliner_mediumv2.1",
        extract_kwargs: Optional[Dict[str, Any]] = None,
    ):
        
        self._labels = labels
        self._kind = kind
        self._extract_kwargs = extract_kwargs or {}
        self._gliner_model = model
        
        # If no model is provided, load the default model
        if self._gliner_model is None:
            self._gliner_model = GLiNERMetadataExtractor.load_default_model(embedding_model)

    @staticmethod
    def load_default_model(embedding_model: str) -> Any:
        try:
            from gliner import GLiNER
            return GLiNER.from_pretrained(embedding_model)
        except ImportError:
            raise ImportError(
                "gliner is required for GLiNERLinkExtractor. "
                "Please install it with `pip install gliner`."
            ) from None

        self._kind = kind
        self._extract_kwargs = extract_kwargs or {}

    def extract_one(self, input: GLiNERInput) -> Set[Any]:  # noqa: A002
        return next(iter(self.extract_many([input])))

    def extract_many(
        self,
        inputs: Iterable[GLiNERInput],
    ) -> Iterable[Set[Any]]:
        strs = [i if isinstance(i, str) else i.page_content for i in inputs]
        for entities in self._gliner_model.batch_predict_entities(
            strs, self._labels, **self._extract_kwargs, 
        ):
            yield {
                (f"{e['label']}:{e['text']}", e['score'])
                for e in entities
            }
