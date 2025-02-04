import pytest
from graph_retriever.adapters import Adapter
from graph_retriever.adapters.in_memory import InMemory
from graph_retriever.testing.adapter_tests import AdapterComplianceSuite


class TestInMemory(AdapterComplianceSuite):
    @pytest.fixture(scope="class")
    def adapter(self, animals: Adapter) -> Adapter:
        assert isinstance(animals, InMemory)
        return animals
