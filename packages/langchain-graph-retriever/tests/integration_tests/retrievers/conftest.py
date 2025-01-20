from tests.integration_tests.invoker import invoker
from tests.integration_tests.retrievers.animal_docs import animal_docs, animal_store

# Imports for definitions.
from tests.integration_tests.stores import (
    enabled_stores,
    store_factory,
    store_param,
)

# Mark these imports as used so they don't removed.
# They need to be imported here so the fixtures are available.
_ = (
    store_factory,
    store_param,
    enabled_stores,
    animal_docs,
    animal_store,
    invoker,
)
