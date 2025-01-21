from tests.integration_tests.animal_docs import animal_docs, animal_store
from tests.integration_tests.invoker import invoker

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
