import pytest
from langchain_core.documents import Document

class BaseTest:
    
    @pytest.fixture
    def test_text(self):
        text = """
        The life of Alexander the Great, one of the most renowned military leaders in history, is a tale of ambition, conquest, and cultural integration. Born in 356 BCE in Pella, the capital of Macedonia, Alexander was the son of King Philip II and Queen Olympias. He was tutored by the philosopher Aristotle, who instilled in him a love for learning and culture.

        Early Life and Education
        Alexander III of Macedon, later known as Alexander the Great, was born on July 20, 356 BCE, in Pella, the capital of Macedonia. His father, King Philip II, united most of the Greek city-states under Macedonian rule. His mother, Queen Olympias, claimed descent from Achilles. Educated by the philosopher Aristotle from age 13 to 16, Alexander studied subjects such as philosophy, politics, and science, developing a lifelong appreciation for knowledge and culture.

        In 340 BCE, at age 16, Alexander acted as regent while Philip campaigned and demonstrated his military aptitude by suppressing a rebellion and founding a city, Alexandropolis.

        Ascension to Power
        When Philip II was assassinated in 336 BCE, Alexander ascended to the throne at age 20. He quickly consolidated power by eliminating rivals and securing loyalty from the Greek city-states, reaffirming Macedonian hegemony through a campaign against Thebes in 335 BCE, which he destroyed as a warning to others.

        Military Campaigns and Conquests
        Conquest of the Persian Empire (334–330 BCE):

        334 BCE: Alexander crossed the Hellespont into Asia Minor and defeated the Persians at the Battle of Granicus.
        333 BCE: At the Battle of Issus, he routed the forces of Darius III, capturing Darius’s family.
        332 BCE: After a lengthy siege, Alexander took Tyre, and in Egypt, he was hailed as a liberator and declared the son of the god Amun. He founded Alexandria, the first of many cities bearing his name.
        331 BCE: In the decisive Battle of Gaugamela, Alexander defeated Darius III, leading to the fall of the Persian Empire.
        """
        return text

    @pytest.fixture
    def test_paragraphs(self, test_text):
        return test_text.strip().split('\n\n')
    
    @pytest.fixture
    def test_documents(self, test_paragraphs):
        return [Document(page_content=paragraph) for paragraph in test_paragraphs]