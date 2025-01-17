import json
import math
import random
from abc import abstractmethod

from langchain_core.embeddings import Embeddings


class BaseEmbeddings(Embeddings):
    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        pass

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(txt) for txt in texts]


class Angular2DEmbeddings(BaseEmbeddings):
    """
    From angles (as strings in units of pi) to unit embedding vectors on a circle.
    """
    def embed_query(self, text: str) -> list[float]:
        """
        Convert input text to a 'vector' (list of floats).
        If the text is a number, use it as the angle for the
        unit vector in units of pi.
        Any other input text becomes the singular result [0, 0] !
        """
        try:
            angle = float(text)
            return [math.cos(angle * math.pi), math.sin(angle * math.pi)]
        except ValueError:
            # Assume: just test string, no attention is paid to values.
            return [0.0, 0.0]

class EarthEmbeddings(BaseEmbeddings):
    def get_vector_near(self, value: float) -> list[float]:
        base_point = [value, (1 - value**2) ** 0.5]
        fluctuation = random.random() / 100.0
        return [base_point[0] + fluctuation, base_point[1] - fluctuation]

    def embed_query(self, text: str) -> list[float]:
        words = set(text.lower().split())
        if "earth" in words:
            vector = self.get_vector_near(0.9)
        elif {"planet", "world", "globe", "sphere"}.intersection(words):
            vector = self.get_vector_near(0.8)
        else:
            vector = self.get_vector_near(0.1)
        return vector


class ParserEmbeddings(BaseEmbeddings):
    """Parse input texts: if they are json for a List[float], fine.
    Otherwise, return all zeros and call it a day.
    """

    def __init__(self, dimension: int) -> None:
        self.dimension = dimension

    def embed_query(self, text: str) -> list[float]:
        try:
            vals = json.loads(text)
        except json.JSONDecodeError:
            return [0.0] * self.dimension
        else:
            assert len(vals) == self.dimension
            return vals


def string_to_number(word: str) -> int:
    return sum(ord(char) for char in word)


class WordEmbeddings(BaseEmbeddings):
    def __init__(self, words: list[str]):
        self._words = words
        self._offsets = [
            string_to_number(word=word) * ((-1) ** i) for i, word in enumerate(words)
        ]

    def embed_query(self, text: str) -> list[float]:
        return [
            1.0 + (100 / self._offsets[i]) if word in text else 0.2 / (i + 1)
            for i, word in enumerate(self._words)
        ]


class AnimalEmbeddings(WordEmbeddings):
    def __init__(self):
        super().__init__(
            words="""
            alli alpa amer amph ante ante antl appe aqua arct arma aust babo
            badg barr bask bear beav beet beha bird biso bite blac blue boar
            bobc brig buff bugl burr bush butt came cani capy cari carn cass
            cate cham chee chic chim chin chir clim coas coat cobr cock colo
            colo comm comp cour coyo crab cran croa croc crow crus wing wool
            cult cunn curi curl damb danc deer defe defe deme dese digg ding
            dise dist dive dolp dome dome donk dove drag drag duck ecos effo
            eigh elab eleg elev elon euca extr eyes falc famo famo fast fast
            feat feet ferr fier figh finc fish flam flig flig food fore foun
            fres frie frog gaze geck gees gent gill gira goat gori grac gras
            gras graz grou grou grou guin hams hard hawk hedg herb herd hero
            high hipp honk horn hors hove howl huma humm hump hunt hyen iden
            igua inde inse inte jack jagu jell jump jung kang koal komo lark
            larv lemu leop life lion liza lobs long loud loya mada magp mamm
            mana mari mars mass mati meat medi melo meta migr milk mimi moos
            mosq moth narw nati natu neck newt noct nort ocea octo ostr pack
            pain patt peac pest pinc pink play plum poll post pouc powe prec
            pred prey prid prim prob prot prow quac quil rais rapi reac rega
            rege regi rego rein rept resi rive roam rode sava scav seab seaf
            seas semi shar shed shel skil smal snak soci soft soli song song
            soun sout spec spee spik spor spot stag stic stin stin stor stre
            stre stre stro surv surv sust symb tail tall talo team teet tent
            term terr thou tiny tong toug tree agil tuft tund tusk umbr unic
            uniq vast vege veno vibr vita vora wadi wasp wate webb wetl wild
            ant bat bee cat cow dog eel elk emu fox pet pig""".split()
        )
