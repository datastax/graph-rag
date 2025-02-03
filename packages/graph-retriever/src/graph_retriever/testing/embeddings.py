import json
import math
import random


def angular_2d_embedding(text: str) -> list[float]:
    """
    Convert input text to a 'vector' (list of floats).

    Parameters
    ----------
    text: str
        The text to embed.

    Returns
    -------
    :
        If the text is a number, use it as the angle for the unit vector in
        units of pi.

        Any other input text becomes the singular result `[0, 0]`.
    """
    try:
        angle = float(text)
        return [math.cos(angle * math.pi), math.sin(angle * math.pi)]
    except ValueError:
        # Assume: just test string, no attention is paid to values.
        return [0.0, 0.0]


def earth_embeddings(text: str) -> list[float]:
    """Split words and return a vector based on that."""

    def vector_near(value: float) -> list[float]:
        base_point = [value, (1 - value**2) ** 0.5]
        fluctuation = random.random() / 100.0
        return [base_point[0] + fluctuation, base_point[1] - fluctuation]

    words = set(text.lower().split())
    if "earth" in words:
        return vector_near(0.9)
    elif {"planet", "world", "globe", "sphere"}.intersection(words):
        return vector_near(0.8)
    else:
        return vector_near(0.1)


class ParserEmbeddings:
    """Parse the tuext as a list of floats, otherwise return zeros."""

    def __init__(self, dimension: int = 10) -> None:
        self.dimension = dimension

    def __call__(self, text: str) -> list[float]:
        """Return the embedding."""
        try:
            vals = json.loads(text)
            assert len(vals) == self.dimension
            return vals
        except json.JSONDecodeError:
            return [0.0] * self.dimension


def _string_to_number(word: str) -> int:
    return sum(ord(char) for char in word)


class WordEmbeddings:
    """Embeddings based on a word list."""

    def __init__(self, words: list[str]):
        self._words = words
        self._offsets = [
            _string_to_number(w) * ((-1) ** i) for i, w in enumerate(words)
        ]

    def __call__(self, text: str) -> list[float]:
        """Return the embedding."""
        return [
            1.0 + (100 / self._offsets[i]) if word in text else 0.2 / (i + 1)
            for i, word in enumerate(self._words)
        ]


class AnimalEmbeddings(WordEmbeddings):
    """Embeddings for animal test-case."""

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
