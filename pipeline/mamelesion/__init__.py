"""MAM-E lesion synthesis: inpainting de lesoes sinteticas em mamografias saudaveis.

Implementacao da metodologia de MAM-E (Montoya-del-Angel et al.,
arXiv:2311.09822v1 / Sensors 24(7):2076) aplicada a imagens PNG/JPEG locais, com
saida padronizada em 1024x1024, interface interativa e bateria de testes de
qualidade sobre a regiao gerada.
"""

from .config import Config
from .generator import Case, list_images, prepare_case, save_result, synthesize
from .inpainting import GenerationResult, LesionInpainter
from .masks import LesionBox

__version__ = "1.0.0"

__all__ = [
    "Config", "Case", "LesionBox", "LesionInpainter", "GenerationResult",
    "prepare_case", "list_images", "synthesize", "save_result",
]
