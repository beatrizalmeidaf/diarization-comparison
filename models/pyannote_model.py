"""
Integração com o modelo de diarização PyAnnote
"""

import time
import logging
from pyannote.audio import Pipeline

logger = logging.getLogger(__name__)

class PyannoteModel:
    """Classe para encapsular o modelo de diarização PyAnnote"""
    
    def __init__(self, model_name, auth_token):
        """
        Inicializa o modelo PyAnnote
        
        Args:
            model_name: Nome do modelo PyAnnote
            auth_token: Token de autenticação HuggingFace
        """
        self.model_name = model_name
        self.auth_token = auth_token
        self.pipeline = None
        
    def load(self):
        """
        Carrega o modelo PyAnnote
        
        Returns:
            bool: True se carregado com sucesso, False caso contrário
        """
        try:
            logger.info(f"Carregando modelo PyAnnote {self.model_name}...")
            self.pipeline = Pipeline.from_pretrained(
                self.model_name,
                use_auth_token=self.auth_token
            )
            logger.info("Modelo PyAnnote carregado com sucesso.")
            return True
        except Exception as e:
            logger.error(f"Erro ao carregar modelo PyAnnote: {e}")
            return False
            
    def process(self, audio_path):
        """
        Processa um arquivo de áudio com PyAnnote
        
        Args:
            audio_path: Caminho do arquivo de áudio
            
        Returns:
            tuple: (output, processing_time, error)
        """
        if self.pipeline is None:
            logger.error("Modelo PyAnnote não carregado. Chame load() primeiro.")
            return None, 0, "Modelo não carregado"
            
        start_time = time.time()
        try:
            pyannote_output = self.pipeline(audio_path)
            processing_time = time.time() - start_time
            return pyannote_output, processing_time, None
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Erro no processamento com PyAnnote: {e}")
            return None, processing_time, str(e)