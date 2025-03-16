"""
Integração com o modelo de diarização SORTFormer do NeMo
"""

import time
import logging
from utils.conversion import convert_diar_to_annotation

logger = logging.getLogger(__name__)

class SortformerModel:
    """Classe para encapsular o modelo de diarização SORTFormer do NeMo"""
    
    def __init__(self, model_name="nvidia/diar_sortformer_4spk-v1"):
        """
        Inicializa o modelo SORTFormer
        
        Args:
            model_name: Nome ou caminho do modelo SORTFormer
        """
        self.model_name = model_name
        self.model = None
        
    def load(self):
        """
        Carrega o modelo SORTFormer
        
        Returns:
            bool: True se carregado com sucesso, False caso contrário
        """
        try:
            logger.info(f"Carregando modelo SORTFormer {self.model_name}...")
            import nemo.collections.asr as nemo_asr
            self.model = nemo_asr.models.SortformerEncLabelModel.from_pretrained(self.model_name)
            self.model.eval()
            logger.info("Modelo SORTFormer carregado com sucesso.")
            return True
        except Exception as e:
            logger.error(f"Erro ao carregar modelo SORTFormer: {e}")
            return False
            
    def process(self, audio_path):
        """
        Processa um arquivo de áudio com SORTFormer
        
        Args:
            audio_path: Caminho do arquivo de áudio
            
        Returns:
            tuple: (output_annotation, processing_time, error)
        """
        if self.model is None:
            logger.error("Modelo SORTFormer não carregado. Chame load() primeiro.")
            return None, 0, "Modelo não carregado"
            
        start_time = time.time()
        try:
            # Usar o modelo diretamente com o método diarize()
            predicted_segments = self.model.diarize(audio=audio_path, batch_size=1)
            
            # Converter a saída para formato PyAnnote
            annotation = convert_diar_to_annotation(predicted_segments)
            
            processing_time = time.time() - start_time
            return annotation, processing_time, None
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Erro no processamento com SORTFormer: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return None, processing_time, str(e)