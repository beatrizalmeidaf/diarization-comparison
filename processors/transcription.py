"""
Processamento de transcrição de áudio usando ASR
"""

import os
import logging
from processors.audio_processor import AudioProcessor

logger = logging.getLogger(__name__)

class TranscriptionProcessor:
    """Classe para processar transcrições de áudio"""
    
    def __init__(self, model_name="openai/whisper-small"):
        """
        Inicializa o processador de transcrição
        
        Args:
            model_name: Nome do modelo ASR
        """
        self.model_name = model_name
        self.asr_model = None
        
    def load(self):
        """
        Carrega o modelo ASR
        
        Returns:
            bool: True se carregado com sucesso, False caso contrário
        """
        try:
            from transformers import pipeline
            logger.info(f"Carregando modelo ASR {self.model_name}...")
            self.asr_model = pipeline("automatic-speech-recognition", model=self.model_name)
            logger.info("Modelo ASR carregado com sucesso")
            return True
        except Exception as e:
            logger.error(f"Erro ao carregar modelo ASR: {e}")
            return False
            
    def transcribe_segments(self, audio_path, diarization_output):
        """
        Transcreve segmentos de áudio conforme a diarização
        
        Args:
            audio_path: Caminho do arquivo de áudio
            diarization_output: Saída da diarização no formato PyAnnote
            
        Returns:
            tuple: (transcriptions, full_transcription)
        """
        if self.asr_model is None:
            logger.error("Modelo ASR não carregado. Chame load() primeiro.")
            return {}, ""
            
        # Carregar áudio
        audio, sr = AudioProcessor.load_audio(audio_path)
        if audio is None:
            return {}, ""
            
        transcriptions = {}
        full_transcription = ""
        
        # Para cada falante e seus segmentos
        for segment, _, speaker in diarization_output.itertracks(yield_label=True):
            # Extrair segmento de áudio
            segment_audio = AudioProcessor.extract_segment(audio, sr, segment.start, segment.end)
            if segment_audio is None:
                continue
                
            # Salvar temporariamente
            temp_path = AudioProcessor.save_segment(segment_audio, sr)
            if temp_path is None:
                continue
                
            # Transcrever
            try:
                result = self.asr_model(temp_path)
                text = result["text"] if isinstance(result, dict) else result
                
                if speaker not in transcriptions:
                    transcriptions[speaker] = []
                    
                transcriptions[speaker].append({
                    "start": segment.start,
                    "end": segment.end,
                    "text": text
                })
                
                full_transcription += f"[{speaker}]: {text} "
            except Exception as e:
                logger.error(f"Erro na transcrição do segmento {speaker} {segment}: {e}")
            
            # Remover arquivo temporário
            AudioProcessor.cleanup_temp_file(temp_path)
        
        return transcriptions, full_transcription.strip()