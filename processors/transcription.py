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
                # Lidar com segmentos muito longos
                duration = segment.end - segment.start
                if duration > 25:  # Se o segmento for maior que 25 segundos
                    logger.warning(f"Segmento longo encontrado: {speaker} {segment}. Dividindo em partes menores.")
                    # Dividir em pedaços de 20 segundos com overlap de 2 segundos
                    chunk_size = 20
                    overlap = 2
                    chunk_texts = []
                    
                    for start_time in range(0, int(duration), chunk_size - overlap):
                        end_time = min(start_time + chunk_size, duration)
                        if end_time - start_time < 2:  # Segmento muito pequeno para processar
                            continue
                            
                        chunk_audio = AudioProcessor.extract_segment(audio, sr, 
                                                                   segment.start + start_time, 
                                                                   segment.start + end_time)
                        if chunk_audio is None:
                            continue
                            
                        chunk_path = AudioProcessor.save_segment(chunk_audio, sr)
                        if chunk_path is None:
                            continue
                            
                        # Transcrever o chunk 
                        chunk_result = None
                        try:
                            # Primeiro tenta com as opções mais recentes
                            try:
                                chunk_result = self.asr_model(chunk_path, task="transcribe", language="pt")
                            except TypeError:
                                # Se falhar, tenta apenas com task="transcribe"
                                chunk_result = self.asr_model(chunk_path, task="transcribe")
                        except TypeError:
                            # Para versões mais antigas que não suportam task
                            chunk_result = self.asr_model(chunk_path)
                        
                        chunk_text = chunk_result["text"] if isinstance(chunk_result, dict) else chunk_result
                        chunk_texts.append(chunk_text)
                        
                        # Limpar arquivo temporário
                        AudioProcessor.cleanup_temp_file(chunk_path)
                    
                    # Combinar os resultados
                    text = " ".join(chunk_texts)
                else:
                    # Transcrição normal para segmentos curtos
                    result = None
                    try:
                        # Primeiro tenta com as opções mais recentes
                        try:
                            result = self.asr_model(temp_path, task="transcribe", language="pt")
                        except TypeError:
                            # Se falhar, tenta apenas com task="transcribe"
                            result = self.asr_model(temp_path, task="transcribe")
                    except TypeError:
                        # Para versões mais antigas que não suportam task
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
