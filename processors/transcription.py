import os
import logging
from processors.audio_processor import AudioProcessor

logger = logging.getLogger(__name__)

class TranscriptionProcessor:
    """Classe para processar transcrições de áudio usando o Whisper diretamente"""
    
    def __init__(self, model_name="openai/whisper-small"):
        """
        Inicializa o processador de transcrição
        
        Args:
            model_name: Nome do modelo Whisper 
        """
        
        if "/" in model_name:
            self.model_name = model_name.split("/")[-1]
        else:
            self.model_name = model_name
            
        
        if "whisper-" in self.model_name:
            self.model_name = self.model_name.replace("whisper-", "")
            
        self.asr_model = None
        
    def load(self):
        """
        Carrega o modelo Whisper diretamente
        
        Returns:
            bool: True se carregado com sucesso, False caso contrário
        """
        try:
            # Importar whisper diretamente
            import whisper
            
            logger.info(f"Carregando modelo Whisper {self.model_name}...")
            self.asr_model = whisper.load_model(self.model_name)
            logger.info("Modelo Whisper carregado com sucesso")
            return True
        except Exception as e:
            logger.error(f"Erro ao carregar modelo Whisper: {e}")
            logger.info("Modelos disponíveis: tiny.en, tiny, base.en, base, small.en, small, medium.en, medium, large-v1, large-v2, large-v3, large")
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
            logger.error("Modelo Whisper não carregado. Chame load() primeiro.")
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
                            
                        # Transcrever o chunk com Whisper forçando português
                        try:
                            # Transcribe com language="pt"
                            chunk_result = self.asr_model.transcribe(chunk_path, language="pt")
                            chunk_text = chunk_result["text"]
                            chunk_texts.append(chunk_text)
                        except Exception as e:
                            logger.error(f"Erro na transcrição do chunk: {e}")
                        
                        # Limpar arquivo temporário
                        AudioProcessor.cleanup_temp_file(chunk_path)
                    
                    # Combinar os resultados
                    text = " ".join(chunk_texts)
                else:
                    # Transcrição normal para segmentos curtos
                    try:
                        # Transcribe com language="pt"
                        result = self.asr_model.transcribe(temp_path, language="pt")
                        text = result["text"]
                    except Exception as e:
                        logger.error(f"Erro na transcrição: {e}")
                        text = ""
                
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
