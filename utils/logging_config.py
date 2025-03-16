"""
Configuração centralizada de logging para o projeto
"""

import logging
import os
from datetime import datetime

def setup_logging(log_level=logging.INFO, log_to_file=True):
    """
    Configura o sistema de logging para o projeto
    
    Args:
        log_level: Nível de logging (default: logging.INFO)
        log_to_file: Se True, logs também serão salvos em arquivo
    
    Returns:
        Logger configurado
    """
    # Formatar logs
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(log_format)
    
    # Configurar logger raiz
    logger = logging.getLogger()
    logger.setLevel(log_level)
    
    # Sempre mostrar logs no console
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Logs em arquivo se solicitado
    if log_to_file:
        from config.settings import RESULTS_DIR
        
        # Criar pasta de logs se não existir
        logs_dir = os.path.join(RESULTS_DIR, "logs")
        os.makedirs(logs_dir, exist_ok=True)
        
        # Nome do arquivo de log com timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(logs_dir, f"diarization_comparison_{timestamp}.log")
        
        # Adicionar handler para arquivo
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        logger.info(f"Logs serão salvos em: {log_file}")
    
    # Suprimindo avisos
    import warnings
    warnings.filterwarnings("ignore")
    
    return logger

# Obter logger
def get_logger(name):
    """
    Retorna um logger com o nome especificado
    
    Args:
        name: Nome do logger, geralmente __name__ do módulo
        
    Returns:
        Logger configurado
    """
    return logging.getLogger(name)