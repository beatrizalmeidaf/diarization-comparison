#!/usr/bin/env python
"""
Ponto de entrada principal para o pipeline de comparação de diarização
entre modelos PyAnnote e SORTFormer
"""

import os
import json
import pandas as pd
import argparse
from datetime import datetime

# Importar configurações e utilidades
from config.settings import AUDIO_FILES, RESULTS_DIR, PYANNOTE_MODEL, PYANNOTE_AUTH_TOKEN, ASR_MODEL
from utils.logging_config import setup_logging, get_logger
from utils.visualization import (
    visualize_diarization, 
    plot_time_comparison, 
    plot_metrics_comparison,
    print_summary_statistics
)

# Importar modelos
from models import PyannoteModel, SortformerModel

# Importar processadores
from processors import AudioProcessor, TranscriptionProcessor, EvaluationProcessor

def parse_arguments():
    """
    Processa os argumentos da linha de comando
    
    Returns:
        argparse.Namespace: Argumentos processados
    """
    parser = argparse.ArgumentParser(description='Pipeline de comparação entre modelos de diarização')
    
    parser.add_argument('--audio', nargs='+', help='Caminhos dos arquivos de áudio para processamento')
    parser.add_argument('--output-dir', help='Diretório de saída para resultados')
    parser.add_argument('--skip-transcription', action='store_true', help='Pular a etapa de transcrição')
    parser.add_argument('--skip-visualization', action='store_true', help='Pular a etapa de visualização')
    parser.add_argument('--pyannote-only', action='store_true', help='Executar apenas o modelo PyAnnote')
    parser.add_argument('--sortformer-only', action='store_true', help='Executar apenas o modelo SORTFormer')
    
    return parser.parse_args()

def setup_output_dir(base_dir=None):
    """
    Configura o diretório de saída para os resultados
    
    Args:
        base_dir: Diretório base para criar o diretório de resultados
        
    Returns:
        str: Caminho do diretório de saída
    """
    if base_dir is None:
        base_dir = RESULTS_DIR
        
    # Criar diretório com timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(base_dir, f"run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    return output_dir

def process_audio_file(audio_file, output_dir, models, processors, options):
    """
    Processa um único arquivo de áudio com ambos os modelos
    
    Args:
        audio_file: Caminho do arquivo de áudio
        output_dir: Diretório de saída
        models: Dicionário com os modelos carregados
        processors: Dicionário com os processadores
        options: Dicionário com opções de processamento
        
    Returns:
        dict: Resultados do processamento
    """
    logger = get_logger(__name__)
    logger.info(f"Processando {audio_file}...")
    
    if not os.path.exists(audio_file):
        logger.error(f"Arquivo {audio_file} não encontrado. Pulando.")
        return None
    
    # Resultados para este arquivo
    result = {
        "audio": audio_file,
        "pyannote_time": None,
        "sortformer_time": None,
        "DER": "N/A",
        "WER": "N/A",
        "CER": "N/A",
        "speed_ratio": None
    }
    
    detailed_result = {
        "audio": audio_file,
        "pyannote": {"time": None, "error": None},
        "sortformer": {"time": None, "error": None},
        "comparison": {"DER": "N/A", "WER": "N/A", "CER": "N/A", "speed_ratio": None}
    }
    
    # Processar com PyAnnote
    pyannote_output = None
    if not options.get("sortformer_only", False):
        pyannote_output, pyannote_time, pyannote_error = models["pyannote"].process(audio_file)
        result["pyannote_time"] = pyannote_time
        detailed_result["pyannote"]["time"] = pyannote_time
        detailed_result["pyannote"]["error"] = pyannote_error
        
        if pyannote_error:
            logger.error(f"Erro no processamento com PyAnnote: {pyannote_error}")
    
    # Processar com SORTFormer
    sortformer_output = None
    if not options.get("pyannote_only", False):
        sortformer_output, sortformer_time, sortformer_error = models["sortformer"].process(audio_file)
        result["sortformer_time"] = sortformer_time
        detailed_result["sortformer"]["time"] = sortformer_time
        detailed_result["sortformer"]["error"] = sortformer_error
        
        if sortformer_error:
            logger.error(f"Erro no processamento com SORTFormer: {sortformer_error}")
    
    # Calcular razão de velocidade se ambos os modelos foram executados
    if result["pyannote_time"] and result["sortformer_time"]:
        result["speed_ratio"] = result["sortformer_time"] / result["pyannote_time"]
        detailed_result["comparison"]["speed_ratio"] = result["speed_ratio"]
    
    # Avaliar diarização se ambos os modelos foram bem-sucedidos
    if pyannote_output and sortformer_output:
        metrics = EvaluationProcessor.evaluate_diarization(pyannote_output, sortformer_output)
        result["DER"] = metrics["DER"]
        detailed_result["comparison"]["DER"] = metrics["DER"]
        logger.info(f"DER calculado: {metrics['DER']}")
        
        # Visualizar resultados
        if not options.get("skip_visualization", False):
            output_image = os.path.join(output_dir, f"{os.path.basename(audio_file)}_comparison.png")
            visualize_diarization(audio_file, pyannote_output, sortformer_output, output_image)
            logger.info(f"Visualização salva em {output_image}")
    
    # Transcrição e cálculo de WER/CER
    if not options.get("skip_transcription", False) and pyannote_output and sortformer_output:
        if processors.get("transcription") and processors["transcription"].asr_model:
            logger.info("Transcrevendo segmentos para cálculo de WER/CER...")
            
            # Transcrever com PyAnnote
            pyannote_transcriptions, pyannote_full = processors["transcription"].transcribe_segments(
                audio_file, pyannote_output)
            
            # Transcrever com SORTFormer
            sortformer_transcriptions, sortformer_full = processors["transcription"].transcribe_segments(
                audio_file, sortformer_output)
            
            # Salvar transcrições
            with open(os.path.join(output_dir, f"{os.path.basename(audio_file)}_pyannote_transcriptions.json"), 'w') as f:
                json.dump(pyannote_transcriptions, f, indent=4)
                
            with open(os.path.join(output_dir, f"{os.path.basename(audio_file)}_sortformer_transcriptions.json"), 'w') as f:
                json.dump(sortformer_transcriptions, f, indent=4)
            
            # Calcular WER/CER
            wer_cer_metrics = EvaluationProcessor.calculate_wer_cer(pyannote_full, sortformer_full)
            result["WER"] = wer_cer_metrics["WER"]
            result["CER"] = wer_cer_metrics["CER"]
            detailed_result["comparison"]["WER"] = wer_cer_metrics["WER"]
            detailed_result["comparison"]["CER"] = wer_cer_metrics["CER"]
            logger.info(f"WER: {wer_cer_metrics['WER']}, CER: {wer_cer_metrics['CER']}")
    
    return result, detailed_result

def main():
    """Função principal do pipeline"""
    # Processar argumentos
    args = parse_arguments()
    
    # Configurar logging
    logger = setup_logging()
    logger.info("Iniciando pipeline de comparação de diarização...")
    
    # Configurar diretório de saída
    output_dir = args.output_dir if args.output_dir else setup_output_dir()
    logger.info(f"Resultados serão salvos em: {output_dir}")
    
    # Determinar arquivos de áudio para processamento
    audio_files = args.audio if args.audio else AUDIO_FILES
    logger.info(f"Arquivos de áudio a serem processados: {audio_files}")
    
    # Opções de processamento
    options = {
        "skip_transcription": args.skip_transcription,
        "skip_visualization": args.skip_visualization,
        "pyannote_only": args.pyannote_only,
        "sortformer_only": args.sortformer_only
    }
    
    # Carregar modelos
    models = {}
    
    # PyAnnote
    if not args.sortformer_only:
        pyannote_model = PyannoteModel(PYANNOTE_MODEL, PYANNOTE_AUTH_TOKEN)
        if pyannote_model.load():
            models["pyannote"] = pyannote_model
        else:
            logger.error("Falha ao carregar modelo PyAnnote. Saindo.")
            if args.pyannote_only:
                return
    
    # SORTFormer
    if not args.pyannote_only:
        sortformer_model = SortformerModel()
        if sortformer_model.load():
            models["sortformer"] = sortformer_model
        else:
            logger.error("Falha ao carregar modelo SORTFormer. Saindo.")
            if args.sortformer_only:
                return
    
    # Carregar processadores
    processors = {}
    
    # Transcrição (ASR)
    if not args.skip_transcription:
        transcription_processor = TranscriptionProcessor(ASR_MODEL)
        if transcription_processor.load():
            processors["transcription"] = transcription_processor
        else:
            logger.warning("Falha ao carregar processador de transcrição. Métricas WER/CER não serão calculadas.")
    
    # Processar cada arquivo de áudio
    results = []
    detailed_results = []
    
    for audio_file in audio_files:
        result, detailed_result = process_audio_file(audio_file, output_dir, models, processors, options)
        if result:
            results.append(result)
            detailed_results.append(detailed_result)
    
    # Resumo dos resultados
    if results:
        results_df = pd.DataFrame(results)
        logger.info("\nResultados:")
        logger.info(results_df)
        
        # Salvar resultados
        results_df.to_csv(os.path.join(output_dir, "resultados_comparacao.csv"), index=False)
        
        # Salvar resultados detalhados
        with open(os.path.join(output_dir, "resultados_detalhados.json"), 'w') as f:
            json.dump(detailed_results, f, indent=4)
        
        # Criar DataFrame mais estruturado para métricas
        metrics_df = pd.DataFrame({
            'Arquivo': [r['audio'] for r in results],
            'PyAnnote_Tempo(s)': [r['pyannote_time'] for r in results],
            'SORTFormer_Tempo(s)': [r['sortformer_time'] for r in results],
            'Razão_Velocidade': [r['speed_ratio'] for r in results],
            'DER': [r['DER'] for r in results],
            'WER': [r['WER'] for r in results],
            'CER': [r['CER'] for r in results]
        })
        
        # Salvar CSV com todas as métricas
        metrics_df.to_csv(os.path.join(output_dir, "metricas_completas.csv"), index=False)
        
        # Plotar comparação de tempo
        plot_time_comparison(results_df, output_dir)
        
        # Plotar métricas WER e CER
        plot_metrics_comparison(results, output_dir)
        
        # Imprimir estatísticas resumidas
        print_summary_statistics(results)
        
        logger.info(f"\nResultados salvos em {output_dir}")
    else:
        logger.warning("Nenhum resultado foi produzido para análise.")

if __name__ == "__main__":
    main()