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
    plot_diarization_metrics,
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
        "pyannote_der": "N/A",
        "sortformer_der": "N/A",
        "pyannote_jer": "N/A",
        "sortformer_jer": "N/A",
        "speed_ratio": None
    }
    
    detailed_result = {
        "audio": audio_file,
        "pyannote": {"time": None, "der": "N/A", "jer": "N/A", "error": None},
        "sortformer": {"time": None, "der": "N/A", "jer": "N/A", "error": None},
        "comparison": {"speed_ratio": None}
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
        # Calcular DER e JER para PyAnnote (assumindo referência externa ou entre si)
        pyannote_metrics = EvaluationProcessor.evaluate_diarization_performance(pyannote_output)
        result["pyannote_der"] = pyannote_metrics.get("DER", "N/A")
        result["pyannote_jer"] = pyannote_metrics.get("JER", "N/A")
        detailed_result["pyannote"]["der"] = pyannote_metrics.get("DER", "N/A")
        detailed_result["pyannote"]["jer"] = pyannote_metrics.get("JER", "N/A")
        
        # Calcular DER e JER para SORTFormer
        sortformer_metrics = EvaluationProcessor.evaluate_diarization_performance(sortformer_output)
        result["sortformer_der"] = sortformer_metrics.get("DER", "N/A")
        result["sortformer_jer"] = sortformer_metrics.get("JER", "N/A")
        detailed_result["sortformer"]["der"] = sortformer_metrics.get("DER", "N/A")
        detailed_result["sortformer"]["jer"] = sortformer_metrics.get("JER", "N/A")
        
        logger.info(f"PyAnnote DER: {result['pyannote_der']}, JER: {result['pyannote_jer']}")
        logger.info(f"SORTFormer DER: {result['sortformer_der']}, JER: {result['sortformer_jer']}")
        
        # Visualizar resultados
        if not options.get("skip_visualization", False):
            output_image = os.path.join(output_dir, f"{os.path.basename(audio_file)}_comparison.png")
            visualize_diarization(audio_file, pyannote_output, sortformer_output, output_image)
            logger.info(f"Visualização salva em {output_image}")
    
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
            'Arquivo': [os.path.basename(r['audio']) for r in results],
            'PyAnnote_Tempo(s)': [r['pyannote_time'] for r in results],
            'SORTFormer_Tempo(s)': [r['sortformer_time'] for r in results],
            'Razão_Velocidade': [r['speed_ratio'] for r in results],
            'PyAnnote_DER': [r['pyannote_der'] for r in results],
            'SORTFormer_DER': [r['sortformer_der'] for r in results],
            'PyAnnote_JER': [r['pyannote_jer'] for r in results],
            'SORTFormer_JER': [r['sortformer_jer'] for r in results]
        })
        
        # Salvar CSV com todas as métricas
        metrics_df.to_csv(os.path.join(output_dir, "metricas_completas.csv"), index=False)
        
        # Plotar comparação de tempo
        time_plot_path = plot_time_comparison(results_df, output_dir)
        logger.info(f"Gráfico de comparação de tempo salvo em: {time_plot_path}")
        
        # Plotar métricas de diarização (DER e JER)
        metrics_plot_path = plot_diarization_metrics(results, output_dir)
        logger.info(f"Gráfico de métricas de diarização salvo em: {metrics_plot_path}")
        
        # Imprimir estatísticas resumidas
        summary = print_summary_statistics(results)
        
        # Salvar também o sumário em JSON
        with open(os.path.join(output_dir, "resumo_estatisticas.json"), 'w') as f:
            json.dump(summary, f, indent=4)
        
        logger.info(f"\nResultados salvos em {output_dir}")
    else:
        logger.warning("Nenhum resultado foi produzido para análise.")

if __name__ == "__main__":
    main()
