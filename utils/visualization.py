"""
Funções para visualização dos resultados de diarização
"""

import matplotlib.pyplot as plt
import seaborn as sns
import torchaudio
import os
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def visualize_diarization(audio_path, pyannote_output, sortformer_output, output_path):
    """
    Salva os resultados de diarização em um arquivo sem visualização.
    
    Args:
        audio_path: Caminho do arquivo de áudio
        pyannote_output: Saída do modelo PyAnnote
        sortformer_output: Saída do modelo SORTFormer convertida para formato PyAnnote
        output_path: Caminho onde salvar a visualização
        
    Returns:
        bool: True se bem-sucedido, False caso contrário
    """
    try:
        
        audio_basename = os.path.splitext(os.path.basename(audio_path))[0]

        clean_output_path = output_path.replace(os.path.basename(audio_path), f"{audio_basename}-comparison.txt")
        

        with open(clean_output_path, 'w') as f:
            f.write(f"Resultados de diarização para: {audio_basename}\n\n")
            
            f.write("PyAnnote Diarization:\n")
            for segment, _, speaker in pyannote_output.itertracks(yield_label=True):
                f.write(f"{speaker}: {segment.start:.2f} - {segment.end:.2f}\n")
            
            f.write("\nSORTFormer Diarization (NeMo):\n")
            for segment, _, speaker in sortformer_output.itertracks(yield_label=True):
                f.write(f"{speaker}: {segment.start:.2f} - {segment.end:.2f}\n")
        
        return True
    except Exception as e:
        logger.error(f"Erro ao salvar resultados de diarização: {e}")
        return False

def plot_time_comparison(results_df, output_dir):
    """
    Plota uma comparação do tempo de processamento entre os modelos
    
    Args:
        results_df: DataFrame com os resultados
        output_dir: Diretório onde salvar o gráfico
        
    Returns:
        str: Caminho do arquivo de imagem salvo
    """
    try:
        plt.figure(figsize=(10, 6))
        sns.barplot(data=results_df.melt(id_vars='audio', 
                                      value_vars=['pyannote_time', 'sortformer_time'],
                                      var_name='modelo', value_name='tempo (s)'))
        plt.title('Comparação de Tempo de Processamento')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        output_path = os.path.join(output_dir, "comparacao_tempo.png")
        plt.savefig(output_path)
        plt.close()
        
        return output_path
    except Exception as e:
        logger.error(f"Erro ao criar gráfico de tempo: {e}")
        return None

def plot_metrics_comparison(results, output_dir):
    """
    Plota uma comparação das métricas WER e CER
    
    Args:
        results: Lista de dicionários com os resultados
        output_dir: Diretório onde salvar o gráfico
        
    Returns:
        str: Caminho do arquivo de imagem salvo ou None
    """
    wer_values = [r['WER'] for r in results if r['WER'] != 'N/A' and r['WER'] is not None]
    cer_values = [r['CER'] for r in results if r['CER'] != 'N/A' and r['CER'] is not None]
    
    if not wer_values or not cer_values:
        logger.warning("Dados insuficientes para plotar métricas WER/CER")
        return None
    
    try:
        plt.figure(figsize=(12, 6))
        metrics_plot = pd.DataFrame({
            'Arquivo': [os.path.splitext(r['audio'])[0] for r in results if r['WER'] != 'N/A' and r['WER'] is not None],
            'WER (%)': [float(r['WER'])*100 for r in results if r['WER'] != 'N/A' and r['WER'] is not None],
            'CER (%)': [float(r['CER'])*100 for r in results if r['CER'] != 'N/A' and r['CER'] is not None]
        })
        
        if metrics_plot.empty:
            logger.warning("DataFrame vazio para plotar métricas WER/CER")
            return None
            
        metrics_plot = metrics_plot.melt(id_vars='Arquivo', 
                                      value_vars=['WER (%)', 'CER (%)'],
                                      var_name='Métrica', value_name='Valor (%)')
        sns.barplot(data=metrics_plot, x='Arquivo', y='Valor (%)', hue='Métrica')
        plt.title('Comparação de WER e CER')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        output_path = os.path.join(output_dir, "comparacao_wer_cer.png")
        plt.savefig(output_path)
        plt.close()
        
        return output_path
    except Exception as e:
        logger.error(f"Erro ao criar gráfico de WER/CER: {e}")
        return None

def print_summary_statistics(results):
    """
    Imprime estatísticas resumidas dos resultados e retorna um dicionário com os valores médios.
    
    Args:
        results: Lista de dicionários com os resultados

    Returns:
        dict: Estatísticas médias dos tempos, DER, WER e CER
    """
    try:
        summary = {}

        pyannote_times = [r['pyannote_time'] for r in results if isinstance(r['pyannote_time'], (int, float))]
        sortformer_times = [r['sortformer_time'] for r in results if isinstance(r['sortformer_time'], (int, float))]
        
        if pyannote_times:
            summary["tempo_medio_pyannote"] = np.mean(pyannote_times)
            logger.info(f"Tempo médio PyAnnote: {summary['tempo_medio_pyannote']:.2f}s")
        
        if sortformer_times:
            summary["tempo_medio_sortformer"] = np.mean(sortformer_times)
            logger.info(f"Tempo médio SORTFormer: {summary['tempo_medio_sortformer']:.2f}s")
            
        # DER médio se disponível
        der_values = [r['DER'] for r in results if r['DER'] != 'N/A' and r['DER'] is not None]
        if der_values:
            summary["der_medio"] = np.mean([float(d) for d in der_values])
            logger.info(f"DER médio: {summary['der_medio']:.4f}")
            
        # WER e CER médios se disponíveis
        wer_values = [r['WER'] for r in results if r['WER'] != 'N/A' and r['WER'] is not None]
        cer_values = [r['CER'] for r in results if r['CER'] != 'N/A' and r['CER'] is not None]
        
        if wer_values:
            summary["wer_medio"] = np.mean([float(w) for w in wer_values])
            logger.info(f"WER médio: {summary['wer_medio']:.4f}")
        
        if cer_values:
            summary["cer_medio"] = np.mean([float(c) for c in cer_values])
            logger.info(f"CER médio: {summary['cer_medio']:.4f}")
        
        return summary

    except Exception as e:
        logger.error(f"Erro ao calcular estatísticas: {e}")
        return None
