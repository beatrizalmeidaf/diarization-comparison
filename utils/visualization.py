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
        clean_output_path = os.path.join(os.path.dirname(output_path), f"{audio_basename}-comparison.txt")
        
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
    Plota uma comparação do tempo de processamento entre os modelos usando gráfico de linha
    
    Args:
        results_df: DataFrame com os resultados
        output_dir: Diretório onde salvar o gráfico
        
    Returns:
        str: Caminho do arquivo de imagem salvo
    """
    try:
        plt.figure(figsize=(12, 6))
        
        # Preparando os dados para o gráfico de linha
        data_melted = results_df.melt(id_vars='audio', 
                                      value_vars=['pyannote_time', 'sortformer_time'],
                                      var_name='modelo', value_name='tempo (s)')
        
        data_melted['modelo'] = data_melted['modelo'].map({
            'pyannote_time': 'PyAnnote', 
            'sortformer_time': 'SORTFormer'
        })
        
        # Extrair apenas o nome do arquivo sem caminho e extensão
        data_melted['audio'] = data_melted['audio'].apply(
            lambda x: os.path.splitext(os.path.basename(x))[0]
        )
        
        # Ordenar por nome do arquivo para melhor visualização
        data_melted = data_melted.sort_values('audio')
        
        # Plotar gráfico de linha
        sns.lineplot(data=data_melted, x='audio', y='tempo (s)', hue='modelo', 
                    markers=True, dashes=False, style='modelo')
        
        plt.title('Comparação de Tempo de Processamento')
        plt.xlabel('Arquivo de Áudio')
        plt.ylabel('Tempo (segundos)')
        plt.xticks(rotation=45)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        output_path = os.path.join(output_dir, "comparacao_tempo.png")
        plt.savefig(output_path)
        plt.close()
        
        return output_path
    except Exception as e:
        logger.error(f"Erro ao criar gráfico de tempo: {e}")
        return None

def plot_diarization_metrics(results, output_dir):
    """
    Plota uma comparação das métricas DER e JER entre PyAnnote e SORTFormer
    
    Args:
        results: Lista de dicionários com os resultados
        output_dir: Diretório onde salvar o gráfico
        
    Returns:
        str: Caminho do arquivo de imagem salvo ou None
    """
    try:
        # Verificar se temos métricas DER para ambos os modelos
        pyannote_der = [r.get('pyannote_der') for r in results 
                        if r.get('pyannote_der') is not None and r.get('pyannote_der') != 'N/A']
        sortformer_der = [r.get('sortformer_der') for r in results 
                          if r.get('sortformer_der') is not None and r.get('sortformer_der') != 'N/A']
        
        # Verificar se temos métricas JER para ambos os modelos
        pyannote_jer = [r.get('pyannote_jer') for r in results 
                        if r.get('pyannote_jer') is not None and r.get('pyannote_jer') != 'N/A']
        sortformer_jer = [r.get('sortformer_jer') for r in results 
                          if r.get('sortformer_jer') is not None and r.get('sortformer_jer') != 'N/A']
        
        if not (pyannote_der and sortformer_der) and not (pyannote_jer and sortformer_jer):
            logger.warning("Dados insuficientes para plotar métricas DER/JER")
            return None
        
        # Criar dataframe para plotagem
        files = [os.path.splitext(os.path.basename(r['audio']))[0] for r in results]
        
        metrics_data = []
        
        # Adicionar dados DER se disponíveis
        for i, r in enumerate(results):
            if r.get('pyannote_der') is not None and r.get('pyannote_der') != 'N/A':
                metrics_data.append({
                    'Arquivo': files[i],
                    'Métrica': 'DER',
                    'Modelo': 'PyAnnote',
                    'Valor (%)': float(r['pyannote_der']) * 100
                })
                
            if r.get('sortformer_der') is not None and r.get('sortformer_der') != 'N/A':
                metrics_data.append({
                    'Arquivo': files[i],
                    'Métrica': 'DER',
                    'Modelo': 'SORTFormer',
                    'Valor (%)': float(r['sortformer_der']) * 100
                })
                
            # Adicionar dados JER se disponíveis    
            if r.get('pyannote_jer') is not None and r.get('pyannote_jer') != 'N/A':
                metrics_data.append({
                    'Arquivo': files[i],
                    'Métrica': 'JER',
                    'Modelo': 'PyAnnote',
                    'Valor (%)': float(r['pyannote_jer']) * 100
                })
                
            if r.get('sortformer_jer') is not None and r.get('sortformer_jer') != 'N/A':
                metrics_data.append({
                    'Arquivo': files[i],
                    'Métrica': 'JER',
                    'Modelo': 'SORTFormer',
                    'Valor (%)': float(r['sortformer_jer']) * 100
                })
        
        metrics_df = pd.DataFrame(metrics_data)
        
        if metrics_df.empty:
            logger.warning("DataFrame vazio para plotar métricas DER/JER")
            return None
        
        # Plotar gráfico de barras agrupadas para DER
        plt.figure(figsize=(14, 8))
        
        # Gráfico para DER
        der_df = metrics_df[metrics_df['Métrica'] == 'DER']
        if not der_df.empty:
            plt.subplot(2, 1, 1)
            sns.barplot(data=der_df, x='Arquivo', y='Valor (%)', hue='Modelo')
            plt.title('Comparação de DER (Diarization Error Rate)')
            plt.ylabel('DER (%)')
            plt.xticks(rotation=45)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend(title='Modelo')
        
        # Gráfico para JER
        jer_df = metrics_df[metrics_df['Métrica'] == 'JER']
        if not jer_df.empty:
            plt.subplot(2, 1, 2)
            sns.barplot(data=jer_df, x='Arquivo', y='Valor (%)', hue='Modelo')
            plt.title('Comparação de JER (Jaccard Error Rate)')
            plt.ylabel('JER (%)')
            plt.xticks(rotation=45)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend(title='Modelo')
        
        plt.tight_layout()
        
        output_path = os.path.join(output_dir, "comparacao_der_jer.png")
        plt.savefig(output_path)
        plt.close()
        
        return output_path
    except Exception as e:
        logger.error(f"Erro ao criar gráfico de métricas DER/JER: {e}")
        return None

def print_summary_statistics(results):
    """
    Imprime estatísticas resumidas dos resultados e retorna um dicionário com os valores médios.
    
    Args:
        results: Lista de dicionários com os resultados

    Returns:
        dict: Estatísticas médias dos tempos, DER e JER para ambos os modelos
    """
    try:
        summary = {}

        # Tempos de processamento
        pyannote_times = [r['pyannote_time'] for r in results if isinstance(r['pyannote_time'], (int, float))]
        sortformer_times = [r['sortformer_time'] for r in results if isinstance(r['sortformer_time'], (int, float))]
        
        if pyannote_times:
            summary["tempo_medio_pyannote"] = np.mean(pyannote_times)
            logger.info(f"Tempo médio PyAnnote: {summary['tempo_medio_pyannote']:.2f}s")
        
        if sortformer_times:
            summary["tempo_medio_sortformer"] = np.mean(sortformer_times)
            logger.info(f"Tempo médio SORTFormer: {summary['tempo_medio_sortformer']:.2f}s")
            
        # DER para cada ferramenta
        pyannote_der = [float(r['pyannote_der']) for r in results 
                         if r.get('pyannote_der') not in ('N/A', None)]
        sortformer_der = [float(r['sortformer_der']) for r in results 
                           if r.get('sortformer_der') not in ('N/A', None)]
        
        if pyannote_der:
            summary["der_medio_pyannote"] = np.mean(pyannote_der)
            logger.info(f"DER médio PyAnnote: {summary['der_medio_pyannote']:.4f}")
        
        if sortformer_der:
            summary["der_medio_sortformer"] = np.mean(sortformer_der)
            logger.info(f"DER médio SORTFormer: {summary['der_medio_sortformer']:.4f}")
            
        # JER para cada ferramenta
        pyannote_jer = [float(r['pyannote_jer']) for r in results 
                         if r.get('pyannote_jer') not in ('N/A', None)]
        sortformer_jer = [float(r['sortformer_jer']) for r in results 
                           if r.get('sortformer_jer') not in ('N/A', None)]
        
        if pyannote_jer:
            summary["jer_medio_pyannote"] = np.mean(pyannote_jer)
            logger.info(f"JER médio PyAnnote: {summary['jer_medio_pyannote']:.4f}")
        
        if sortformer_jer:
            summary["jer_medio_sortformer"] = np.mean(sortformer_jer)
            logger.info(f"JER médio SORTFormer: {summary['jer_medio_sortformer']:.4f}")
        
        # Se tiver métricas para ambos os modelos, calcular qual é melhor
        if pyannote_der and sortformer_der:
            melhor_der = "PyAnnote" if summary["der_medio_pyannote"] < summary["der_medio_sortformer"] else "SORTFormer"
            logger.info(f"Modelo com menor DER médio: {melhor_der}")
            summary["melhor_modelo_der"] = melhor_der
            
        if pyannote_jer and sortformer_jer:
            melhor_jer = "PyAnnote" if summary["jer_medio_pyannote"] < summary["jer_medio_sortformer"] else "SORTFormer"
            logger.info(f"Modelo com menor JER médio: {melhor_jer}")
            summary["melhor_modelo_jer"] = melhor_jer
        
        return summary

    except Exception as e:
        logger.error(f"Erro ao calcular estatísticas: {e}")
        return None
