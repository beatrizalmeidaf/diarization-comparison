# Diarization Comparison: PyAnnote vs. SortFormer (NVIDIA)

Esse repositório tem como objetivo comparar a performance de duas abordagens para diarização de áudio: **PyAnnote** e **SortFormer (NVIDIA)**. A análise inclui a execução dos modelos, extração de métricas e comparação dos resultados.

---

## Configuração do Ambiente

### Clonando o Repositório
Para iniciar o projeto, clone esse repositório e acesse a pasta:

```bash
git clone https://github.com/beatrizalmeidaf/diarization-comparison.git
cd diarization-comparison
```

### Criando e Ativando um Ambiente Virtual
Crie e ative um ambiente virtual para garantir a instalação isolada das dependências:

```bash
python -m venv diar_env
source diar_env/bin/activate  # No Windows: diar_env\Scripts\activate
```

### Instalando Dependências
Instale todas as dependências necessárias executando:

```bash
pip install -r requirements.txt
```

---

## Executando a Diarização
Para executar a diarização de áudio, utilize o comando:

```bash
python main.py
```

---

## Adicionando Arquivos de Áudio
Os arquivos de áudio a serem analisados devem ser inseridos na pasta `audios/`.

---

## Análise de Resultados

Os resultados da diarização são armazenados em logs e podem ser analisados através das métricas implementadas no código. A comparação entre os modelos inclui:

- **Precisão da diarização**: Avaliação da qualidade da separação dos falantes.  
- **Tempo de execução**: Mede a velocidade do processamento da diarização.  
- **Eficiência na separação de falantes**: Avalia o quão bem os falantes são identificados corretamente.  

### Métricas de Avaliação

- **DER (Diarization Error Rate)**: Mede o erro na diarização considerando falantes incorretamente atribuídos, falantes ausentes e falsas detecções. Quanto menor o DER, melhor a diarização.  
- **JER (Jaccard Error Rate)**: Avalia o erro na identificação de segmentos falados, comparando com a referência. Penaliza sobreposição e atribuições erradas. Quanto menor, melhor.  



