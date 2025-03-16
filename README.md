# Diarization Comparison: PyAnote vs. SortFormer (NVIDIA)

Esse repositório compara a performance de duas abordagens para diarização de áudio: **PyAnote** e **SortFormer (NVIDIA)**.

## 📌 Clonando o Repositório
Para começar, clone esse repositório usando o comando:
```bash
git clone https://github.com/beatrizalmeidaf/diarization-comparison.git
cd diarization-comparison
```

## 📌 Instalação e Configuração

Para rodar o projeto, siga os passos abaixo:

### 1️⃣ Criar e ativar ambiente virtual
```bash
python -m venv diar_env
source diar_env/bin/activate  # No Windows: diar_env\Scripts\activate
```

### 2️⃣ Instalar dependências
```bash
pip install -r requirements.txt
```

## ▶️ Execução
Para rodar a diarização, basta executar:
```bash
python main.py
```

## 🎵 Adicionando Áudios
Os arquivos de áudio devem ser adicionados à pasta `audios/`.

Depois, edite o arquivo `config/settings.py`, adicionando os caminhos dos áudios na variável `AUDIO_FILES`. Exemplo:
```python
AUDIO_FILES = [
    "audios/audio1.wav",
    "audios/audio2.wav"
]
```

## 📊 Resultados e Comparação
Os resultados da diarização podem ser analisados e comparados através dos logs gerados e métricas específicas implementadas no código.



