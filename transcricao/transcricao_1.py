import speech_recognition as sr
from pydub import AudioSegment
import os

# Defina o caminho completo para o executável ffmpeg
AudioSegment.converter = r"C:\ffmpeg\bin\ffmpeg.exe"
AudioSegment.ffprobe   = r"C:\ffmpeg\bin\ffprobe.exe"


def transcrever_audio(arquivo_audio, idioma="pt-BR"):
    """
    Transcreve um arquivo de áudio para texto.

    :param arquivo_audio: Caminho do arquivo de áudio.
    :param idioma: Idioma do áudio (padrão: pt-BR).
    :return: Texto transcrito.
    """
    try:
        # Convertendo áudio para formato compatível (se necessário)
        if not arquivo_audio.endswith(".wav"):
            audio = AudioSegment.from_file(arquivo_audio)
            arquivo_wav = "temporario.wav"
            audio.export(arquivo_wav, format="wav")
        else:
            arquivo_wav = arquivo_audio

        # Inicializando o reconhecedor
        reconhecedor = sr.Recognizer()
        with sr.AudioFile(arquivo_wav) as source:
            print("Processando o áudio...")
            audio_data = reconhecedor.record(source)

        # Realizando a transcrição
        texto = reconhecedor.recognize_google(audio_data, language=idioma)
        return texto

    except sr.UnknownValueError:
        return "O áudio não pôde ser compreendido."
    except sr.RequestError as e:
        return f"Erro ao acessar o serviço de reconhecimento: {e}"
    except Exception as e:
        return f"Ocorreu um erro: {e}"

# Ajuste do caminho do arquivo
caminho_diretorio = "C:/Users/acsfarias/Desktop/Pós-Ciência de Dados/pos/transcricao/"
nome_arquivo = "URecorder_20241203_150606.mp3"
caminho_audio = os.path.join(caminho_diretorio, nome_arquivo)

# Executando a transcrição
resultado = transcrever_audio(caminho_audio, idioma="pt-BR")
print("\nTexto transcrito:")
print(resultado)
