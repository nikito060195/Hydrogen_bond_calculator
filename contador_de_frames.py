import numpy as np


# Commented out IPython magic to ensure Python compatibility.
# %cd /content/drive/MyDrive/Membrana\ Gox

"""# Processamento dos dados"""

# Função para processar um frame
def processar_frame(frame):
    # Separar as linhas do frame
    linhas = frame.strip().split('\n')

    # Obter o número guardado na segunda linha
    numero_guardado = int(linhas[0].split()[-1])  # Extrair o último elemento da segunda linha

    # Filtrar linhas cuja primeira coluna é "O"
    linhas_filtradas = [linha.split() for linha in linhas[2:] if linha.split()[0] == "3"\
                        or linha.split()[0] == "4"]
    # Criar o array resultante com rótulos
    resultado = [
        {"ID": linha[0], "Timestep": numero_guardado, "x": float(linha[1]), \
         "y": float(linha[2]), "z": float(linha[3])}
        for i, linha in enumerate(linhas_filtradas)
    ]

    return resultado

# Função para ler o arquivo e criar o array de dados
def ler_arquivo(nome_arquivo):
    with open(nome_arquivo, 'r') as arquivo:
        conteudo = arquivo.read()

    # Separar os frames
    frames = conteudo.split('Atoms. Timestep: ')[1:]

    # Processar cada frame e armazenar os resultados em um array
    dados = []

    for i,frame in enumerate(frames):
      if i % 5 == 0:
            # Processar o frame apenas se a condição for atendida
            resultado_processamento = processar_frame(frame)
            dados.append(resultado_processamento)

    return dados

nome_arquivo = 'traj3_nvt.xyz'
dados = ler_arquivo(nome_arquivo)

print(len(dados[0]))

"""# Escreve novo arquivo com limite de frames"""

with open('traj3_filtrado.xyz', 'w') as arquivo_saida:
    for i, dados_frame in enumerate(dados):
     #   if dados_frame:  # Verificar se há dados para o frame
            arquivo_saida.write(f"{len(dados[i])}\n")
            arquivo_saida.write(f"Atoms. Timestep: {i}\n")
            for linha in dados[i]:
                arquivo_saida.write(f"{linha['ID']} \
                {linha['x']} {linha['y']} {linha['z']}\n")

