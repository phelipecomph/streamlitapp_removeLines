import streamlit as st
import cv2
import math
import numpy as np
import json
from PIL import Image, ImageDraw

# Função para desenhar linhas na imagem
def draw_lines(image, lines, color, thickness):
    img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    for line in lines:
        x1, y1, x2, y2 = line
        draw.line((x1, y1, x2, y2), fill=color, width=thickness)
    return img

# Função para desenhar uma única linha na imagem
def draw_single_line(image, line, color, thickness):
    img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    x1, y1, x2, y2 = line
    draw.line((x1, y1, x2, y2), fill=color, width=thickness)
    return img

def filter_horizontal_lines(lines, threshold=1.0):
    """
    Filtra as linhas mais horizontais do que verticais.
    
    :param lines: Lista de linhas, cada linha representada por [x1, y1, x2, y2].
    :param threshold: O valor limite da inclinação para considerar a linha como horizontal.
                      Por exemplo, threshold=1.0 significa que consideraremos horizontal
                      se a inclinação estiver entre -1 e 1.
    :return: Lista de linhas filtradas.
    """
    horizontal_lines = []
    for line in lines:
        x1, y1, x2, y2 = line
        if x2 != x1:  # Evita divisão por zero
            slope = (y2 - y1) / (x2 - x1)
            if abs(slope) <= threshold:
                horizontal_lines.append(line)
    return horizontal_lines

# Carregar a imagem
uploaded_file = st.file_uploader("Escolha uma imagem...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = np.array(Image.open(uploaded_file))
    imagem_cinza = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Aplicar limiarização de Otsu
    _, imagem_binaria = cv2.threshold(imagem_cinza, 0.99, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Aplicar detecção de bordas para realçar as linhas
    bordas = cv2.Canny(imagem_binaria, 50, 150, apertureSize=3)
    image = imagem_binaria.copy()
    
    # Carregar linhas
    if 'lines' not in st.session_state:
        st.session_state['lines'] = [list(line[0]) for line in cv2.HoughLinesP(bordas, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)]
    lines = st.session_state['lines']
    lines = filter_horizontal_lines(lines)

    # Checkbox para mostrar todas as linhas
    show_all_lines = st.checkbox('Mostrar todas as linhas', value=False)

    # Checkbox para alterar a cor das linhas
    use_red_color = st.sidebar.checkbox('Usar cor vermelha para linhas', value=True)
    line_color = (255, 0, 0) if use_red_color else (255, 255, 255)

    # Slider para alterar a espessura das linhas
    line_thickness = st.sidebar.slider('Espessura das linhas', min_value=1, max_value=20, value=5)

    # Selecionar linha
    if 'selected_line_idx' not in st.session_state:
        st.session_state['selected_line_idx'] = 0

    selected_line_idx = st.sidebar.selectbox(
        'Selecionar linha', 
        list(range(len(lines))), 
        format_func=lambda x: f"Linha {x+1}",
        index=st.session_state['selected_line_idx']
    )
    current_line = lines[selected_line_idx]

    # Inputs para editar a linha atual
    x1 = st.sidebar.number_input('x1', value=current_line[0], key='x1')
    y1 = st.sidebar.number_input('y1', value=current_line[1], key='y1')
    x2 = st.sidebar.number_input('x2', value=current_line[2], key='x2')
    y2 = st.sidebar.number_input('y2', value=current_line[3], key='y2')

    # Atualizar linha na lista
    lines[selected_line_idx] = [x1, y1, x2, y2]
    st.session_state['lines'] = lines

    # Desenhar a imagem com a linha(s)
    if show_all_lines:
        img = draw_lines(image, lines, line_color, line_thickness)
    else:
        img = draw_single_line(image, lines[selected_line_idx], line_color, line_thickness)
    
    st.image(img, caption='Imagem com linha(s)', use_column_width=True)

    # Botão para remover a linha atual
    if st.sidebar.button('Remover linha'):
        lines.pop(selected_line_idx)
        st.session_state['lines'] = lines
        if selected_line_idx >= len(lines):
            selected_line_idx = len(lines) - 1
        st.session_state['selected_line_idx'] = selected_line_idx
        st.experimental_rerun()

    # Botão para adicionar uma nova linha
    if st.sidebar.button('Adicionar linha'):
        lines.append([0, 0, 0, 0])
        st.session_state['lines'] = lines
        st.session_state['selected_line_idx'] = len(lines) - 1
        st.experimental_rerun()

    # Botão para salvar a imagem final e as linhas em JSON
    if st.sidebar.button('Salvar linhas e imagem'):
        final_img = draw_lines(image, lines, line_color, line_thickness)
        final_img.save('imagem_final.png')
        with open('linhas_final.json', 'w') as f:
            f.write(str(lines))
        st.sidebar.write("Imagem e linhas salvas!")
