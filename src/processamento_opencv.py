from pathlib import Path

import cv2
import numpy as np


def processar_imagem(
    caminho_imagem: str | Path,
    limiar_inferior: int = 50,
    limiar_superior: int = 150,
) -> dict:
    """
    Executa o pipeline clássico de processamento em uma imagem.

    Etapas:
    1. carregamento;
    2. conversão para escala de cinza;
    3. suavização com Gaussian Blur;
    4. limiarização de Otsu;
    5. detecção de bordas com Canny;
    6. dilatação e erosão;
    7. sobreposição dos contornos.
    """
    caminho_imagem = Path(caminho_imagem)

    imagem_bgr = cv2.imread(str(caminho_imagem))

    if imagem_bgr is None:
        raise FileNotFoundError(
            f"Não foi possível carregar a imagem: {caminho_imagem}"
        )

    # Prepara as representações usadas durante o processamento.
    imagem_rgb = cv2.cvtColor(imagem_bgr, cv2.COLOR_BGR2RGB)
    imagem_cinza = cv2.cvtColor(imagem_bgr, cv2.COLOR_BGR2GRAY)
    imagem_gaussian = cv2.GaussianBlur(imagem_cinza, (5, 5), 0)

    valor_limiar, imagem_limiarizada = cv2.threshold(
        imagem_gaussian,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )

    bordas_canny = cv2.Canny(
        imagem_gaussian,
        limiar_inferior,
        limiar_superior,
    )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (3, 3),
    )

    bordas_dilatadas = cv2.dilate(
        bordas_canny,
        kernel,
        iterations=1,
    )

    bordas_fechamento = cv2.erode(
        bordas_dilatadas,
        kernel,
        iterations=1,
    )

    # Sobrepõe em vermelho os contornos processados.
    imagem_sobreposta = imagem_rgb.copy()
    imagem_sobreposta[bordas_fechamento > 0] = [255, 0, 0]

    return {
        "original": imagem_rgb,
        "cinza": imagem_cinza,
        "gaussian": imagem_gaussian,
        "limiarizada": imagem_limiarizada,
        "valor_limiar": valor_limiar,
        "canny": bordas_canny,
        "dilatada": bordas_dilatadas,
        "fechamento": bordas_fechamento,
        "sobreposta": imagem_sobreposta,
    }