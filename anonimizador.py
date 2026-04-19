import fitz  # PyMuPDF
import re
import easyocr
import numpy as np
from PIL import Image
import time
import os
import argparse
from tqdm import tqdm
from pathlib import Path

class PDFAnonymizer:
    def __init__(self, use_gpu=True):
        print(f"[*] Inicializando Motor OCR (GPU: {use_gpu})...")
        # Aproveitando os 12GB da sua 5070
        self.reader = easyocr.Reader(['pt'], gpu=use_gpu)
        self.cpf_pattern = re.compile(r'\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b')

    def is_page_image_only(self, page):
        """Verifica se a página é essencialmente uma imagem (sem texto selecionável)."""
        text = page.get_text("text").strip()
        # Se houver menos de 10 caracteres, tratamos como imagem para garantir via OCR
        return len(text) < 10 

    def process_file(self, input_path, output_dir):
        input_path = Path(input_path)
        output_path = Path(output_dir) / f"ANON_{input_path.name}"
        
        try:
            doc = fitz.open(str(input_path))
        except Exception as e:
            print(f"\n[!] Erro ao abrir {input_path.name}: {e}")
            return 0
            
        start_time = time.perf_counter()
        
        # Barra de progresso interna para as páginas
        pbar = tqdm(total=len(doc), desc=f"-> {input_path.name[:20]}", leave=False, unit="pág")

        for page_index in range(len(doc)):
            page = doc[page_index]
            found_sensitive = False
            
            # 1. Busca em texto digital (rápido)
            words = page.get_text("words")
            for w in words:
                if self.cpf_pattern.search(w[4]):
                    page.add_redact_annot(w[:4], fill=(0, 0, 0))
                    found_sensitive = True

            # 2. OCR condicional (apenas se for página de imagem)
            if self.is_page_image_only(page):
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                results = self.reader.readtext(np.array(img))

                for (bbox, text, prob) in results:
                    (tl, tr, br, bl) = bbox
                    rect = fitz.Rect(tl[0]/2, tl[1]/2, br[0]/2, br[1]/2)
                    
                    if self.cpf_pattern.search(text):
                        page.add_redact_annot(rect, fill=(0, 0, 0))
                        found_sensitive = True
                    else:
                        # Torna a página pesquisável (Texto invisível)
                        page.insert_text(rect.bl, text, fontsize=8, render_mode=3)

            # Aplica as tarjas permanentemente
            if found_sensitive:
                page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE)
            
            pbar.update(1)

        # Salvamento otimizado (Removido linear=True para evitar erro de versão)
        doc.save(str(output_path), garbage=4, deflate=True, clean=True)
        doc.close()
        pbar.close()
        
        return time.perf_counter() - start_time

def main():
    parser = argparse.ArgumentParser(description="Anonimizador de CPF em lote com OCR via GPU.")
    parser.add_argument("caminho", help="Arquivo PDF ou pasta contendo PDFs")
    args = parser.parse_args()

    input_path = Path(args.caminho)
    output_folder = Path("(anonimizações processadas)")
    output_folder.mkdir(exist_ok=True)

    # Lógica de captura de arquivos (Case-Insensitive)
    files = []
    if input_path.is_file():
        if input_path.suffix.lower() == ".pdf":
            files = [input_path]
    elif input_path.is_dir():
        # Captura .pdf, .PDF, .Pdf, etc.
        files = [f for f in input_path.iterdir() if f.suffix.lower() == ".pdf"]
    else:
        print(f"[ERRO] Caminho inválido: {args.caminho}")
        return

    if not files:
        print(f"[!] Nenhum arquivo PDF localizado em: {args.caminho}")
        return

    print(f"[*] Total de arquivos localizados: {len(files)}")
    anonimizador = PDFAnonymizer(use_gpu=True)
    
    total_start = time.perf_counter()
    
    for f in files:
        t_file = anonimizador.process_file(f, output_folder)
        if t_file > 0:
            print(f"    Concluído: {f.name} ({t_file:.2f}s)")

    total_end = time.perf_counter()
    print("-" * 50)
    print(f"[!] Lote finalizado em {total_end - total_start:.2f} segundos.")
    print(f"[!] Arquivos salvos em: {output_folder.absolute()}")

if __name__ == "__main__":
    main()