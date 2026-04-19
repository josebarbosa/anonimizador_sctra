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
        self.reader = easyocr.Reader(['pt'], gpu=use_gpu)
        self.cpf_pattern = re.compile(r'\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b')

    def is_page_image_only(self, page):
        """Verifica se a página não possui texto extraível (é uma imagem)."""
        text = page.get_text("text").strip()
        return len(text) < 10  # Margem de segurança para ruídos

    def process_file(self, input_path, output_dir):
        input_path = Path(input_path)
        output_path = Path(output_dir) / f"ANON_{input_path.name}"
        
        doc = fitz.open(str(input_path))
        start_time = time.perf_counter()
        
        # Barra de progresso para as páginas do arquivo atual
        pbar = tqdm(total=len(doc), desc=f"-> {input_path.name[:20]}", leave=False)

        for page_index in range(len(doc)):
            page = doc[page_index]
            found_sensitive = False
            
            # 1. Busca em texto digital (sempre tentamos primeiro)
            words = page.get_text("words")
            for w in words:
                if self.cpf_pattern.search(w[4]):
                    page.add_redact_annot(w[:4], fill=(0, 0, 0))
                    found_sensitive = True

            # 2. OCR condicional: apenas se a página for detectada como imagem
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
                        # Torna o PDF pesquisável apenas para páginas que eram imagens
                        page.insert_text(rect.bl, text, fontsize=8, render_mode=3)

            if found_sensitive:
                page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE)
            
            pbar.update(1)

        doc.save(str(output_path), garbage=4, deflate=True, clean=True)
        doc.close()
        pbar.close()
        
        elapsed = time.perf_counter() - start_time
        return elapsed

def main():
    parser = argparse.ArgumentParser(description="Anonimizador de CPF em lote com OCR via GPU.")
    parser.add_argument("caminho", help="Arquivo PDF ou pasta contendo PDFs")
    args = parser.parse_args()

    # Configuração de pastas
    input_path = Path(args.caminho)
    output_folder = Path("(anonimizações processadas)")
    output_folder.mkdir(exist_ok=True)

    # Identifica arquivos para processar
    if input_path.is_file():
        files = [input_path]
    elif input_path.is_dir():
        files = list(input_path.glob("*.pdf"))
    else:
        print(f"[ERRO] Caminho inválido: {args.caminho}")
        return

    if not files:
        print("[!] Nenhum PDF encontrado para processar.")
        return

    print(f"[*] Total de arquivos localizados: {len(files)}")
    anonimizador = PDFAnonymizer(use_gpu=True)
    
    total_start = time.perf_counter()
    
    for f in files:
        t_file = anonimizador.process_file(f, output_folder)
        print(f"    Concluído: {f.name} ({t_file:.2f}s)")

    total_end = time.perf_counter()
    print("-" * 50)
    print(f"[!] Lote finalizado em {total_end - total_start:.2f} segundos.")
    print(f"[!] Arquivos salvos em: {output_folder.absolute()}")

if __name__ == "__main__":
    main()
