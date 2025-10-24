#!/usr/bin/env python3
"""
Script para atualizar referências de imagens nos notebooks para URLs do GitHub
"""

import json
import os
import re
from pathlib import Path

def update_notebook_images(notebook_path):
    """Atualiza as referências de imagens em um notebook"""
    
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    updated = False
    
    for cell in notebook['cells']:
        if cell['cell_type'] == 'markdown':
            source = cell['source']
            if isinstance(source, list):
                source = ''.join(source)
            
            # Padrão para encontrar referências de imagens
            pattern = r'!\[([^\]]*)\]\(images/([^)]+)\)'
            
            def replace_image_url(match):
                alt_text = match.group(1)
                image_path = match.group(2)
                
                # Construir URL do GitHub
                github_url = f"https://raw.githubusercontent.com/rfapo/visao-computacional/main/images/{image_path}"
                return f"![{alt_text}]({github_url})"
            
            new_source = re.sub(pattern, replace_image_url, source)
            
            if new_source != source:
                cell['source'] = new_source.split('\n')
                updated = True
    
    if updated:
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, indent=2, ensure_ascii=False)
        print(f"✅ Atualizado: {notebook_path}")
    else:
        print(f"ℹ️  Nenhuma alteração necessária: {notebook_path}")

def main():
    """Função principal"""
    
    print("🔄 Atualizando referências de imagens nos notebooks...")
    
    # Lista de notebooks
    notebooks = [
        "01_introducao_historia_visao_computacional.ipynb",
        "02_processamento_digital_imagem_fundamentos.ipynb",
        "03_deep_learning_visao_computacional.ipynb",
        "04_transfer_learning_aplicacoes_praticas.ipynb",
        "05_tarefas_fundamentais_visao_computacional.ipynb",
        "06_ocr_reconhecimento_texto.ipynb",
        "07_gans_vaes_geracao_sintetica.ipynb",
        "08_vision_transformers_atencao.ipynb",
        "09_foundation_models_visao_computacional.ipynb",
        "10_atividade_final_pratica.ipynb"
    ]
    
    for notebook in notebooks:
        if os.path.exists(notebook):
            update_notebook_images(notebook)
        else:
            print(f"❌ Arquivo não encontrado: {notebook}")
    
    print("\n🎉 Atualização concluída!")

if __name__ == "__main__":
    main()
