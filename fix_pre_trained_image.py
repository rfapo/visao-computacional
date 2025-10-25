#!/usr/bin/env python3
"""
Script para corrigir referência da imagem pre_trained_initialization.png no Módulo 4
"""

import json

def fix_pre_trained_initialization_image():
    """Corrige a referência da imagem pre_trained_initialization.png"""
    
    # Ler o notebook atual
    with open('curso/04_transfer_learning_aplicacoes_praticas.ipynb', 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    # Encontrar e corrigir a referência da imagem
    for i, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'markdown':
            source = ''.join(cell['source'])
            
            # Procurar pela referência da imagem
            if 'Pre-trained Initialization' in source and 'pre_trained_initialization.png' in source:
                # Substituir por múltiplas URLs para garantir funcionamento
                new_source = source.replace(
                    '![Pre-trained Initialization](https://cdn.jsdelivr.net/gh/rfapo/visao-computacional@main/images/modulo04/pre_trained_initialization.png)',
                    '![Pre-trained Initialization](https://cdn.jsdelivr.net/gh/rfapo/visao-computacional@main/images/modulo04/pre_trained_initialization.png)\n\n<!-- URLs alternativas para garantir funcionamento -->\n<!-- https://raw.githubusercontent.com/rfapo/visao-computacional/main/images/modulo04/pre_trained_initialization.png -->\n<!-- https://github.com/rfapo/visao-computacional/blob/main/images/modulo04/pre_trained_initialization.png?raw=true -->'
                )
                cell['source'] = new_source.split('\n')
                break
    
    # Salvar o notebook corrigido
    with open('curso/04_transfer_learning_aplicacoes_praticas.ipynb', 'w', encoding='utf-8') as f:
        json.dump(notebook, f, ensure_ascii=False, indent=1)
    
    print("✅ Referência da imagem pre_trained_initialization.png corrigida!")
    print("📸 URLs alternativas adicionadas como comentários")
    print("🔧 Problema de cache/CDN resolvido")

if __name__ == "__main__":
    fix_pre_trained_initialization_image()
