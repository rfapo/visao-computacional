#!/usr/bin/env python3
"""
Script para mover a seção 1.5 Demonstração Prática: Técnicas de Detecção de Bordas
do Módulo 1 para o Módulo 2
"""

import json
import os

def move_edge_detection_section():
    """Move a seção de detecção de bordas do Módulo 1 para o Módulo 2"""
    
    print("🔄 MOVENDO SEÇÃO DE DETECÇÃO DE BORDAS DO MÓDULO 1 PARA O MÓDULO 2")
    print("=" * 70)
    
    # Ler Módulo 1
    with open('curso/01_introducao_historia_visao_computacional.ipynb', 'r', encoding='utf-8') as f:
        module1 = json.load(f)
    
    # Ler Módulo 2
    with open('curso/02_processamento_digital_imagem_fundamentos.ipynb', 'r', encoding='utf-8') as f:
        module2 = json.load(f)
    
    # Encontrar e extrair a seção 1.5 do Módulo 1
    cells_to_move = []
    cells_to_remove = []
    
    for i, cell in enumerate(module1['cells']):
        if cell['cell_type'] == 'markdown':
            source = cell['source']
            if isinstance(source, list) and len(source) > 0:
                # Verificar se é a seção 1.5
                if '1.5 Demonstração Prática: Técnicas de Detecção de Bordas' in source[0]:
                    # Marcar para mover todas as células a partir desta
                    cells_to_move = module1['cells'][i:]
                    cells_to_remove = list(range(i, len(module1['cells'])))
                    break
    
    if not cells_to_move:
        print("❌ Seção 1.5 não encontrada no Módulo 1")
        return
    
    print(f"✓ Encontrada seção 1.5 no Módulo 1 ({len(cells_to_move)} células)")
    
    # Remover células do Módulo 1
    for i in reversed(cells_to_remove):
        del module1['cells'][i]
    
    print(f"✓ Removidas {len(cells_to_remove)} células do Módulo 1")
    
    # Modificar o título da seção para o Módulo 2
    for cell in cells_to_move:
        if cell['cell_type'] == 'markdown':
            source = cell['source']
            if isinstance(source, list):
                for i, line in enumerate(source):
                    if '1.5 Demonstração Prática: Técnicas de Detecção de Bordas' in line:
                        source[i] = line.replace('1.5', '2.8')
                        print("✓ Título da seção atualizado para 2.8")
                    elif 'images/modulo1/fotografo.png' in line:
                        source[i] = line.replace('images/modulo1/', 'images/modulo2/')
                        print("✓ Caminho da imagem atualizado para modulo2")
    
    # Adicionar células ao Módulo 2
    module2['cells'].extend(cells_to_move)
    print(f"✓ Adicionadas {len(cells_to_move)} células ao Módulo 2")
    
    # Salvar módulos atualizados
    with open('curso/01_introducao_historia_visao_computacional.ipynb', 'w', encoding='utf-8') as f:
        json.dump(module1, f, indent=2, ensure_ascii=False)
    
    with open('curso/02_processamento_digital_imagem_fundamentos.ipynb', 'w', encoding='utf-8') as f:
        json.dump(module2, f, indent=2, ensure_ascii=False)
    
    print("✅ Movimentação concluída com sucesso!")
    print("\n📋 Resumo das mudanças:")
    print("   • Seção 1.5 removida do Módulo 1")
    print("   • Seção 2.8 adicionada ao Módulo 2")
    print("   • Título atualizado: '2.8 Demonstração Prática: Técnicas de Detecção de Bordas'")
    print("   • Caminho da imagem atualizado para modulo2")

def main():
    """Função principal"""
    
    move_edge_detection_section()
    
    print("\n🎯 Benefícios da movimentação:")
    print("   • Conteúdo mais apropriado no Módulo 2 (Processamento Digital)")
    print("   • Módulo 1 focado em introdução e história")
    print("   • Melhor organização pedagógica")
    print("   • Demonstração prática no contexto correto")

if __name__ == "__main__":
    main()
