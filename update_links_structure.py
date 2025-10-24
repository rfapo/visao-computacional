#!/usr/bin/env python3
"""
Script para atualizar todos os links dos módulos no index.html e README.md
Adiciona o prefixo 'curso/' aos links dos notebooks
"""

import re

def update_html_links():
    """Atualiza os links no index.html"""
    
    print("🔄 Atualizando links no index.html...")
    
    with open('index.html', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Padrão para encontrar links dos módulos
    pattern = r'(https://github\.com/rfapo/visao-computacional/blob/main/)(\d+_.*\.ipynb)'
    
    # Substituir adicionando 'curso/' antes do nome do arquivo
    updated_content = re.sub(pattern, r'\1curso/\2', content)
    
    with open('index.html', 'w', encoding='utf-8') as f:
        f.write(updated_content)
    
    # Contar quantas substituições foram feitas
    matches = re.findall(pattern, content)
    print(f"✅ {len(matches)} links atualizados no index.html")

def update_readme_links():
    """Atualiza os links no README.md"""
    
    print("🔄 Atualizando links no README.md...")
    
    with open('README.md', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Padrão para encontrar links dos módulos
    pattern = r'(https://github\.com/rfapo/visao-computacional/blob/main/)(\d+_.*\.ipynb)'
    
    # Substituir adicionando 'curso/' antes do nome do arquivo
    updated_content = re.sub(pattern, r'\1curso/\2', content)
    
    with open('README.md', 'w', encoding='utf-8') as f:
        f.write(updated_content)
    
    # Contar quantas substituições foram feitas
    matches = re.findall(pattern, content)
    print(f"✅ {len(matches)} links atualizados no README.md")

def main():
    """Função principal"""
    
    print("📁 Atualizando estrutura do projeto - pasta 'curso/'")
    print("=" * 60)
    
    update_html_links()
    update_readme_links()
    
    print("=" * 60)
    print("✅ Atualização concluída!")
    print("📂 Todos os links agora apontam para pasta 'curso/'")

if __name__ == "__main__":
    main()
