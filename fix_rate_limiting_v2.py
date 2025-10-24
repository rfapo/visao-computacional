#!/usr/bin/env python3
"""
Script para resolver problema de rate limiting do GitHub
Converte URLs raw.githubusercontent.com para CDN jsdelivr
"""

import json
import os
import glob

def fix_github_rate_limiting():
    """Converte URLs do GitHub para CDN jsdelivr para resolver rate limiting"""
    
    print("🔧 RESOLVENDO PROBLEMA DE RATE LIMITING DO GITHUB")
    print("=" * 60)
    
    total_changes = 0
    
    for module_num in range(1, 11):
        # Encontrar o arquivo do notebook
        notebook_files = glob.glob(f"curso/{module_num:02d}_*.ipynb")
        if not notebook_files:
            continue
            
        notebook_path = notebook_files[0]
        
        # Ler o notebook
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook = json.load(f)
        
        changes_made = 0
        
        for cell in notebook['cells']:
            if cell['cell_type'] == 'markdown':
                source = cell['source']
                if isinstance(source, list):
                    for i, line in enumerate(source):
                        # Substituir URLs raw.githubusercontent.com por jsdelivr CDN
                        if 'raw.githubusercontent.com/rfapo/visao-computacional/main/images/' in line:
                            # Converter URL do GitHub para jsdelivr CDN
                            old_url = 'https://raw.githubusercontent.com/rfapo/visao-computacional/main/images/'
                            new_url = 'https://cdn.jsdelivr.net/gh/rfapo/visao-computacional@main/images/'
                            
                            # Substituir URL e remover ?raw=true (não necessário no jsdelivr)
                            new_line = line.replace(old_url, new_url).replace('?raw=true', '')
                            source[i] = new_line
                            changes_made += 1
                            print(f"  ✓ Convertida URL em {notebook_path}")
        
        # Salvar o notebook atualizado
        if changes_made > 0:
            with open(notebook_path, 'w', encoding='utf-8') as f:
                json.dump(notebook, f, indent=2, ensure_ascii=False)
            print(f"✅ Módulo {module_num}: {changes_made} URLs convertidas")
            total_changes += changes_made
        else:
            print(f"⚠️  Módulo {module_num}: Nenhuma URL encontrada")
    
    return total_changes

def create_optimization_guide():
    """Cria guia de otimização para Google Colab"""
    
    guide_content = """
# 🚀 Guia de Otimização para Google Colab

## Problema: Rate Limiting do GitHub (429 Too Many Requests)

### Causa
O Google Colab carrega muitas imagens simultaneamente do GitHub, excedendo os limites de requisições (rate limiting).

### Solução Implementada
Conversão de URLs do GitHub para CDN jsdelivr:

**Antes:**
```
https://raw.githubusercontent.com/rfapo/visao-computacional/main/images/moduloX/imagem.png?raw=true
```

**Depois:**
```
https://cdn.jsdelivr.net/gh/rfapo/visao-computacional@main/images/moduloX/imagem.png
```

### Benefícios do CDN jsdelivr:
- ✅ **Sem rate limiting**: Limites muito mais altos
- ✅ **Performance superior**: CDN global otimizado
- ✅ **Cache inteligente**: Reduz requisições desnecessárias
- ✅ **Compatibilidade total**: Funciona em Colab, GitHub e Cursor
- ✅ **Disponibilidade alta**: 99.9% uptime

### URLs Alternativas (se necessário):
Se ainda houver problemas, use estas alternativas:

1. **GitHub Pages** (se configurado):
   ```
   https://rfapo.github.io/visao-computacional/images/moduloX/imagem.png
   ```

2. **GitHub Raw** (com cache):
   ```
   https://raw.githubusercontent.com/rfapo/visao-computacional/main/images/moduloX/imagem.png
   ```

3. **Download local**:
   ```python
   # No Colab, baixar imagens localmente
   !wget https://cdn.jsdelivr.net/gh/rfapo/visao-computacional@main/images/moduloX/imagem.png
   ```

### Monitoramento:
- Verifique o console do navegador para erros 429
- Use ferramentas de desenvolvedor para monitorar requisições
- Implemente retry com backoff exponencial se necessário
"""
    
    # Salvar guia
    with open("GUIA_OTIMIZACAO_COLAB.md", 'w', encoding='utf-8') as f:
        f.write(guide_content)
    
    print("📝 Guia de otimização criado: GUIA_OTIMIZACAO_COLAB.md")

def main():
    """Função principal"""
    
    total_changes = fix_github_rate_limiting()
    create_optimization_guide()
    
    print("\n" + "=" * 60)
    print(f"✅ Correção concluída! Total: {total_changes} URLs convertidas")
    print("\n🎯 Benefícios:")
    print("   • Resolve erro 429 (Too Many Requests)")
    print("   • Melhora performance no Google Colab")
    print("   • Mantém compatibilidade com GitHub e Cursor")
    print("   • Usa CDN otimizado (jsdelivr)")
    print("   • Guia de otimização criado")

if __name__ == "__main__":
    main()
