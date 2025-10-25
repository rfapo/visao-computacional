# 🚀 Script de Otimização para Google Colab
# Execute este código no início do seu notebook para resolver problemas de carregamento de imagens

import time
import requests
from IPython.display import Image, display
import os

class ImageLoader:
    """Classe para carregamento otimizado de imagens"""
    
    def __init__(self):
        self.cache_dir = '/tmp/image_cache'
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
        # Criar diretório de cache
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def load_image_with_fallback(self, url, alt_urls=None, max_retries=3):
        """Carrega imagem com fallback e retry"""
        
        urls_to_try = [url] + (alt_urls or [])
        
        for attempt in range(max_retries):
            for current_url in urls_to_try:
                try:
                    # Tentar carregar imagem
                    response = self.session.get(current_url, timeout=10)
                    response.raise_for_status()
                    
                    # Salvar em cache
                    filename = os.path.basename(current_url)
                    cache_path = os.path.join(self.cache_dir, filename)
                    
                    with open(cache_path, 'wb') as f:
                        f.write(response.content)
                    
                    print(f"✅ Imagem carregada: {filename}")
                    return Image(cache_path)
                    
                except requests.exceptions.RequestException as e:
                    print(f"⚠️  Tentativa {attempt + 1} falhou para {current_url}: {e}")
                    
                    if attempt < max_retries - 1:
                        # Backoff exponencial
                        wait_time = 2 ** attempt
                        print(f"⏳ Aguardando {wait_time}s antes da próxima tentativa...")
                        time.sleep(wait_time)
        
        print(f"❌ Falha ao carregar imagem após {max_retries} tentativas")
        return None
    
    def get_alternative_urls(self, original_url):
        """Gera URLs alternativas para uma imagem"""
        
        # Extrair informações da URL
        if 'jsdelivr' in original_url:
            # Converter jsdelivr para outras opções
            alt_urls = [
                original_url.replace('cdn.jsdelivr.net/gh/', 'raw.githubusercontent.com/').replace('@main', '/main'),
                original_url.replace('cdn.jsdelivr.net/gh/', 'github.com/').replace('@main', '/blob/main') + '?raw=true'
            ]
        elif 'raw.githubusercontent.com' in original_url:
            # Converter raw.githubusercontent para outras opções
            alt_urls = [
                original_url.replace('raw.githubusercontent.com/', 'cdn.jsdelivr.net/gh/').replace('/main/', '@main/'),
                original_url.replace('raw.githubusercontent.com/', 'github.com/').replace('/main/', '/blob/main/') + '?raw=true'
            ]
        else:
            alt_urls = []
        
        return alt_urls

# Inicializar loader
image_loader = ImageLoader()

def display_image_safe(url, description="Imagem"):
    """Exibe imagem com fallback automático"""
    
    print(f"🖼️  Carregando: {description}")
    
    # Gerar URLs alternativas
    alt_urls = image_loader.get_alternative_urls(url)
    
    # Tentar carregar com fallback
    image = image_loader.load_image_with_fallback(url, alt_urls)
    
    if image:
        display(image)
    else:
        print(f"❌ Não foi possível carregar: {description}")
        print(f"🔗 URL original: {url}")

# Exemplo de uso:
print("🚀 Sistema de carregamento otimizado de imagens inicializado!")
print("📋 Use display_image_safe(url, 'Descrição') para carregar imagens com fallback automático")

# URLs de exemplo (substitua pelas suas URLs)
example_urls = [
    "https://cdn.jsdelivr.net/gh/rfapo/visao-computacional@main/images/modulo1/definicao_visao_computacional.png",
    "https://cdn.jsdelivr.net/gh/rfapo/visao-computacional@main/images/modulo2/fundamentos_processamento.png"
]

print("\n📖 Exemplos de uso:")
for i, url in enumerate(example_urls, 1):
    print(f"display_image_safe('{url}', 'Exemplo {i}')")
