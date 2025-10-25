#!/usr/bin/env python3
"""
Comparação de Arquiteturas CNN com Modelos Pré-treinados
Otimizado para Google Colab
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision import datasets
import matplotlib.pyplot as plt
import numpy as np
import time
from sklearn.metrics import accuracy_score, classification_report
import seaborn as sns
from tqdm import tqdm

# Configurações para Colab
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Usando dispositivo: {device}")

class PretrainedModelComparison:
    """Classe para comparar modelos pré-treinados"""
    
    def __init__(self, num_classes=10):
        self.num_classes = num_classes
        self.device = device
        self.models = {}
        self.results = {}
        
    def load_pretrained_models(self):
        """Carrega modelos pré-treinados do torchvision"""
        
        print("Carregando modelos pré-treinados...")
        
        # AlexNet pré-treinado
        self.models['alexnet'] = models.alexnet(pretrained=True)
        # Modificar última camada para número de classes
        self.models['alexnet'].classifier[6] = nn.Linear(4096, self.num_classes)
        
        # VGG-16 pré-treinado
        self.models['vgg16'] = models.vgg16(pretrained=True)
        # Modificar última camada
        self.models['vgg16'].classifier[6] = nn.Linear(4096, self.num_classes)
        
        # ResNet-50 pré-treinado
        self.models['resnet50'] = models.resnet50(pretrained=True)
        # Modificar última camada
        self.models['resnet50'].fc = nn.Linear(2048, self.num_classes)
        
        # EfficientNet-B0 pré-treinado
        try:
            self.models['efficientnet'] = models.efficientnet_b0(pretrained=True)
            self.models['efficientnet'].classifier[1] = nn.Linear(1280, self.num_classes)
        except:
            print("EfficientNet não disponível, usando ResNet-18")
            self.models['efficientnet'] = models.resnet18(pretrained=True)
            self.models['efficientnet'].fc = nn.Linear(512, self.num_classes)
        
        # Mover todos os modelos para o dispositivo
        for name, model in self.models.items():
            self.models[name] = model.to(self.device)
            
        print("✅ Modelos carregados com sucesso!")
        
    def count_parameters(self):
        """Conta parâmetros de cada modelo"""
        
        param_counts = {}
        for name, model in self.models.items():
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            param_counts[name] = {
                'total': total_params,
                'trainable': trainable_params
            }
            
        return param_counts
    
    def create_sample_data(self, batch_size=32):
        """Cria dados de exemplo para teste"""
        
        # Transformações para imagens
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Criar dados sintéticos
        images = torch.randn(batch_size, 3, 224, 224).to(self.device)
        labels = torch.randint(0, self.num_classes, (batch_size,)).to(self.device)
        
        return images, labels
    
    def measure_inference_time(self, model, images, num_iterations=100):
        """Mede tempo de inferência"""
        
        model.eval()
        times = []
        
        with torch.no_grad():
            # Warmup
            for _ in range(10):
                _ = model(images)
            
            # Medição real
            for _ in range(num_iterations):
                start_time = time.time()
                _ = model(images)
                end_time = time.time()
                times.append(end_time - start_time)
        
        return np.mean(times), np.std(times)
    
    def calculate_model_size(self, model):
        """Calcula tamanho do modelo em MB"""
        
        param_size = 0
        buffer_size = 0
        
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
            
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
            
        size_all_mb = (param_size + buffer_size) / 1024**2
        return size_all_mb
    
    def evaluate_models(self, images, labels):
        """Avalia performance dos modelos"""
        
        results = {}
        
        for name, model in self.models.items():
            print(f"\nAvaliando {name}...")
            
            model.eval()
            
            # Tempo de inferência
            inference_time, inference_std = self.measure_inference_time(model, images)
            
            # Predições
            with torch.no_grad():
                outputs = model(images)
                predictions = torch.argmax(outputs, dim=1)
                accuracy = accuracy_score(labels.cpu(), predictions.cpu())
            
            # Tamanho do modelo
            model_size = self.calculate_model_size(model)
            
            # Contagem de parâmetros
            total_params = sum(p.numel() for p in model.parameters())
            
            results[name] = {
                'accuracy': accuracy,
                'inference_time': inference_time,
                'inference_std': inference_std,
                'model_size_mb': model_size,
                'total_params': total_params,
                'predictions': predictions.cpu().numpy(),
                'outputs': outputs.cpu().numpy()
            }
            
            print(f"✅ {name}: Accuracy={accuracy:.3f}, Time={inference_time:.4f}s")
        
        return results
    
    def visualize_results(self, param_counts, results):
        """Visualiza resultados da comparação"""
        
        model_names = list(results.keys())
        
        # Preparar dados para visualização
        accuracies = [results[name]['accuracy'] for name in model_names]
        inference_times = [results[name]['inference_time'] for name in model_names]
        model_sizes = [results[name]['model_size_mb'] for name in model_names]
        total_params = [param_counts[name]['total'] for name in model_names]
        
        # Criar figura com subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Acurácia
        bars1 = axes[0, 0].bar(model_names, accuracies, color=['red', 'blue', 'green', 'orange'])
        axes[0, 0].set_title('Acurácia dos Modelos', fontsize=14, fontweight='bold')
        axes[0, 0].set_ylabel('Acurácia')
        axes[0, 0].set_ylim(0, 1)
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Adicionar valores nas barras
        for bar, acc in zip(bars1, accuracies):
            axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                           f'{acc:.3f}', ha='center', va='bottom')
        
        # 2. Tempo de Inferência
        bars2 = axes[0, 1].bar(model_names, inference_times, color=['red', 'blue', 'green', 'orange'])
        axes[0, 1].set_title('Tempo de Inferência', fontsize=14, fontweight='bold')
        axes[0, 1].set_ylabel('Tempo (segundos)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        for bar, time_val in zip(bars2, inference_times):
            axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0001, 
                           f'{time_val:.4f}s', ha='center', va='bottom')
        
        # 3. Tamanho do Modelo
        bars3 = axes[0, 2].bar(model_names, model_sizes, color=['red', 'blue', 'green', 'orange'])
        axes[0, 2].set_title('Tamanho do Modelo', fontsize=14, fontweight='bold')
        axes[0, 2].set_ylabel('Tamanho (MB)')
        axes[0, 2].tick_params(axis='x', rotation=45)
        
        for bar, size in zip(bars3, model_sizes):
            axes[0, 2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                           f'{size:.1f}MB', ha='center', va='bottom')
        
        # 4. Número de Parâmetros
        bars4 = axes[1, 0].bar(model_names, [p/1000000 for p in total_params], 
                               color=['red', 'blue', 'green', 'orange'])
        axes[1, 0].set_title('Número de Parâmetros', fontsize=14, fontweight='bold')
        axes[1, 0].set_ylabel('Parâmetros (Milhões)')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        for bar, params in zip(bars4, total_params):
            axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                           f'{params/1000000:.1f}M', ha='center', va='bottom')
        
        # 5. Eficiência (Acurácia por Parâmetro)
        efficiency = [acc/(params/1000000) for acc, params in zip(accuracies, total_params)]
        bars5 = axes[1, 1].bar(model_names, efficiency, color=['red', 'blue', 'green', 'orange'])
        axes[1, 1].set_title('Eficiência (Acurácia/Parâmetros)', fontsize=14, fontweight='bold')
        axes[1, 1].set_ylabel('Eficiência')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        for bar, eff in zip(bars5, efficiency):
            axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                           f'{eff:.2f}', ha='center', va='bottom')
        
        # 6. Comparação Normalizada
        x = np.arange(len(model_names))
        width = 0.2
        
        # Normalizar valores para comparação
        norm_acc = [acc/max(accuracies) for acc in accuracies]
        norm_time = [1 - (time_val - min(inference_times))/(max(inference_times) - min(inference_times)) 
                    for time_val in inference_times]  # Inverter para melhor = maior
        norm_size = [1 - (size - min(model_sizes))/(max(model_sizes) - min(model_sizes)) 
                    for size in model_sizes]  # Inverter para melhor = maior
        
        axes[1, 2].bar(x - width, norm_acc, width, label='Acurácia (norm)', alpha=0.8)
        axes[1, 2].bar(x, norm_time, width, label='Velocidade (norm)', alpha=0.8)
        axes[1, 2].bar(x + width, norm_size, width, label='Eficiência (norm)', alpha=0.8)
        
        axes[1, 2].set_title('Comparação Normalizada', fontsize=14, fontweight='bold')
        axes[1, 2].set_ylabel('Valor Normalizado')
        axes[1, 2].set_xticks(x)
        axes[1, 2].set_xticklabels(model_names, rotation=45)
        axes[1, 2].legend()
        
        plt.tight_layout()
        plt.show()
    
    def print_detailed_results(self, param_counts, results):
        """Imprime resultados detalhados"""
        
        print("\n" + "="*80)
        print("COMPARAÇÃO DETALHADA DE MODELOS PRÉ-TREINADOS")
        print("="*80)
        
        for name in results.keys():
            print(f"\n📊 {name.upper()}:")
            print(f"   • Acurácia: {results[name]['accuracy']:.4f} ({results[name]['accuracy']*100:.2f}%)")
            print(f"   • Tempo de Inferência: {results[name]['inference_time']:.4f}s ± {results[name]['inference_std']:.4f}s")
            print(f"   • Tamanho do Modelo: {results[name]['model_size_mb']:.2f} MB")
            print(f"   • Parâmetros Totais: {param_counts[name]['total']:,} ({param_counts[name]['total']/1000000:.1f}M)")
            print(f"   • Parâmetros Treináveis: {param_counts[name]['trainable']:,}")
            
            # Calcular métricas adicionais
            efficiency = results[name]['accuracy'] / (param_counts[name]['total']/1000000)
            speed_score = 1 / results[name]['inference_time']  # Maior = melhor
            
            print(f"   • Eficiência: {efficiency:.2f} (acurácia por M parâmetros)")
            print(f"   • Score de Velocidade: {speed_score:.2f} (inferências por segundo)")
        
        # Ranking
        print(f"\n🏆 RANKINGS:")
        
        # Por Acurácia
        accuracy_ranking = sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
        print(f"   🥇 Melhor Acurácia: {accuracy_ranking[0][0]} ({accuracy_ranking[0][1]['accuracy']:.4f})")
        
        # Por Velocidade
        speed_ranking = sorted(results.items(), key=lambda x: x[1]['inference_time'])
        print(f"   ⚡ Mais Rápido: {speed_ranking[0][0]} ({speed_ranking[0][1]['inference_time']:.4f}s)")
        
        # Por Eficiência
        efficiency_ranking = sorted(results.items(), 
                                  key=lambda x: x[1]['accuracy'] / (param_counts[x[0]]['total']/1000000), 
                                  reverse=True)
        print(f"   🎯 Mais Eficiente: {efficiency_ranking[0][0]} ({efficiency_ranking[0][1]['accuracy']/(param_counts[efficiency_ranking[0][0]]['total']/1000000):.2f})")
        
        # Por Tamanho (menor é melhor)
        size_ranking = sorted(results.items(), key=lambda x: x[1]['model_size_mb'])
        print(f"   📦 Menor Tamanho: {size_ranking[0][0]} ({size_ranking[0][1]['model_size_mb']:.2f}MB)")
        
        print("\n" + "="*80)

def main():
    """Função principal"""
    
    print("🚀 Iniciando Comparação de Modelos Pré-treinados")
    print("="*60)
    
    # Criar instância da comparação
    comparison = PretrainedModelComparison(num_classes=10)
    
    # Carregar modelos pré-treinados
    comparison.load_pretrained_models()
    
    # Contar parâmetros
    param_counts = comparison.count_parameters()
    
    # Criar dados de exemplo
    print("\nCriando dados de exemplo...")
    images, labels = comparison.create_sample_data(batch_size=64)
    print(f"✅ Dados criados: {images.shape[0]} imagens de {images.shape[1]}x{images.shape[2]}x{images.shape[3]}")
    
    # Avaliar modelos
    print("\nAvaliando modelos...")
    results = comparison.evaluate_models(images, labels)
    
    # Visualizar resultados
    print("\nGerando visualizações...")
    comparison.visualize_results(param_counts, results)
    
    # Imprimir resultados detalhados
    comparison.print_detailed_results(param_counts, results)
    
    print("\n✅ Comparação concluída!")
    
    return comparison, param_counts, results

if __name__ == "__main__":
    # Executar comparação
    comparison_obj, param_counts, results = main()
