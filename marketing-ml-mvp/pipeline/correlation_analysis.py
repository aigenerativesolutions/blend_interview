"""
Análisis de Correlaciones para Marketing Campaign
Identifica correlaciones fuertes y filtra redundancias obvias
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime
from .json_utils import safe_json_dump

logger = logging.getLogger(__name__)

class CorrelationAnalyzer:
    """
    Analizador de correlaciones con filtrado inteligente de redundancias
    """
    
    def __init__(self):
        self.correlation_matrix = None
        self.filtered_correlations = None
        
        # Definir correlaciones redundantes a filtrar
        self.redundant_pairs = {
            # Variables compuestas con sus componentes
            'Total_Kids': ['Kidhome', 'Teenhome'],
            'Total_Spent': ['MntWines', 'MntFruits', 'MntMeatProducts', 
                           'MntFishProducts', 'MntSweetProducts', 'MntGoldProds'],
            'AcceptedCmpOverall': ['AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 
                                  'AcceptedCmp4', 'AcceptedCmp5'],
            
            # Grupos de variables categóricas one-hot
            'Education_Group': ['Education_2n Cycle', 'Education_Basic', 
                               'Education_Graduation', 'Education_Master', 'Education_PhD'],
            'Marital_Status_Group': ['Marital_Status_Divorced', 'Marital_Status_Married', 
                                    'Marital_Status_Single', 'Marital_Status_Together', 
                                    'Marital_Status_Widow', 'Marital_Status_YOLO']
        }
    
    def calculate_correlation_matrix(self, X: pd.DataFrame) -> np.ndarray:
        """
        Calcular matriz de correlación completa
        """
        logger.info("📊 Calculando matriz de correlación...")
        
        # Calcular correlaciones Pearson
        self.correlation_matrix = X.corr()
        
        logger.info(f"Matriz de correlación calculada: {self.correlation_matrix.shape}")
        
        return self.correlation_matrix
    
    def _is_redundant_pair(self, var1: str, var2: str) -> bool:
        """
        Verificar si un par de variables es redundante
        """
        # Evitar correlación consigo mismo
        if var1 == var2:
            return True
            
        # Verificar pares compuestos-componentes
        for composite, components in self.redundant_pairs.items():
            if composite == 'Education_Group' or composite == 'Marital_Status_Group':
                # Para grupos one-hot, evitar correlaciones entre miembros del mismo grupo
                if var1 in components and var2 in components:
                    return True
            else:
                # Para variables compuestas, evitar correlación con componentes
                if (var1 == composite and var2 in components) or \
                   (var2 == composite and var1 in components):
                    return True
        
        return False
    
    def filter_redundant_correlations(self, correlation_matrix: pd.DataFrame) -> Dict[str, Any]:
        """
        Filtrar correlaciones redundantes y extraer significativas
        """
        logger.info("🔍 Filtrando correlaciones redundantes...")
        
        strong_positive = []  # > 0.5
        strong_negative = []  # < -0.5
        all_correlations = []
        
        # Obtener triángulo superior para evitar duplicados
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool), k=1)
        
        for i in range(len(correlation_matrix.columns)):
            for j in range(i + 1, len(correlation_matrix.columns)):
                var1 = correlation_matrix.columns[i]
                var2 = correlation_matrix.columns[j]
                corr_value = correlation_matrix.iloc[i, j]
                
                # Saltar si es NaN o redundante
                if pd.isna(corr_value) or self._is_redundant_pair(var1, var2):
                    continue
                
                correlation_entry = {
                    'variable_1': var1,
                    'variable_2': var2,
                    'correlation': float(corr_value),
                    'abs_correlation': float(abs(corr_value)),
                    'interpretation': self._interpret_correlation(corr_value)
                }
                
                all_correlations.append(correlation_entry)
                
                # Clasificar por fuerza
                if corr_value > 0.5:
                    strong_positive.append(correlation_entry)
                elif corr_value < -0.5:
                    strong_negative.append(correlation_entry)
        
        # Ordenar por valor absoluto descendente
        strong_positive.sort(key=lambda x: x['correlation'], reverse=True)
        strong_negative.sort(key=lambda x: x['correlation'])
        all_correlations.sort(key=lambda x: x['abs_correlation'], reverse=True)
        
        self.filtered_correlations = {
            'strong_positive': strong_positive,
            'strong_negative': strong_negative,
            'all_significant': all_correlations[:50],  # Top 50 correlaciones
            'summary': {
                'total_pairs_analyzed': len(all_correlations),
                'strong_positive_count': len(strong_positive),
                'strong_negative_count': len(strong_negative),
                'redundant_pairs_filtered': self._count_redundant_pairs()
            }
        }
        
        logger.info(f"📈 Correlaciones fuertes positivas (>0.5): {len(strong_positive)}")
        logger.info(f"📉 Correlaciones fuertes negativas (<-0.5): {len(strong_negative)}")
        logger.info(f"🔢 Total correlaciones analizadas: {len(all_correlations)}")
        
        return self.filtered_correlations
    
    def _interpret_correlation(self, corr_value: float) -> str:
        """
        Interpretar la fuerza de la correlación
        """
        abs_corr = abs(corr_value)
        
        if abs_corr >= 0.8:
            strength = "muy_fuerte"
        elif abs_corr >= 0.6:
            strength = "fuerte" 
        elif abs_corr >= 0.4:
            strength = "moderada"
        elif abs_corr >= 0.2:
            strength = "débil"
        else:
            strength = "muy_débil"
        
        direction = "positiva" if corr_value > 0 else "negativa"
        
        return f"{strength}_{direction}"
    
    def _count_redundant_pairs(self) -> int:
        """
        Contar cuántos pares redundantes se filtraron
        """
        count = 0
        for composite, components in self.redundant_pairs.items():
            if composite.endswith('_Group'):
                # Combinaciones dentro del grupo
                count += len(components) * (len(components) - 1) // 2
            else:
                # Composite con cada componente
                count += len(components)
        
        return count
    
    def create_correlation_plots(self, correlation_matrix: pd.DataFrame, 
                               artifacts_dir: Path) -> Dict[str, str]:
        """
        Crear visualizaciones de correlaciones
        """
        logger.info("📊 Generando visualizaciones de correlaciones...")
        
        plots_dir = artifacts_dir / "correlation_analysis"
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        plot_paths = {}
        
        # 1. Heatmap completo
        plt.figure(figsize=(16, 14))
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool), k=1)
        
        sns.heatmap(
            correlation_matrix,
            mask=mask,
            annot=False,
            cmap='RdBu_r',
            center=0,
            square=True,
            cbar_kws={'label': 'Correlación'}
        )
        
        plt.title('Matriz de Correlación Completa', fontsize=16, pad=20)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        heatmap_path = plots_dir / "correlation_heatmap.png"
        plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
        plt.close()
        plot_paths['heatmap_complete'] = str(heatmap_path)
        
        # 2. Heatmap solo correlaciones significativas (>0.4 o <-0.4)
        significant_mask = (np.abs(correlation_matrix) >= 0.4) & ~np.eye(len(correlation_matrix), dtype=bool)
        
        if significant_mask.any().any():
            plt.figure(figsize=(12, 10))
            
            # Crear máscara combinada (triángulo superior + no significativas)
            combined_mask = mask | (~significant_mask)
            
            sns.heatmap(
                correlation_matrix,
                mask=combined_mask,
                annot=True,
                fmt='.2f',
                cmap='RdBu_r',
                center=0,
                square=True,
                cbar_kws={'label': 'Correlación'},
                annot_kws={'size': 8}
            )
            
            plt.title('Correlaciones Significativas (|r| ≥ 0.4)', fontsize=16, pad=20)
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            
            significant_path = plots_dir / "significant_correlations.png"
            plt.savefig(significant_path, dpi=300, bbox_inches='tight')
            plt.close()
            plot_paths['heatmap_significant'] = str(significant_path)
        
        # 3. Distribución de correlaciones
        plt.figure(figsize=(10, 6))
        
        # Obtener correlaciones del triángulo superior
        upper_triangle = correlation_matrix.where(mask).stack().values
        upper_triangle = upper_triangle[~np.isnan(upper_triangle)]
        
        plt.hist(upper_triangle, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        plt.axvline(x=0.5, color='red', linestyle='--', label='Fuerte positiva (0.5)')
        plt.axvline(x=-0.5, color='red', linestyle='--', label='Fuerte negativa (-0.5)')
        plt.axvline(x=0, color='black', linestyle='-', alpha=0.5)
        
        plt.xlabel('Valor de Correlación')
        plt.ylabel('Frecuencia')
        plt.title('Distribución de Correlaciones', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        dist_path = plots_dir / "correlation_distribution.png"
        plt.savefig(dist_path, dpi=300, bbox_inches='tight')
        plt.close()
        plot_paths['distribution'] = str(dist_path)
        
        logger.info(f"📊 {len(plot_paths)} visualizaciones guardadas en {plots_dir}")
        
        return plot_paths
    
    def save_analysis_results(self, correlation_matrix: pd.DataFrame,
                            filtered_correlations: Dict[str, Any],
                            plot_paths: Dict[str, str],
                            artifacts_dir: Path) -> None:
        """
        Guardar todos los resultados del análisis de correlaciones
        """
        logger.info("💾 Guardando resultados de análisis de correlaciones...")
        
        corr_dir = artifacts_dir / "correlation_analysis"
        corr_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Matriz de correlación completa
        matrix_path = corr_dir / "correlation_matrix.json"
        correlation_dict = correlation_matrix.to_dict()
        
        with open(matrix_path, 'w') as f:
            safe_json_dump(correlation_dict, f, indent=2)
        
        # 2. Correlaciones filtradas
        filtered_path = corr_dir / "filtered_correlations.json"
        with open(filtered_path, 'w') as f:
            safe_json_dump(filtered_correlations, f, indent=2)
        
        # 3. Metadatos del análisis
        metadata = {
            'analysis_timestamp': datetime.now().isoformat(),
            'matrix_shape': list(correlation_matrix.shape),
            'features_analyzed': list(correlation_matrix.columns),
            'redundant_pairs_definition': self.redundant_pairs,
            'thresholds': {
                'strong_positive': 0.5,
                'strong_negative': -0.5,
                'significant_visualization': 0.4
            },
            'plot_files': plot_paths,
            'version': '1.0.0'
        }
        
        metadata_path = corr_dir / "correlation_metadata.json"
        with open(metadata_path, 'w') as f:
            safe_json_dump(metadata, f, indent=2)
        
        logger.info(f"💾 Análisis de correlaciones guardado en {corr_dir}")

def run_correlation_analysis(X: pd.DataFrame, artifacts_dir: Path) -> Dict[str, Any]:
    """
    Ejecutar análisis completo de correlaciones
    """
    logger.info("🔗 Iniciando análisis completo de correlaciones...")
    
    # Crear analizador
    analyzer = CorrelationAnalyzer()
    
    # 1. Calcular matriz de correlación
    correlation_matrix = analyzer.calculate_correlation_matrix(X)
    
    # 2. Filtrar correlaciones redundantes y extraer significativas
    filtered_correlations = analyzer.filter_redundant_correlations(correlation_matrix)
    
    # 3. Crear visualizaciones
    plot_paths = analyzer.create_correlation_plots(correlation_matrix, artifacts_dir)
    
    # 4. Guardar resultados
    analyzer.save_analysis_results(
        correlation_matrix, filtered_correlations, plot_paths, artifacts_dir
    )
    
    # Mostrar resumen
    summary = filtered_correlations['summary']
    logger.info(f"📈 Top 3 correlaciones positivas:")
    for i, corr in enumerate(filtered_correlations['strong_positive'][:3]):
        logger.info(f"  {i+1}. {corr['variable_1']} ↔ {corr['variable_2']}: {corr['correlation']:.3f}")
    
    if filtered_correlations['strong_negative']:
        logger.info(f"📉 Top 3 correlaciones negativas:")
        for i, corr in enumerate(filtered_correlations['strong_negative'][:3]):
            logger.info(f"  {i+1}. {corr['variable_1']} ↔ {corr['variable_2']}: {corr['correlation']:.3f}")
    
    logger.info("🔗 Análisis de correlaciones completado")
    
    return {
        'correlation_analyzer': analyzer,
        'correlation_matrix': correlation_matrix,
        'filtered_correlations': filtered_correlations,
        'plot_paths': plot_paths,
        'summary': summary
    }

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("🔗 Script de Análisis de Correlaciones")
    print("Identifica correlaciones fuertes y filtra redundancias")
    print("Usar run_pipeline.py para ejecutar el pipeline completo")