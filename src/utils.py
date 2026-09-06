import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics

def create_regression_table(results):
    """Принимает словарь с результатами моделей, возвращает pandas DataFrame."""
    df = pd.DataFrame.from_dict(results, orient='index')
    df = df.sort_values('R2', ascending=False)
    return df

def plot_regression_results(y_test, y_pred, model_name, r2):
    """Строит scatter plot для регрессии."""
    plt.figure(figsize=(6, 5))
    plt.scatter(y_test, y_pred, alpha=0.5, s=15, color='blue')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
    plt.xlabel('Фактические shares')
    plt.ylabel('Предсказанные shares')
    plt.title(f'{model_name}\nR² = {r2:.3f}')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_classification_results(y_test, y_pred, model_name):
    """Строит матрицу ошибок и сравнительный график."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Матрица ошибок
    cm = metrics.confusion_matrix(y_test, y_pred)
    metrics.ConfusionMatrixDisplay(cm, display_labels=['Непопулярные', 'Популярные']).plot(ax=axes[0], cmap='Blues')
    axes[0].set_title(f'{model_name}: Матрица ошибок')
    
    # Сравнение первых 100 предсказаний
    axes[1].scatter(range(100), y_test[:100], alpha=0.7, s=80, label='Фактические', marker='o')
    axes[1].scatter(range(100), y_pred[:100], alpha=0.7, s=40, label='Предсказанные', marker='x', color='red')
    axes[1].set_xlabel('Наблюдение')
    axes[1].set_ylabel('Класс')
    axes[1].set_title('Сравнение (первые 100)')
    axes[1].set_yticks([0, 1])
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()