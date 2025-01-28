import matplotlib.pyplot as plt


def plot_learning_curves(standard_results, mvo_results):
    plt.figure(figsize=(15, 5))
    
    # Training Loss
    plt.subplot(1, 2, 1)
    plt.plot(standard_results['training_losses'], label='Standard SGD')
    plt.plot(mvo_results['training_losses'], label='MVO-optimized SGD')
    plt.title('Training Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Validation Loss
    plt.subplot(1, 2, 2)
    plt.plot(standard_results['validation_losses'], label='Standard SGD')
    plt.plot(mvo_results['validation_losses'], label='MVO-optimized SGD')
    plt.title('Validation Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def plot_prediction_comparison(y_true, standard_pred, mvo_pred):
    plt.figure(figsize=(15, 5))
    
    # Standard SGD predictions
    plt.subplot(1, 2, 1)
    plt.scatter(y_true, standard_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.title('Standard SGD Predictions vs True Values')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    
    # MVO-optimized predictions
    plt.subplot(1, 2, 2)
    plt.scatter(y_true, mvo_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.title('MVO-optimized SGD Predictions vs True Values')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    
    plt.tight_layout()
    plt.show()

