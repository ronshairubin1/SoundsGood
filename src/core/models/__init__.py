from .cnn import CNNModel
from .rf import RandomForestModel
from .ensemble import EnsembleModel

def create_model(model_type, model_dir='models'):
    """
    Factory function to create the appropriate model type.
    
    Args:
        model_type (str): Type of model to create ('cnn', 'rf', or 'ensemble')
        model_dir (str): Directory for model storage
        
    Returns:
        BaseModel: An instance of the requested model type
    """
    model_type = model_type.lower()
    
    if model_type == 'cnn':
        return CNNModel(model_dir=model_dir)
    elif model_type == 'rf':
        return RandomForestModel(model_dir=model_dir)
    elif model_type == 'ensemble':
        return EnsembleModel(model_dir=model_dir)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Choose from: 'cnn', 'rf', 'ensemble'")
