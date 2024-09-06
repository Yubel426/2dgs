# shader/__init__.py

import os
import importlib

def get_model(model_name):
    """
    Returns the model object based on the given model name.
    """
    models_dir = os.path.dirname(os.path.abspath(__file__))
    model_module_name = "shader.model"  # Correct module path
    
    try:
        model_module = importlib.import_module(model_module_name)
        model_class = getattr(model_module, model_name)
        return model_class
    except ImportError:
        raise ValueError(f"Model module '{model_module_name}' not found in the shader directory. Checked path: {models_dir}")
    except AttributeError:
        raise ValueError(f"Class '{model_name}' not found in the module '{model_module_name}'.")

# Example usage:
model_class = get_model('MLPWithPE')
model_instance = model_class()  # Assuming MLPWithPE has an initializer without required arguments