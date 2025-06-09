def plot_molecular_structure(molecule, filename='molecule.png'):
    """
    Visualizes the molecular structure and saves it as an image file.

    Parameters:
    molecule: The molecular structure to visualize.
    filename: The name of the file to save the image.
    """
    from rdkit import Chem
    from rdkit.Chem import Draw

    img = Draw.MolToImage(molecule)
    img.save(filename)

def plot_feature_importance(importances, feature_names, title='Feature Importance'):
    """
    Plots the feature importance for a model.

    Parameters:
    importances: A list or array of feature importances.
    feature_names: A list of feature names corresponding to the importances.
    title: The title of the plot.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    indices = np.argsort(importances)[::-1]
    plt.figure()
    plt.title(title)
    plt.bar(range(len(importances)), importances[indices], align='center')
    plt.xticks(range(len(importances)), np.array(feature_names)[indices], rotation=90)
    plt.xlim([-1, len(importances)])
    plt.tight_layout()
    plt.show()

def plot_model_performance(history, title='Model Performance'):
    """
    Plots the training and validation performance of a model.

    Parameters:
    history: The history object returned by the model's fit method.
    title: The title of the plot.
    """
    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()