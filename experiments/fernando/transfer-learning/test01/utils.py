from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

def transform_diagnosis_to_numerical(diagnosis_result, dictonary):
    """
    Trasnform the diagnois to numerical numbers
    """
    return dictonary[diagnosis_result]

def add_prefix_to_string(string, prefix):
    """
    Add a sring prefix to the input string
    """
    return prefix+string

def transform_image_for_prediction(image):
    image_resized = image.resize((224, 224))
    image_right_shape = np.array(image_resized)
    image_to_classify = np.expand_dims(image_right_shape, 0)
    return image_to_classify

def predict_image(model, image_path):
    image = transform_image_for_prediction(Image.open(image_path))
    if image.shape[3] == 4:
        image = image[:,:,:,:3]
    return model.predict(image)

def probability_to_categorical(array_probabilities, classe_names):
    return classe_names[np.argmax(array_probabilities)]
    
def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=True,
                          title=None,
                          cmap=plt.cm.Greens, save_to_file = False):
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        #print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    #print(cm)

    fig, ax = plt.subplots(figsize = (16,16))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    if save_to_file:
        plt.savefig('Assets/files/' + title + '.pdf')
    return ax
    



