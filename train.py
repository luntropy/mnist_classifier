from mnist_classifier.config import CFG
from mnist_classifier.model import MNISTClassifier

def run():
    """Builds model, loads data, trains and evaluates"""

    model = MNISTClassifier(CFG)
    model.load_data()
    model.build()
    model.train()
    model.evaluate()    

if __name__ == '__main__':
    run()
