from cloudpickle import load
import numpy as np
from typing import List
import mlrun

class ClassifierModel(mlrun.serving.V2ModelServer):
    def load(self):
        # Load the trained model artifact=
        model_file, _ = self.get_model('.pkl')
        self.model = load(open(model_file, 'rb'))

    def predict(self, body: dict) -> List:
        feats = np.asarray(body['inputs'])
        # Predict using the loaded model
        results: np.ndarray = self.model.predict(feats)
        return results.tolist()