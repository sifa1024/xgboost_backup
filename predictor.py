import base64
import io
import kserve
import os
import xgboost
import shap
import base64


from kserve.errors import InferenceError
from modelpath import modelpath_join
from typing import Dict
from io import BytesIO

class XGBoostModel(kserve.Model):
    def __init__(self, name: str):
        super().__init__(name)
        self.name = name
        self.cfg = None
        self._xgboost = None

    def load(self):
        weights = modelpath_join("model.bst")
        self._xgboost = xgboost.XGBClassifier()
        self._xgboost.load_model(weights)
        
        self.ready = True
        return self.ready

    # def _preprocess(self, ):
    def _global_explain_generate(self, data):
        explainer = shap.TreeExplainer(self.xgboost)
        shap_values = explainer(data)
        shap.summary_plot(shap_values, X_test, show=False)
        tmpfile = BytesIO()
        plt.savefig(tmpfile, format = "png")
        tmpfile.seek(0)
        return base64.b64encode(tmpfile.read()).decode("utf-8")
        
    def predict(self, request: Dict, headers: Dict[str, str] = None) -> Dict:
        try:
            # Use of list as input is deprecated see https://github.com/dmlc/xgboost/pull/3970
            inputs = request["instances"]
            result = self._xgboost.predict(inputs)
            encoded = self._global_explain_generate(inputs)
            return {"predictions": result.tolist(), "image": encoded}
        except Exception as e:
            raise InferenceError(str(e))
