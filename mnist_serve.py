from __future__ import annotations

from gridserve_sdk import Composition, GridModel, ModelComponent, expose
from gridserve_sdk.types import Image, Number


class MNISTInference(ModelComponent):
    def __init__(self, model):  # skipcq: PYL-W1401, PYL-W0621
        self.model = model

    @expose(
        inputs={"img": Image()},
        outputs={"prediction": Number()},
    )
    def classify(self, img):
        img = img.float() / 255
        out = self.model(img)
        return out.argmax()


if __name__ == "__main__":
    mnist = GridModel(os.getenv("GRIDSERVE_MODEL_PATH"))
    component = MNISTInference(mnist)
    composition = Composition(classifier=component)
    composition.serve()
