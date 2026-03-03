import json
import sys
import os
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import (
    ProbabilisticCircuit,
)

from probabilistic_model.gui.main_window import MainWindow
from probabilistic_model.probabilistic_model import ProbabilisticModel
from qt_material import apply_stylesheet


def main(model_path: str = None):
    """
    Main entry point for starting the Probabilistic Model GUI.

    :param model_path: The path to the model to display in the GUI.
    """
    # Recommended for some Linux environments/Docker
    os.environ["QTWEBENGINE_DISABLE_SANDBOX"] = "1"

    # Required for QWebEngineView to work in many environments
    QApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts)

    # Sometimes helps with rendering on certain GPUs/drivers
    # In Qt6, AA_UseDesktopOpenGL is still available but often default.
    # On some systems, AA_UseOpenGLES or AA_UseSoftwareOpenGL might be needed instead.
    model_path = "/home/tom_sch/.config/JetBrains/PyCharm2025.3/scratches/model.pm"

    if model_path is None:
        model = None
    else:
        with open(model_path) as f:
            model = ProbabilisticCircuit.from_json(json.load(f))

    app = QApplication(sys.argv)
    apply_stylesheet(app, theme="dark_amber.xml")
    window = MainWindow(model=model)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
