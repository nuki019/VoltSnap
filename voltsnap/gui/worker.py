"""异步 Worker — 推理和仿真后台线程"""
from __future__ import annotations

import logging
import traceback

from PyQt6.QtCore import QThread, pyqtSignal

from voltsnap.config import Config
from voltsnap.models import SimulationResult

logger = logging.getLogger("voltsnap.gui.worker")


class InferenceWorker(QThread):
    """识别推理 Worker"""

    finished = pyqtSignal(object)  # RecognitionResult
    error = pyqtSignal(str)

    def __init__(self, image_path: str, parent=None):
        super().__init__(parent)
        self.image_path = image_path

    def run(self):
        try:
            from voltsnap.recognition.pipeline import RecognitionPipeline

            pipeline = RecognitionPipeline()
            result = pipeline.process(self.image_path)
            self.finished.emit(result)
        except Exception as e:
            logger.error("Inference failed: %s", e, exc_info=True)
            self.error.emit(f"{e}\n{traceback.format_exc()}")


class SimulationWorker(QThread):
    """仿真 Worker"""

    finished = pyqtSignal(object)  # SimulationResult
    error = pyqtSignal(str)

    def __init__(self, netlist: str, parent=None):
        super().__init__(parent)
        self.netlist = netlist

    def run(self):
        try:
            from voltsnap.simulation.ngspice_runner import NgspiceRunner

            runner = NgspiceRunner(
                ngspice_path=Config.NGSPICE_PATH,
                timeout=Config.SIM_TIMEOUT,
            )
            result = runner.run(self.netlist)
            self.finished.emit(result)
        except Exception as e:
            logger.error("Simulation failed: %s", e, exc_info=True)
            self.error.emit(f"{e}\n{traceback.format_exc()}")
