from PyQt5.QtWidgets import QDoubleSpinBox, QSpinBox
from PyQt5.QtGui import QValidator
from pyxalign.api.constants import divisor

class NoScrollSpinBox(QSpinBox):
    def wheelEvent(self, event):
        self.setMinimum(0)
        self.setMaximum(1000000)  # Adjust as needed
        event.ignore()  # Prevent changing value on scroll


class NoScrollDoubleSpinBox(QDoubleSpinBox):
    def wheelEvent(self, event):
        event.ignore()  # Prevent changing value on scroll


class MinimalDecimalSpinBox(NoScrollDoubleSpinBox):
    def textFromValue(self, value):
        # Format to suppress trailing zeros, but respect min/max decimals
        text = f"{value:.10f}".rstrip("0").rstrip(".")
        if text == "-0":  # Optional: fix "-0" to "0"
            text = "0"
        return text

class MultipleOfDivisorSpinBox(QSpinBox):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimum(0)
        self.setMaximum(1000000)  # Set as needed
        self.setSingleStep(divisor)

    def valueFromText(self, text):
        try:
            val = int(text)
            return self._round_to_nearest_divisor(val)
        except ValueError:
            return self.value()  # fallback to current value on invalid input

    def textFromValue(self, value):
        return str(value)

    def _round_to_nearest_divisor(self, value):
        return round(value / divisor) * divisor