from PyQt5.QtWidgets import QDoubleSpinBox, QSpinBox
from PyQt5.QtGui import QValidator, QRegExpValidator
from PyQt5.QtCore import QRegExp
from pyxalign.api.constants import divisor

action_button_style_sheet = "QPushButton { background-color: green; font-weight: bold; font-size: 11pt; color: white; padding: 2px 6px;}"

class NoScrollSpinBox(QSpinBox):
    def wheelEvent(self, event):
        self.setMinimum(0)
        self.setMaximum(1000000)  # Adjust as needed
        event.ignore()  # Prevent changing value on scroll


class NoScrollDoubleSpinBox(QDoubleSpinBox):
    def wheelEvent(self, event):
        event.ignore()  # Prevent changing value on scroll


class CustomDoubleSpinBox(NoScrollDoubleSpinBox):
    def __init__(self, parent=None):
        super().__init__(parent)
        # Set up validator for decimal input including scientific notation
        # Pattern allows: optional minus, digits, optional decimal point with digits,
        # optional 'e' or 'E' followed by optional +/- and digits
        validator = QRegExpValidator(QRegExp(r"^-?\d*\.?\d*([eE][+-]?\d+)?$"))
        self.lineEdit().setValidator(validator)

    def textFromValue(self, value):
        # Return scientific notation if less than 1e-3
        if (abs(value) < 0.001 or abs(value) > 10000) and value != 0:
            # Use more precision in scientific notation, then strip trailing zeros
            sci_text = f"{value:.10e}"
            # Split into mantissa and exponent parts
            mantissa, exponent = sci_text.split('e')
            # Strip trailing zeros from mantissa
            mantissa = mantissa.rstrip('0').rstrip('.')
            return f"{mantissa}e{exponent}"

        # Format to suppress trailing zeros, but respect min/max decimals
        text = f"{value:.10f}".rstrip("0").rstrip(".")
        if text == "-0":  # Optional: fix "-0" to "0"
            text = "0"
        return text

    def valueFromText(self, text):
        try:
            return float(text)
        except ValueError:
            return 0.0  # or raise an error if preferred

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
