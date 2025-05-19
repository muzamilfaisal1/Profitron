"""
About window in the GUI.
"""

from PyQt5.QtWidgets import QDialog, QHBoxLayout, QMainWindow, QPlainTextEdit

import profitron


class About(QDialog):
    """
    About window.
    """
    def __init__(self, parent: QMainWindow = None):
        super(About, self).__init__(parent)
        layout = QHBoxLayout()

        description = "profitron\n\n" \
                      f"profitron is an open-source software written in Python that enables users to create automated " \
                      f"bots that can trade, simulate, optimize, or backtest with strategies implemented.\n\n" \
                      f"This program is not responsible for any financial burdens or debt incurred.\n\n" \
                      f"Use at your own risk.\n\n" 

        plain_text_edit = QPlainTextEdit()
        plain_text_edit.setPlainText(description)
        plain_text_edit.setReadOnly(True)
        layout.addWidget(plain_text_edit)

        self.setWindowTitle('About profitron')
        self.setLayout(layout)
        self.setFixedSize(plain_text_edit.size())
