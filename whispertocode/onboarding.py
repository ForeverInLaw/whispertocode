from __future__ import annotations

from dataclasses import replace
from typing import Optional

from .config_store import AppSettings


def run_onboarding(initial: AppSettings) -> Optional[AppSettings]:
    try:
        from PySide6 import QtCore, QtGui, QtWidgets
    except Exception as exc:
        raise RuntimeError(f"Onboarding UI is unavailable: {exc}") from exc

    app = QtWidgets.QApplication.instance()
    if app is not None and app.thread() is not QtCore.QThread.currentThread():
        raise RuntimeError(
            "Onboarding cannot open while another Qt UI thread is active."
        )
    owns_app = app is None
    if owns_app:
        app = QtWidgets.QApplication([])
        app.setQuitOnLastWindowClosed(False)

    try:
        return run_onboarding_with_qt(QtCore, QtGui, QtWidgets, initial)
    finally:
        if owns_app:
            app.quit()


def run_onboarding_with_qt(qt_core, qt_gui, qt_widgets, initial: AppSettings) -> Optional[AppSettings]:
    app = qt_widgets.QApplication.instance()
    if app is not None:
        # Keep overlay/runtime alive when settings wizard is closed.
        app.setQuitOnLastWindowClosed(False)
    wizard = _OnboardingWizard(qt_core, qt_gui, qt_widgets, initial)
    result = wizard.exec()
    if result != int(qt_widgets.QDialog.Accepted):
        return None
    return wizard.collect_settings()


class _OnboardingWizard:
    def __init__(self, qt_core, qt_gui, qt_widgets, initial: AppSettings) -> None:
        self._qt_core = qt_core
        self._qt_gui = qt_gui
        self._qt_widgets = qt_widgets
        self._initial = initial
        self._existing_api_key = initial.nvidia_api_key.strip()

        wizard = qt_widgets.QWizard()
        wizard.setWindowTitle("WhisperToCode Setup")
        wizard.setOption(qt_widgets.QWizard.NoBackButtonOnStartPage, True)
        wizard.setOption(qt_widgets.QWizard.NoCancelButton, False)
        flags = (
            qt_core.Qt.Window
            | qt_core.Qt.WindowTitleHint
            | qt_core.Qt.WindowSystemMenuHint
            | qt_core.Qt.WindowCloseButtonHint
        )
        wizard.setWindowFlags(flags)
        wizard.setWindowModality(qt_core.Qt.ApplicationModal)
        wizard.setWizardStyle(qt_widgets.QWizard.ModernStyle)
        wizard.resize(760, 520)
        wizard.setMinimumSize(700, 480)
        wizard.setButtonText(qt_widgets.QWizard.CancelButton, "Cancel")
        wizard.setButtonText(qt_widgets.QWizard.FinishButton, "Save")

        self._wizard = wizard
        self._apply_visual_theme()
        self._build_pages()
        self._install_shortcuts()

    def _apply_visual_theme(self) -> None:
        self._wizard.setStyleSheet(
            """
            QWizard {
                background: #121214;
                color: rgba(255, 255, 255, 0.9);
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            }
            QWizardPage {
                background: transparent;
            }
            QLabel {
                color: rgba(255, 255, 255, 0.9);
                font-size: 14px;
            }
            QLabel#onboardingMeta {
                color: rgba(255, 255, 255, 0.5);
                font-size: 13px;
                letter-spacing: 0.3px;
                font-weight: 500;
            }
            QFrame#onboardingCard {
                background: #18181a;
                border: 1px solid rgba(255, 255, 255, 0.08);
                border-radius: 12px;
            }
            QLineEdit {
                background: #121214;
                border: 1px solid rgba(255, 255, 255, 0.12);
                border-radius: 8px;
                color: rgba(255, 255, 255, 0.9);
                padding: 10px 14px;
                font-size: 14px;
                selection-background-color: rgba(255, 255, 255, 0.2);
            }
            QLineEdit:focus {
                border: 1px solid rgba(255, 255, 255, 0.4);
                background: #1a1a1c;
            }
            QCheckBox {
                spacing: 12px;
                color: rgba(255, 255, 255, 0.9);
                font-size: 14px;
            }
            QCheckBox::indicator {
                width: 20px;
                height: 20px;
                border-radius: 6px;
                border: 1px solid rgba(255, 255, 255, 0.15);
                background: #121214;
            }
            QCheckBox::indicator:hover {
                border: 1px solid rgba(255, 255, 255, 0.3);
            }
            QCheckBox::indicator:checked {
                border: 1px solid rgba(255, 255, 255, 0.9);
                background: rgba(255, 255, 255, 0.2);
            }
            QPushButton {
                background: rgba(255, 255, 255, 0.05);
                color: rgba(255, 255, 255, 0.9);
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 8px;
                padding: 8px 20px;
                font-size: 14px;
                font-weight: 500;
                min-width: 100px;
            }
            QPushButton:hover {
                background: rgba(255, 255, 255, 0.1);
                border: 1px solid rgba(255, 255, 255, 0.2);
            }
            QPushButton:pressed {
                background: rgba(255, 255, 255, 0.15);
                border: 1px solid rgba(255, 255, 255, 0.3);
            }
            QPushButton#qt_wizard_nextbutton, QPushButton#qt_wizard_finishbutton {
                background: #ffffff;
                border: 1px solid #ffffff;
                color: #121214;
                font-weight: 600;
            }
            QPushButton#qt_wizard_nextbutton:hover, QPushButton#qt_wizard_finishbutton:hover {
                background: #e6e6e6;
                border: 1px solid #e6e6e6;
            }
            QPushButton#qt_wizard_nextbutton:pressed, QPushButton#qt_wizard_finishbutton:pressed {
                background: #cccccc;
                border: 1px solid #cccccc;
            }
            """
        )

    def _install_shortcuts(self) -> None:
        esc_shortcut = self._qt_gui.QShortcut(
            self._qt_gui.QKeySequence("Esc"),
            self._wizard,
        )
        esc_shortcut.activated.connect(self._wizard.reject)

    def _build_pages(self) -> None:
        qt_widgets = self._qt_widgets

        self._api_key_page = qt_widgets.QWizardPage()
        self._api_key_page.setTitle("Step 1: API Key")
        if self._existing_api_key:
            self._api_key_page.setSubTitle(
                "Enter a new NVIDIA API key, or leave blank to keep the current one."
            )
        else:
            self._api_key_page.setSubTitle("Enter your NVIDIA API key to continue.")
        key_layout = qt_widgets.QVBoxLayout()
        key_layout.setContentsMargins(0, 8, 0, 0)
        key_layout.setSpacing(12)
        key_meta = qt_widgets.QLabel("Secure connection setup")
        key_meta.setObjectName("onboardingMeta")
        key_meta.setWordWrap(True)
        key_layout.addWidget(key_meta)
        key_card = self._build_card()
        key_form = qt_widgets.QFormLayout()
        key_form.setContentsMargins(20, 18, 20, 20)
        key_form.setSpacing(10)
        self._key_input = qt_widgets.QLineEdit()
        self._key_input.setEchoMode(qt_widgets.QLineEdit.PasswordEchoOnEdit)
        self._key_input.setPlaceholderText("nvapi-...")
        self._api_key_page.registerField("nvidia_api_key", self._key_input)
        key_form.addRow("NVIDIA_API_KEY", self._key_input)
        if self._existing_api_key:
            keep_hint = qt_widgets.QLabel("Current key is configured. Leave this blank to keep it.")
            keep_hint.setObjectName("onboardingMeta")
            keep_hint.setWordWrap(True)
            key_form.addRow("", keep_hint)
        key_card.setLayout(key_form)
        key_layout.addWidget(key_card)
        key_layout.addStretch(1)
        self._api_key_page.setLayout(key_layout)

        self._mode_page = qt_widgets.QWizardPage()
        self._mode_page.setTitle("Step 2: Endpoint/Model Setup")
        self._mode_page.setSubTitle(
            "You can keep defaults or configure custom endpoints and models."
        )
        mode_layout = qt_widgets.QVBoxLayout()
        mode_layout.setContentsMargins(0, 8, 0, 0)
        mode_layout.setSpacing(12)
        mode_label = qt_widgets.QLabel(
            "Defaults work out of the box. Enable advanced setup if you want custom Riva/Nemotron values."
        )
        mode_label.setWordWrap(True)
        mode_label.setObjectName("onboardingMeta")
        self._customize_checkbox = qt_widgets.QCheckBox(
            "Customize endpoints and models"
        )
        self._customize_checkbox.setChecked(False)
        self._mode_page.registerField("customize_advanced", self._customize_checkbox)
        mode_card = self._build_card()
        mode_card_layout = qt_widgets.QVBoxLayout()
        mode_card_layout.setContentsMargins(20, 18, 20, 20)
        mode_card_layout.setSpacing(12)
        mode_card_layout.addWidget(mode_label)
        mode_card_layout.addWidget(self._customize_checkbox)
        mode_card_layout.addStretch(1)
        mode_card.setLayout(mode_card_layout)
        mode_layout.addWidget(mode_card)
        mode_layout.addStretch(1)
        self._mode_page.setLayout(mode_layout)

        self._riva_page = qt_widgets.QWizardPage()
        self._riva_page.setTitle("Step 3: Riva Endpoint")
        riva_layout = qt_widgets.QVBoxLayout()
        riva_layout.setContentsMargins(0, 8, 0, 0)
        riva_layout.setSpacing(12)
        riva_meta = qt_widgets.QLabel("Speech recognition backend")
        riva_meta.setObjectName("onboardingMeta")
        riva_card = self._build_card()
        riva_form = qt_widgets.QFormLayout()
        riva_form.setContentsMargins(20, 18, 20, 20)
        riva_form.setSpacing(10)
        self._riva_server_input = qt_widgets.QLineEdit(self._initial.riva_server)
        self._riva_function_input = qt_widgets.QLineEdit(self._initial.riva_function_id)
        riva_form.addRow("Riva server", self._riva_server_input)
        riva_form.addRow("Riva function ID", self._riva_function_input)
        riva_card.setLayout(riva_form)
        riva_layout.addWidget(riva_meta)
        riva_layout.addWidget(riva_card)
        riva_layout.addStretch(1)
        self._riva_page.setLayout(riva_layout)

        self._nemotron_page = qt_widgets.QWizardPage()
        self._nemotron_page.setTitle("Step 4: Nemotron Endpoint + Model")
        nem_layout = qt_widgets.QVBoxLayout()
        nem_layout.setContentsMargins(0, 8, 0, 0)
        nem_layout.setSpacing(12)
        nem_meta = qt_widgets.QLabel("SMART mode rewrite backend")
        nem_meta.setObjectName("onboardingMeta")
        nem_card = self._build_card()
        nem_form = qt_widgets.QFormLayout()
        nem_form.setContentsMargins(20, 18, 20, 20)
        nem_form.setSpacing(10)
        self._nem_base_url_input = qt_widgets.QLineEdit(self._initial.nemotron_base_url)
        self._nem_model_input = qt_widgets.QLineEdit(self._initial.nemotron_model)
        self._temperature_input = qt_widgets.QLineEdit(
            str(self._initial.nemotron_temperature)
        )
        self._top_p_input = qt_widgets.QLineEdit(str(self._initial.nemotron_top_p))
        self._max_tokens_input = qt_widgets.QLineEdit(
            str(self._initial.nemotron_max_tokens)
        )
        self._reasoning_budget_input = qt_widgets.QLineEdit(
            str(self._initial.nemotron_reasoning_budget)
        )
        self._reasoning_print_limit_input = qt_widgets.QLineEdit(
            str(self._initial.nemotron_reasoning_print_limit)
        )
        self._enable_thinking_checkbox = qt_widgets.QCheckBox("Enable thinking")
        self._enable_thinking_checkbox.setChecked(self._initial.nemotron_enable_thinking)
        nem_form.addRow("NEMOTRON_BASE_URL", self._nem_base_url_input)
        nem_form.addRow("NEMOTRON_MODEL", self._nem_model_input)
        nem_form.addRow("NEMOTRON_TEMPERATURE", self._temperature_input)
        nem_form.addRow("NEMOTRON_TOP_P", self._top_p_input)
        nem_form.addRow("NEMOTRON_MAX_TOKENS", self._max_tokens_input)
        nem_form.addRow("NEMOTRON_REASONING_BUDGET", self._reasoning_budget_input)
        nem_form.addRow(
            "NEMOTRON_REASONING_PRINT_LIMIT",
            self._reasoning_print_limit_input,
        )
        nem_form.addRow(self._enable_thinking_checkbox)
        nem_card.setLayout(nem_form)
        nem_layout.addWidget(nem_meta)
        nem_layout.addWidget(nem_card)
        nem_layout.addStretch(1)
        self._nemotron_page.setLayout(nem_layout)

        self._review_page = qt_widgets.QWizardPage()
        self._review_page.setTitle("Step 5: Review")
        self._review_page.setSubTitle("Confirm settings and click Finish to save.")
        review_layout = qt_widgets.QVBoxLayout()
        review_layout.setContentsMargins(0, 8, 0, 0)
        review_layout.setSpacing(12)
        review_meta = qt_widgets.QLabel("Final check before writing config.json")
        review_meta.setObjectName("onboardingMeta")
        review_card = self._build_card()
        review_card_layout = qt_widgets.QVBoxLayout()
        review_card_layout.setContentsMargins(20, 18, 20, 20)
        review_card_layout.setSpacing(10)
        self._review_label = qt_widgets.QLabel("")
        self._review_label.setWordWrap(True)
        self._review_label.setTextInteractionFlags(self._qt_core.Qt.TextSelectableByMouse)
        review_card_layout.addWidget(self._review_label)
        review_card.setLayout(review_card_layout)
        review_layout.addWidget(review_meta)
        review_layout.addWidget(review_card)
        review_layout.addStretch(1)
        self._review_page.setLayout(review_layout)

        self._wizard.setPage(0, self._api_key_page)
        self._wizard.setPage(1, self._mode_page)
        self._wizard.setPage(2, self._riva_page)
        self._wizard.setPage(3, self._nemotron_page)
        self._wizard.setPage(4, self._review_page)
        self._wizard.setStartId(0)

        self._mode_page.nextId = self._mode_next_id  # type: ignore[method-assign]
        self._riva_page.nextId = self._riva_next_id  # type: ignore[method-assign]
        self._nemotron_page.nextId = self._nem_next_id  # type: ignore[method-assign]
        self._review_page.initializePage = self._init_review_page  # type: ignore[method-assign]

        self._api_key_page.validatePage = self._validate_api_key_page  # type: ignore[method-assign]
        self._riva_page.validatePage = self._validate_riva_page  # type: ignore[method-assign]
        self._nemotron_page.validatePage = self._validate_nemotron_page  # type: ignore[method-assign]

    def _build_card(self):
        card = self._qt_widgets.QFrame()
        card.setObjectName("onboardingCard")
        return card

    def _mode_next_id(self) -> int:
        return 2 if self._customize_checkbox.isChecked() else 4

    def _validate_api_key_page(self) -> bool:
        if self._key_input.text().strip() or self._existing_api_key:
            return True
        self._show_invalid("NVIDIA_API_KEY cannot be empty.")
        return False

    @staticmethod
    def _riva_next_id() -> int:
        return 3

    @staticmethod
    def _nem_next_id() -> int:
        return 4

    def _validate_riva_page(self) -> bool:
        server = self._riva_server_input.text().strip()
        function_id = self._riva_function_input.text().strip()
        if server and function_id:
            return True
        self._qt_widgets.QMessageBox.warning(
            self._wizard,
            "Validation",
            "Riva server and function ID cannot be empty.",
        )
        return False

    def _validate_nemotron_page(self) -> bool:
        fields = [
            ("NEMOTRON_BASE_URL", self._nem_base_url_input.text().strip()),
            ("NEMOTRON_MODEL", self._nem_model_input.text().strip()),
        ]
        for label, value in fields:
            if not value:
                self._qt_widgets.QMessageBox.warning(
                    self._wizard,
                    "Validation",
                    f"{label} cannot be empty.",
                )
                return False

        if _parse_float(self._temperature_input.text()) is None:
            self._show_invalid("NEMOTRON_TEMPERATURE must be a valid number.")
            return False
        top_p = _parse_float(self._top_p_input.text())
        if top_p is None or top_p < 0.0 or top_p > 1.0:
            self._show_invalid("NEMOTRON_TOP_P must be in range 0..1.")
            return False
        if _parse_int(self._max_tokens_input.text()) is None:
            self._show_invalid("NEMOTRON_MAX_TOKENS must be a valid integer.")
            return False
        if _parse_int(self._reasoning_budget_input.text()) is None:
            self._show_invalid("NEMOTRON_REASONING_BUDGET must be a valid integer.")
            return False
        if _parse_int(self._reasoning_print_limit_input.text()) is None:
            self._show_invalid(
                "NEMOTRON_REASONING_PRINT_LIMIT must be a valid integer."
            )
            return False
        return True

    def _show_invalid(self, message: str) -> None:
        self._qt_widgets.QMessageBox.warning(self._wizard, "Validation", message)

    def _init_review_page(self) -> None:
        settings = self.collect_settings()
        key_status = "configured" if settings.nvidia_api_key else "missing"
        self._review_label.setText(
            (
                f"API key: {key_status}\n"
                f"Riva server: {settings.riva_server}\n"
                f"Riva function ID: {settings.riva_function_id}\n"
                f"Nemotron URL: {settings.nemotron_base_url}\n"
                f"Nemotron model: {settings.nemotron_model}\n"
                f"Temperature / top_p: {settings.nemotron_temperature} / {settings.nemotron_top_p}\n"
                f"Max tokens: {settings.nemotron_max_tokens}\n"
                f"Reasoning budget: {settings.nemotron_reasoning_budget}\n"
                f"Reasoning print limit: {settings.nemotron_reasoning_print_limit}\n"
                f"Enable thinking: {settings.nemotron_enable_thinking}"
            )
        )

    def exec(self) -> int:
        return int(self._wizard.exec())

    def collect_settings(self) -> AppSettings:
        entered_key = self._key_input.text().strip()
        key = entered_key if entered_key else self._existing_api_key
        customize = self._customize_checkbox.isChecked()
        if not customize:
            return replace(self._initial, nvidia_api_key=key)

        temperature = _parse_float(self._temperature_input.text())
        top_p = _parse_float(self._top_p_input.text())
        max_tokens = _parse_int(self._max_tokens_input.text())
        reasoning_budget = _parse_int(self._reasoning_budget_input.text())
        reasoning_print_limit = _parse_int(self._reasoning_print_limit_input.text())

        return AppSettings(
            nvidia_api_key=key,
            riva_server=self._riva_server_input.text().strip(),
            riva_function_id=self._riva_function_input.text().strip(),
            nemotron_base_url=self._nem_base_url_input.text().strip(),
            nemotron_model=self._nem_model_input.text().strip(),
            nemotron_temperature=temperature if temperature is not None else 1.0,
            nemotron_top_p=top_p if top_p is not None else 1.0,
            nemotron_max_tokens=max_tokens if max_tokens is not None else 16384,
            nemotron_reasoning_budget=(
                reasoning_budget if reasoning_budget is not None else 4096
            ),
            nemotron_reasoning_print_limit=(
                reasoning_print_limit if reasoning_print_limit is not None else 600
            ),
            nemotron_enable_thinking=self._enable_thinking_checkbox.isChecked(),
        )


def _parse_float(raw: str) -> Optional[float]:
    try:
        return float(raw.strip())
    except (TypeError, ValueError):
        return None


def _parse_int(raw: str) -> Optional[int]:
    try:
        return int(raw.strip())
    except (TypeError, ValueError):
        return None
