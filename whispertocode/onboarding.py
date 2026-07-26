from __future__ import annotations

from dataclasses import replace
from typing import Optional

from .config_store import AppSettings, get_config_path, normalize_stt_backend
from .constants import (
    LOCAL_MODELS,
    NEMOTRON_REASONING_BUDGET_MAX,
    NEMOTRON_REASONING_PRINT_LIMIT_MAX,
    STT_BACKEND_LOCAL,
    STT_BACKEND_RIVA,
)
from .local_asr import default_model_dir


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


def _validating_page_class(qt_widgets):
    """Build the QWizardPage subclass shared by every page that validates.

    A real subclass rather than the instance-attribute assignment the other
    pages use for ``nextId``: overriding a C++ virtual that way only works
    because Shiboken looks the name up on the wrapper object, and a page that
    also has to carry per-page state (its error label, whether the user has
    touched it) reads better as a type than as three bolted-on attributes.
    The Qt module only exists at call time, hence a factory.
    """

    class _ValidatingPage(qt_widgets.QWizardPage):
        def __init__(self) -> None:
            super().__init__()
            self._problems = list
            self._error_label = None
            self._touched = False

        def bind(self, problems, error_label, signals) -> None:
            """Drive Next off ``problems()`` and report it in ``error_label``."""
            self._problems = problems
            self._error_label = error_label
            for signal in signals:
                signal.connect(self._on_input_changed)
            self._refresh()

        def isComplete(self) -> bool:
            return not self._problems()

        def _on_input_changed(self, *_args) -> None:
            self._touched = True
            self._refresh()
            self.completeChanged.emit()

        def _refresh(self) -> None:
            # Stays quiet until the user has actually edited this page: a blank
            # first-run form should read as unfinished, not as a mistake. Next
            # is already disabled, so nothing invalid can be submitted anyway.
            problems = self._problems() if self._touched else ()
            self._error_label.setText("\n".join(problems))
            self._error_label.setVisible(bool(problems))

    return _ValidatingPage


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
        qt_gui = self._qt_gui
        background = qt_gui.QColor("#121214")
        surface = qt_gui.QColor("#18181a")
        text = qt_gui.QColor("#e8e8ea")
        muted = qt_gui.QColor("#7a7a82")
        palette = qt_gui.QPalette()
        palette.setColor(qt_gui.QPalette.Window, background)
        palette.setColor(qt_gui.QPalette.Base, background)
        palette.setColor(qt_gui.QPalette.Button, surface)
        palette.setColor(qt_gui.QPalette.WindowText, text)
        palette.setColor(qt_gui.QPalette.Text, text)
        palette.setColor(qt_gui.QPalette.ButtonText, text)
        palette.setColor(qt_gui.QPalette.PlaceholderText, muted)
        palette.setColor(qt_gui.QPalette.Highlight, qt_gui.QColor("#3d3d40"))
        palette.setColor(qt_gui.QPalette.HighlightedText, text)
        self._chrome_palette = palette
        # ModernStyle paints its title/subtitle band from the palette, never
        # from the stylesheet. Without an explicit dark palette that band keeps
        # the system's colours, so on a light desktop the header text below
        # renders white on white.
        self._wizard.setPalette(palette)
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
            /* Scoped to page bodies: a bare QLabel rule also hits the wizard's
               own header title/subtitle, forcing them white over a header band
               the stylesheet cannot reach and overriding Qt's header font. */
            QWizardPage QLabel {
                color: rgba(255, 255, 255, 0.9);
                font-size: 14px;
            }
            QWizardPage QLabel#onboardingMeta {
                color: rgba(255, 255, 255, 0.5);
                font-size: 13px;
                letter-spacing: 0.3px;
                font-weight: 500;
            }
            /* Louder than a caption on purpose: this is the only place a
               blocked page explains itself now that the popup is gone. */
            QWizardPage QLabel#onboardingError {
                color: #ff8a8a;
                font-size: 13px;
                font-weight: 600;
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
                placeholder-text-color: #7a7a82;
                padding: 10px 14px;
                font-size: 14px;
                selection-background-color: rgba(255, 255, 255, 0.2);
            }
            QLineEdit:focus {
                border: 1px solid rgba(255, 255, 255, 0.4);
                background: #1a1a1c;
            }
            QSpinBox, QDoubleSpinBox {
                background: #121214;
                border: 1px solid rgba(255, 255, 255, 0.12);
                border-radius: 8px;
                color: rgba(255, 255, 255, 0.9);
                padding: 10px 14px;
                font-size: 14px;
                selection-background-color: rgba(255, 255, 255, 0.2);
            }
            QSpinBox:focus, QDoubleSpinBox:focus {
                border: 1px solid rgba(255, 255, 255, 0.4);
                background: #1a1a1c;
            }
            /* Unstyled ::up-button / ::down-button keep the native light
               chrome, which reads as two grey chips on the dark field. */
            QSpinBox::up-button, QDoubleSpinBox::up-button,
            QSpinBox::down-button, QDoubleSpinBox::down-button {
                subcontrol-origin: border;
                width: 22px;
                background: rgba(255, 255, 255, 0.06);
                border: none;
                border-left: 1px solid rgba(255, 255, 255, 0.12);
                margin: 1px;
            }
            QSpinBox::up-button, QDoubleSpinBox::up-button {
                subcontrol-position: top right;
                border-top-right-radius: 7px;
            }
            QSpinBox::down-button, QDoubleSpinBox::down-button {
                subcontrol-position: bottom right;
                border-bottom-right-radius: 7px;
            }
            QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
            QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {
                background: rgba(255, 255, 255, 0.12);
            }
            QSpinBox::up-button:pressed, QDoubleSpinBox::up-button:pressed,
            QSpinBox::down-button:pressed, QDoubleSpinBox::down-button:pressed {
                background: rgba(255, 255, 255, 0.2);
            }
            /* Arrows drawn as CSS border triangles: Qt stylesheets can only
               load arrow images from files or :/resources, and this dialog
               ships neither. */
            QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {
                width: 0;
                height: 0;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-bottom: 5px solid rgba(255, 255, 255, 0.75);
            }
            QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {
                width: 0;
                height: 0;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-top: 5px solid rgba(255, 255, 255, 0.75);
            }
            QSpinBox::up-arrow:off, QDoubleSpinBox::up-arrow:off {
                border-bottom: 5px solid rgba(255, 255, 255, 0.25);
            }
            QSpinBox::down-arrow:off, QDoubleSpinBox::down-arrow:off {
                border-top: 5px solid rgba(255, 255, 255, 0.25);
            }
            QComboBox {
                background: #121214;
                border: 1px solid rgba(255, 255, 255, 0.12);
                border-radius: 8px;
                color: rgba(255, 255, 255, 0.9);
                padding: 10px 14px;
                font-size: 14px;
            }
            QComboBox:focus {
                border: 1px solid rgba(255, 255, 255, 0.4);
                background: #1a1a1c;
            }
            QComboBox QAbstractItemView {
                background: #18181a;
                border: 1px solid rgba(255, 255, 255, 0.12);
                color: rgba(255, 255, 255, 0.9);
                selection-background-color: rgba(255, 255, 255, 0.2);
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
        # A stylesheet makes QStyleSheetStyle re-seed every styled descendant
        # from the *application* palette, so the palette set above never
        # reaches the header band. Qt builds that band lazily on first show,
        # hence repainting the chrome whenever the page changes.
        self._wizard.currentIdChanged.connect(self._apply_chrome_palette)

    def _apply_chrome_palette(self) -> None:
        # Plain QWidget children are wizard chrome only: the ModernStyle header
        # band and its containers. Pages and real controls are subclasses and
        # keep the stylesheet's own colours.
        widget_cls = self._qt_widgets.QWidget
        label_cls = self._qt_widgets.QLabel
        for child in self._wizard.findChildren(widget_cls):
            if type(child) is not widget_cls:
                continue
            child.setPalette(self._chrome_palette)
            for grandchild in child.children():
                if isinstance(grandchild, label_cls):
                    grandchild.setPalette(self._chrome_palette)

    def _install_shortcuts(self) -> None:
        esc_shortcut = self._qt_gui.QShortcut(
            self._qt_gui.QKeySequence("Esc"),
            self._wizard,
        )
        esc_shortcut.activated.connect(self._wizard.reject)

    def _build_pages(self) -> None:
        qt_widgets = self._qt_widgets
        validating_page = _validating_page_class(qt_widgets)

        self._api_key_page = validating_page()
        self._api_key_page.setTitle("Speech backend")
        if self._existing_api_key:
            self._api_key_page.setSubTitle(
                "Choose how your speech is transcribed. The key you saved earlier stays in place."
            )
        else:
            self._api_key_page.setSubTitle(
                "Choose how your speech is transcribed: in the cloud with an NVIDIA API key, "
                "or offline on this machine."
            )
        key_layout = qt_widgets.QVBoxLayout()
        key_layout.setContentsMargins(0, 8, 0, 0)
        key_layout.setSpacing(12)
        key_card = self._build_card()
        key_form = qt_widgets.QFormLayout()
        key_form.setContentsMargins(20, 18, 20, 20)
        key_form.setSpacing(10)
        self._backend_combo = qt_widgets.QComboBox()
        self._backend_combo.addItem("NVIDIA Riva (cloud)", STT_BACKEND_RIVA)
        self._backend_combo.addItem("whisper.cpp (local)", STT_BACKEND_LOCAL)
        initial_backend = normalize_stt_backend(self._initial.stt_backend)
        self._backend_combo.setCurrentIndex(
            1 if initial_backend == STT_BACKEND_LOCAL else 0
        )
        # Not editable: a typo here would only surface as an upstream download
        # error minutes later, on the first dictation. config.json is the
        # escape hatch for models not listed.
        self._local_model_combo = qt_widgets.QComboBox()
        for name, size in LOCAL_MODELS:
            self._local_model_combo.addItem(f"{name} — {size}", name)
        initial_model_index = self._local_model_combo.findData(self._initial.local_model)
        if initial_model_index >= 0:
            self._local_model_combo.setCurrentIndex(initial_model_index)
        self._local_model_dir_input = qt_widgets.QLineEdit(self._initial.local_model_dir)
        self._local_model_dir_input.setPlaceholderText(str(default_model_dir()))
        key_form.addRow("Transcribe with", self._backend_combo)
        key_form.addRow("Local model", self._local_model_combo)
        key_form.addRow(
            "",
            self._build_caption(
                "Downloaded the first time you dictate, not now. Reused after that."
            ),
        )
        key_form.addRow("Model folder", self._local_model_dir_input)
        self._key_input = qt_widgets.QLineEdit()
        self._key_input.setEchoMode(qt_widgets.QLineEdit.PasswordEchoOnEdit)
        self._key_input.setPlaceholderText("nvapi-...")
        self._api_key_page.registerField("nvidia_api_key", self._key_input)
        key_form.addRow("NVIDIA API key", self._key_input)
        if self._existing_api_key:
            key_caption = "NVIDIA_API_KEY — a key is already saved. Leave this blank to keep it."
        else:
            key_caption = (
                "NVIDIA_API_KEY — required for cloud speech. Local speech works without it, "
                "unless you also want the app to tidy up what you dictated."
            )
        key_form.addRow("", self._build_caption(key_caption))
        self._key_error = self._build_error()
        key_form.addRow("", self._key_error)
        key_card.setLayout(key_form)
        key_layout.addWidget(key_card)
        key_layout.addStretch(1)
        self._api_key_page.setLayout(key_layout)
        self._api_key_page.bind(
            self._api_key_problems,
            self._key_error,
            (self._key_input.textChanged, self._backend_combo.currentIndexChanged),
        )

        self._mode_page = qt_widgets.QWizardPage()
        self._mode_page.setTitle("Advanced setup")
        self._mode_page.setSubTitle(
            "Optional. The defaults already point at NVIDIA's hosted services."
        )
        mode_layout = qt_widgets.QVBoxLayout()
        mode_layout.setContentsMargins(0, 8, 0, 0)
        mode_layout.setSpacing(12)
        mode_label = self._build_caption(
            "Turn this on only if you need to send speech somewhere else, use a different "
            "cleanup model, or change how that model writes."
        )
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

        self._riva_page = validating_page()
        self._riva_page.setTitle("Riva endpoint")
        riva_layout = qt_widgets.QVBoxLayout()
        riva_layout.setContentsMargins(0, 8, 0, 0)
        riva_layout.setSpacing(12)
        riva_meta = self._build_caption(
            "Where cloud speech is sent. Change these only if NVIDIA gave you a different endpoint."
        )
        riva_card = self._build_card()
        riva_form = qt_widgets.QFormLayout()
        riva_form.setContentsMargins(20, 18, 20, 20)
        riva_form.setSpacing(10)
        self._riva_server_input = qt_widgets.QLineEdit(self._initial.riva_server)
        self._riva_function_input = qt_widgets.QLineEdit(self._initial.riva_function_id)
        riva_form.addRow("Server address", self._riva_server_input)
        riva_form.addRow("", self._build_caption("RIVA_SERVER"))
        riva_form.addRow("Function ID", self._riva_function_input)
        riva_form.addRow("", self._build_caption("RIVA_FUNCTION_ID"))
        self._riva_error = self._build_error()
        riva_form.addRow("", self._riva_error)
        riva_card.setLayout(riva_form)
        riva_layout.addWidget(riva_meta)
        riva_layout.addWidget(riva_card)
        riva_layout.addStretch(1)
        self._riva_page.setLayout(riva_layout)
        self._riva_page.bind(
            self._riva_problems,
            self._riva_error,
            (
                self._riva_server_input.textChanged,
                self._riva_function_input.textChanged,
            ),
        )

        self._nemotron_page = validating_page()
        self._nemotron_page.setTitle("Cleanup model")
        nem_layout = qt_widgets.QVBoxLayout()
        nem_layout.setContentsMargins(0, 8, 0, 0)
        nem_layout.setSpacing(12)
        nem_meta = self._build_caption(
            "The model that strips filler words and fixes punctuation before the text is typed. "
            "It only runs when you switch the tray menu to SMART mode, and it always needs the API key."
        )
        nem_card = self._build_card()
        nem_form = qt_widgets.QFormLayout()
        nem_form.setContentsMargins(20, 18, 20, 20)
        nem_form.setSpacing(10)
        self._nem_base_url_input = qt_widgets.QLineEdit(self._initial.nemotron_base_url)
        self._nem_model_input = qt_widgets.QLineEdit(self._initial.nemotron_model)
        # Bounded so the wizard cannot offer a value the app would refuse or
        # silently clamp; the two reasoning fields reuse the app's own maxima.
        self._temperature_spin = self._build_decimal_field(
            self._initial.nemotron_temperature, 0.0, 2.0, 0.05
        )
        self._top_p_spin = self._build_decimal_field(
            self._initial.nemotron_top_p, 0.0, 1.0, 0.05
        )
        self._max_tokens_spin = self._build_whole_field(
            self._initial.nemotron_max_tokens, 1, 131072, 256
        )
        self._reasoning_budget_spin = self._build_whole_field(
            self._initial.nemotron_reasoning_budget,
            0,
            NEMOTRON_REASONING_BUDGET_MAX,
            128,
        )
        self._reasoning_print_limit_spin = self._build_whole_field(
            self._initial.nemotron_reasoning_print_limit,
            0,
            NEMOTRON_REASONING_PRINT_LIMIT_MAX,
            100,
        )
        self._enable_thinking_checkbox = qt_widgets.QCheckBox(
            "Let the model think before it rewrites"
        )
        self._enable_thinking_checkbox.setChecked(self._initial.nemotron_enable_thinking)
        nem_form.addRow("API endpoint", self._nem_base_url_input)
        nem_form.addRow("", self._build_caption("NEMOTRON_BASE_URL"))
        nem_form.addRow("Model", self._nem_model_input)
        nem_form.addRow("", self._build_caption("NEMOTRON_MODEL"))
        nem_form.addRow("Temperature", self._temperature_spin)
        nem_form.addRow("Top-p", self._top_p_spin)
        nem_form.addRow(
            "",
            self._build_caption(
                "Narrows word choice to the most likely options. 1.0 leaves it wide open."
            ),
        )
        nem_form.addRow("Response token limit", self._max_tokens_spin)
        nem_form.addRow("Thinking budget", self._reasoning_budget_spin)
        nem_form.addRow(
            "",
            self._build_caption(
                "Tokens the model may spend thinking before it answers."
            ),
        )
        nem_form.addRow("Thinking preview limit", self._reasoning_print_limit_spin)
        nem_form.addRow(
            "",
            self._build_caption(
                "Debug console only. It never changes the text that gets typed."
            ),
        )
        nem_form.addRow(self._enable_thinking_checkbox)
        self._nem_error = self._build_error()
        nem_form.addRow("", self._nem_error)
        nem_card.setLayout(nem_form)
        nem_layout.addWidget(nem_meta)
        nem_layout.addWidget(nem_card)
        nem_layout.addStretch(1)
        self._nemotron_page.setLayout(nem_layout)
        self._nemotron_page.bind(
            self._nemotron_problems,
            self._nem_error,
            (
                self._nem_base_url_input.textChanged,
                self._nem_model_input.textChanged,
            ),
        )

        self._review_page = qt_widgets.QWizardPage()
        self._review_page.setTitle("Review")
        self._review_page.setSubTitle("Confirm your settings, then click Save.")
        review_layout = qt_widgets.QVBoxLayout()
        review_layout.setContentsMargins(0, 8, 0, 0)
        review_layout.setSpacing(12)
        review_meta = self._build_caption("Final check before writing config.json")
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

    def _build_card(self):
        card = self._qt_widgets.QFrame()
        card.setObjectName("onboardingCard")
        return card

    def _build_caption(self, text: str):
        """Dim secondary line: field captions, env-var names, page hints."""
        caption = self._qt_widgets.QLabel(text)
        caption.setObjectName("onboardingMeta")
        caption.setWordWrap(True)
        return caption

    def _build_error(self):
        """Inline problem list for one card; hidden while the page is valid."""
        label = self._qt_widgets.QLabel("")
        label.setObjectName("onboardingError")
        label.setWordWrap(True)
        label.setVisible(False)
        return label

    def _build_whole_field(self, value: int, low: int, high: int, step: int):
        """Whole-number field whose range makes the invalid states untypable."""
        box = self._qt_widgets.QSpinBox()
        box.setRange(low, high)
        box.setSingleStep(step)
        box.setValue(int(value))
        return box

    def _build_decimal_field(self, value: float, low: float, high: float, step: float):
        """Same, for the sampling knobs that are fractions."""
        box = self._qt_widgets.QDoubleSpinBox()
        box.setDecimals(2)
        box.setRange(low, high)
        box.setSingleStep(step)
        box.setValue(float(value))
        return box

    def _mode_next_id(self) -> int:
        if not self._customize_checkbox.isChecked():
            return 4
        # Skip the Riva page for local speech: it validates a cloud server and
        # function ID that a local-only setup will never contact, and blocks
        # Next when they are blank.
        if self._selected_backend() == STT_BACKEND_LOCAL:
            return 3
        return 2

    def _selected_backend(self) -> str:
        return normalize_stt_backend(self._backend_combo.currentData())

    def _api_key_problems(self) -> list[str]:
        if self._key_input.text().strip() or self._existing_api_key:
            return []
        if self._selected_backend() == STT_BACKEND_LOCAL:
            return []
        return [
            "Enter an NVIDIA API key, or choose whisper.cpp (local) to transcribe "
            "offline without one."
        ]

    @staticmethod
    def _riva_next_id() -> int:
        return 3

    @staticmethod
    def _nem_next_id() -> int:
        return 4

    def _riva_problems(self) -> list[str]:
        server = self._riva_server_input.text().strip()
        function_id = self._riva_function_input.text().strip()
        if server and function_id:
            return []
        return ["Fill in both the server address and the function ID."]

    def _nemotron_problems(self) -> list[str]:
        fields = [
            ("an API endpoint", self._nem_base_url_input.text().strip()),
            ("a model name", self._nem_model_input.text().strip()),
        ]
        # Every blank field at once: the popup this replaced named only the
        # first, so a page with both empty took two rounds to get past.
        # The numeric fields need no check: their spin boxes cannot hold a value
        # outside the range the app accepts.
        return [
            f"The cleanup model needs {label}."
            for label, value in fields
            if not value
        ]

    def _init_review_page(self) -> None:
        # Only what this configuration will actually use: a local-only setup has
        # no reason to read back a Riva endpoint or Nemotron sampling params.
        settings = self.collect_settings()
        lines = []
        if normalize_stt_backend(settings.stt_backend) == STT_BACKEND_LOCAL:
            size = dict(LOCAL_MODELS).get(settings.local_model)
            size_note = f" — {size}, downloads on your first dictation" if size else ""
            lines.append(f"Speech: whisper.cpp on this machine, {settings.local_model}{size_note}")
            lines.append(f"Models stored in {settings.local_model_dir or default_model_dir()}")
        else:
            lines.append(f"Speech: NVIDIA Riva cloud, {settings.riva_server}")

        if settings.nvidia_api_key:
            lines.append(
                f"Transcript cleanup: ready with {settings.nemotron_model}, "
                "switch it on from the tray menu"
            )
        else:
            lines.append("Transcript cleanup: needs an NVIDIA API key before it can run")

        lines.append(f"Saved to {get_config_path()}")
        self._review_label.setText("\n".join(lines))

    def exec(self) -> int:
        return int(self._wizard.exec())

    def collect_settings(self) -> AppSettings:
        entered_key = self._key_input.text().strip()
        key = entered_key if entered_key else self._existing_api_key
        backend = self._selected_backend()
        local_model = self._local_model_combo.currentData()
        local_model_dir = self._local_model_dir_input.text().strip()
        customize = self._customize_checkbox.isChecked()
        if not customize:
            return replace(
                self._initial,
                nvidia_api_key=key,
                stt_backend=backend,
                local_model=local_model,
                local_model_dir=local_model_dir,
            )

        # replace(), not AppSettings(), so a field added later is carried over
        # instead of being silently reset for everyone who ticks Customize.
        return replace(
            self._initial,
            nvidia_api_key=key,
            stt_backend=backend,
            local_model=local_model,
            local_model_dir=local_model_dir,
            riva_server=self._riva_server_input.text().strip(),
            riva_function_id=self._riva_function_input.text().strip(),
            nemotron_base_url=self._nem_base_url_input.text().strip(),
            nemotron_model=self._nem_model_input.text().strip(),
            nemotron_temperature=self._temperature_spin.value(),
            nemotron_top_p=self._top_p_spin.value(),
            nemotron_max_tokens=self._max_tokens_spin.value(),
            nemotron_reasoning_budget=self._reasoning_budget_spin.value(),
            nemotron_reasoning_print_limit=self._reasoning_print_limit_spin.value(),
            nemotron_enable_thinking=self._enable_thinking_checkbox.isChecked(),
        )
