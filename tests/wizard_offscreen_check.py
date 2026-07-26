"""Drives the real onboarding wizard offscreen and asserts its behaviour.

Not a unittest module: it needs PySide6 and a real QApplication, which the
rest of the suite deliberately stubs out. Run it on its own:

    cd <repo root>
    PYTHONPATH=. QT_QPA_PLATFORM=offscreen python tests/wizard_offscreen_check.py < /dev/null

Exits non-zero and lists the failures if anything regressed.

Three rules learned the hard way:
  * feed it /dev/null - Qt hangs if stdin is a pipe holding the script
  * QMessageBox blocks forever headless. The wizard no longer opens one
    (validation is inline), but the stub below stays as a tripwire:
    POPUPS must end up empty.
  * QWizard does not attach a page to its widget tree until the page is
    visited, so anything measured on a rendered page needs show() plus
    real navigation. Click the buttons; do not call nextId() directly.
"""

import faulthandler
import sys
import types

faulthandler.dump_traceback_later(40, exit=True)


def stub(name, **attrs):
    m = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(m, k, v)
    sys.modules[name] = m
    return m


riva = stub("riva")
riva.client = stub(
    "riva.client",
    Auth=object,
    ASRService=object,
    RecognitionConfig=object,
    AudioEncoding=types.SimpleNamespace(LINEAR_PCM="LINEAR_PCM"),
)
stub("sounddevice", InputStream=object)
stub("dotenv", load_dotenv=lambda *a, **k: None)
pk = stub("pynput")
pk.keyboard = stub(
    "pynput.keyboard",
    Controller=object,
    Listener=object,
    KeyCode=object,
    Key=types.SimpleNamespace(
        shift="shift", shift_l="shift_l", shift_r="shift_r", esc="esc",
        ctrl="ctrl", ctrl_l="ctrl_l", ctrl_r="ctrl_r", left="left", right="right",
    ),
)

from PySide6 import QtCore, QtGui, QtWidgets  # noqa: E402

from whispertocode.config_store import AppSettings  # noqa: E402
from whispertocode import onboarding as ob  # noqa: E402

app = QtWidgets.QApplication([])

# Modal popups never return without a human; record them instead.
POPUPS = []
QtWidgets.QMessageBox.warning = staticmethod(
    lambda *a, **k: POPUPS.append(a[2] if len(a) > 2 else a)
)
QtWidgets.QMessageBox.information = staticmethod(
    lambda *a, **k: POPUPS.append(a[2] if len(a) > 2 else a)
)


def build(mode=ob.MODE_SETUP, **kwargs):
    POPUPS.clear()
    return ob._OnboardingWizard(QtCore, QtGui, QtWidgets, AppSettings(**kwargs), mode)


def check(label, condition, detail=""):
    status = "ok  " if condition else "FAIL"
    print(f"  [{status}] {label}{(' -> ' + str(detail)) if detail else ''}")
    if not condition:
        FAILURES.append(label)


FAILURES = []

print("== local backend, no key ==")
w = build(nvidia_api_key="", stt_backend="local", local_model="small-q5_1")
titles = [w._wizard.page(i).title() for i in range(w._wizard.pageIds()[-1] + 1)
          if w._wizard.page(i) is not None]
print("  titles:", titles)
check("no 'Step N' numbering in titles", not any("Step" in t for t in titles))
check("pages are exactly the four that survive, in route order",
      titles == ["Speech backend", "Riva endpoint", "Cleanup model", "Review"], titles)
check("the Advanced setup page is gone", "Advanced setup" not in titles)
check("page ids are contiguous from zero", w._wizard.pageIds() == [0, 1, 2, 3],
      w._wizard.pageIds())
check("start id falls out of the ids, no setStartId needed",
      w._wizard.startId() == 0, w._wizard.startId())
check("model combo is not free text", not w._local_model_combo.isEditable())
check("model combo shows a size", "—" in w._local_model_combo.itemText(0),
      w._local_model_combo.itemText(0))
check("model combo data is the bare name",
      w._local_model_combo.currentData() == "small-q5_1", w._local_model_combo.currentData())
check("model dir placeholder is a real path",
      "WhisperToCode" in w._local_model_dir_input.placeholderText(),
      w._local_model_dir_input.placeholderText())
check("local backend does not demand a key", w._api_key_page.isComplete() is True)

settings = w.collect_settings()
check("collects local backend", settings.stt_backend == "local")
check("collects chosen model", settings.local_model == "small-q5_1")

print("== customize checkbox now lives on the Speech backend page ==")
w1 = build(nvidia_api_key="", stt_backend="riva")
check("checkbox is parented to the Speech backend page",
      w1._customize_checkbox.parentWidget() is w1._api_key_page,
      w1._customize_checkbox.parentWidget())
check("checkbox keeps its wording",
      w1._customize_checkbox.text() == "Customize endpoints and models",
      w1._customize_checkbox.text())
check("checkbox starts unchecked", w1._customize_checkbox.isChecked() is False)
check("ticking it cannot complete a page that is missing its key",
      w1._api_key_page.isComplete() is False)
w1._customize_checkbox.setChecked(True)
app.processEvents()
check("still incomplete after ticking", w1._api_key_page.isComplete() is False)
check("and ticking does not count as touching the form",
      not w1._key_error.isVisible(), repr(w1._key_error.text()))
w2a = build(nvidia_api_key="", stt_backend="local")
check("ticking cannot break a page that was already complete",
      w2a._api_key_page.isComplete() is True)
w2a._customize_checkbox.setChecked(True)
app.processEvents()
check("still complete after ticking", w2a._api_key_page.isComplete() is True)

print("== riva backend ==")
w2 = build(nvidia_api_key="", stt_backend="riva")
check("riva backend demands a key", w2._api_key_page.isComplete() is False)
check("but says so quietly until the field is touched",
      not w2._key_error.isVisible() and w2._key_error.text() == "",
      repr(w2._key_error.text()))

print("== customize branch preserves unrelated fields ==")
w3 = build(nvidia_api_key="k", stt_backend="riva", nemotron_max_tokens=999)
w3._customize_checkbox.setChecked(True)
check("nemotron_max_tokens survives", w3.collect_settings().nemotron_max_tokens == 999,
      w3.collect_settings().nemotron_max_tokens)

print("== review page ==")
w4 = build(nvidia_api_key="", stt_backend="local")
w4._review_page.initializePage()
review = w4._review_label.text()
print("  " + review.replace("\n", "\n  "))
check("review names no Riva server for a local setup", "grpc.nvcf" not in review)
check("review mentions the config path", "config.json" in review)

# Everything below needs a rendered wizard: QWizard only builds a page's
# widgets and wires the buttons once the page is actually visited.
def shown(mode=ob.MODE_SETUP, **kwargs):
    w = build(mode, **kwargs)
    w._wizard.show()
    app.processEvents()
    return w


def next_btn(w):
    return w._wizard.button(QtWidgets.QWizard.NextButton)


def back_btn(w):
    return w._wizard.button(QtWidgets.QWizard.BackButton)


def title(w):
    return w._wizard.currentPage().title()


def walk(label, w, expected_titles):
    """Click the real Next button once per expected title and assert each landing."""
    check(f"{label}: opens on Speech backend", title(w) == "Speech backend", title(w))
    reached = []
    for expected in expected_titles:
        check(f"{label}: Next is enabled before '{expected}'",
              next_btn(w).isEnabled() is True)
        next_btn(w).click()
        app.processEvents()
        reached.append(title(w))
        check(f"{label}: Next lands on '{expected}'", reached[-1] == expected, reached[-1])
    check(f"{label}: route is exactly {expected_titles}", reached == expected_titles, reached)
    check(f"{label}: the deleted page is never shown", "Advanced setup" not in reached)
    return w


print("== route: unchecked -> Speech backend -> Review ==")
r1 = walk("default", shown(nvidia_api_key="k", stt_backend="riva"), ["Review"])
check("default: Save is enabled on Review",
      r1._wizard.button(QtWidgets.QWizard.FinishButton).isEnabled() is True)
check("default: Back is offered on Review", back_btn(r1).isEnabled() is True)
back_btn(r1).click()
app.processEvents()
check("default: Back returns to Speech backend", title(r1) == "Speech backend", title(r1))

print("== route: checked + cloud -> Riva endpoint -> Cleanup model -> Review ==")
r2 = shown(nvidia_api_key="k", stt_backend="riva")
r2._customize_checkbox.setChecked(True)
app.processEvents()
walk("customize+cloud", r2, ["Riva endpoint", "Cleanup model", "Review"])
check("customize+cloud: Save is enabled on Review",
      r2._wizard.button(QtWidgets.QWizard.FinishButton).isEnabled() is True)

print("== route: checked + local -> Cleanup model -> Review ==")
r3 = shown(nvidia_api_key="", stt_backend="local")
r3._customize_checkbox.setChecked(True)
app.processEvents()
walk("customize+local", r3, ["Cleanup model", "Review"])
check("customize+local: the Riva page is skipped for local speech",
      "Riva endpoint" not in [p.title() for p in
                              [r3._wizard.page(i) for i in r3._wizard.visitedIds()]],
      [r3._wizard.page(i).title() for i in r3._wizard.visitedIds()])

print("== route: the checkbox decides at the moment Next is clicked ==")
r4 = shown(nvidia_api_key="k", stt_backend="riva")
r4._customize_checkbox.setChecked(True)
r4._customize_checkbox.setChecked(False)
app.processEvents()
next_btn(r4).click()
app.processEvents()
check("un-ticking before Next goes straight to Review", title(r4) == "Review", title(r4))

print("== live validation: Speech backend ==")
w5 = shown(nvidia_api_key="", stt_backend="riva")
check("Next disabled on a blank key", next_btn(w5).isEnabled() is False,
      next_btn(w5).isEnabled())
check("no red text on an untouched form", not w5._key_error.isVisible())
w5._key_input.setText("nvapi-xyz")
app.processEvents()
check("Next enabled the moment a key is typed", next_btn(w5).isEnabled() is True,
      next_btn(w5).isEnabled())
w5._key_input.setText("")
app.processEvents()
check("Next disabled again when the key is cleared", next_btn(w5).isEnabled() is False)
check("error is inline and human", w5._key_error.isVisible()
      and w5._key_error.text().startswith("Enter an NVIDIA API key"), w5._key_error.text())
check("error label uses the themed object name",
      w5._key_error.objectName() == "onboardingError", w5._key_error.objectName())
check("the customize checkbox cannot rescue a blocked page",
      next_btn(w5).isEnabled() is False)
w5._customize_checkbox.setChecked(True)
app.processEvents()
check("still blocked after ticking customize", next_btn(w5).isEnabled() is False)
w5._customize_checkbox.setChecked(False)
w5._backend_combo.setCurrentIndex(1)  # whisper.cpp (local)
app.processEvents()
check("switching the combo to local re-enables Next with no key",
      next_btn(w5).isEnabled() is True, next_btn(w5).isEnabled())
check("and clears the error", not w5._key_error.isVisible())

print("== live validation: Riva endpoint + Cleanup model ==")
w6 = shown(nvidia_api_key="k", stt_backend="riva")
w6._customize_checkbox.setChecked(True)
app.processEvents()
next_btn(w6).click()   # customize + riva -> Riva endpoint, straight off page 0
app.processEvents()
check("reached the Riva page", title(w6) == "Riva endpoint", title(w6))
check("prefilled defaults leave Next enabled", next_btn(w6).isEnabled() is True)
w6._riva_server_input.setText("")
app.processEvents()
check("blank server disables Next", next_btn(w6).isEnabled() is False)
check("riva error keeps its old wording",
      w6._riva_error.text() == "Fill in both the server address and the function ID.",
      w6._riva_error.text())
w6._riva_server_input.setText("grpc.nvcf.nvidia.com:443")
app.processEvents()
check("retyping the server re-enables Next", next_btn(w6).isEnabled() is True)
w6._riva_function_input.setText("")
app.processEvents()
check("blank function id disables Next", next_btn(w6).isEnabled() is False)
w6._riva_function_input.setText("fn-1")
app.processEvents()
next_btn(w6).click()   # default ascending route -> Cleanup model
app.processEvents()
check("reached the Cleanup page", title(w6) == "Cleanup model", title(w6))
w6._nem_base_url_input.setText("")
w6._nem_model_input.setText("")
app.processEvents()
check("both blanks disable Next", next_btn(w6).isEnabled() is False)
check("both problems are listed at once, not just the first",
      w6._nem_error.text().split("\n") == [
          "The cleanup model needs an API endpoint.",
          "The cleanup model needs a model name.",
      ], w6._nem_error.text().split("\n"))
w6._nem_base_url_input.setText("https://integrate.api.nvidia.com/v1")
app.processEvents()
check("the fixed problem drops off the list",
      w6._nem_error.text() == "The cleanup model needs a model name.", w6._nem_error.text())
w6._nem_model_input.setText("nvidia/x")
app.processEvents()
check("Next enabled once both are filled", next_btn(w6).isEnabled() is True)
check("error hidden again", not w6._nem_error.isVisible())

print("== attribute-assigned overrides still reach Qt ==")
w7 = shown(nvidia_api_key="k", stt_backend="riva")
next_btn(w7).click()   # customize unchecked -> _backend_next_id() == Review
app.processEvents()
check("nextId attribute override works (skipped straight to Review)",
      title(w7) == "Review", title(w7))
check("initializePage attribute override works (Qt filled the review label)",
      "Saved to" in w7._review_label.text(), repr(w7._review_label.text()[:32]))
check("Save is enabled on Review",
      w7._wizard.button(QtWidgets.QWizard.FinishButton).isEnabled() is True)

STORED = "nvapi-0123456789wxyz"
FINGERPRINT = "••••wxyz"

print("== stored key is shown as a fingerprint, never in full ==")
check("fingerprint reveals only the tail of a long key",
      ob._key_fingerprint(STORED) == FINGERPRINT, ob._key_fingerprint(STORED))
check("fingerprint masks a short key whole",
      set(ob._key_fingerprint("k")) == {"•"}, ob._key_fingerprint("k"))
check("fingerprint masks a key one char below the reveal threshold",
      set(ob._key_fingerprint("abcdefg")) == {"•"}, ob._key_fingerprint("abcdefg"))

# isVisible() is False for every widget of an unshown wizard; isHidden() is the
# explicit setVisible(False) flag, which is what the build actually sets.
k0 = build(nvidia_api_key="", stt_backend="riva")
check("no stored key: the Remove button is hidden",
      k0._remove_key_button.isHidden() is True)
check("no stored key: placeholder is still the plain hint",
      k0._key_input.placeholderText() == "nvapi-...", k0._key_input.placeholderText())
k0._remove_key_button.click()
app.processEvents()
check("no stored key: clicking the hidden button cannot arm a phantom removal",
      k0._key_removed is False)
check("no stored key: a no-op click still collects nothing",
      k0.collect_settings().nvidia_api_key == "",
      repr(k0.collect_settings().nvidia_api_key))

k1 = build(nvidia_api_key=STORED, stt_backend="riva")
check("stored key: the Remove button is not hidden",
      k1._remove_key_button.isHidden() is False)
check("stored key: the Remove button exists and reads 'Remove'",
      k1._remove_key_button.text() == "Remove", k1._remove_key_button.text())
check("stored key: the button carries the themed object name",
      k1._remove_key_button.objectName() == "onboardingInlineButton",
      k1._remove_key_button.objectName())
check("stored key: the field is not left looking unset",
      k1._key_input.placeholderText() != "nvapi-..." and
      k1._key_input.placeholderText() != "", k1._key_input.placeholderText())
check("stored key: placeholder carries the masked fingerprint",
      FINGERPRINT in k1._key_input.placeholderText(), k1._key_input.placeholderText())
check("stored key: caption carries the same fingerprint",
      FINGERPRINT in k1._key_caption.text(), k1._key_caption.text())
check("stored key: the secret itself is never rendered",
      STORED not in k1._key_input.placeholderText()
      and STORED not in k1._key_caption.text()
      and STORED not in k1._api_key_page.subTitle()
      and k1._key_input.text() == "",
      repr(k1._key_input.text()))
check("stored key: only the last four characters are revealed",
      "0123456789" not in k1._key_input.placeholderText(),
      k1._key_input.placeholderText())
k1short = build(nvidia_api_key="hunter2x", stt_backend="riva")
check("a minimum-length stored key still gets a Remove button",
      k1short._remove_key_button.isHidden() is False)
check("the head of a stored key is never echoed into the placeholder",
      "hunter2" not in k1short._key_input.placeholderText(),
      k1short._key_input.placeholderText())

print("== state matrix: untouched / replaced / removed / undone ==")
m1 = build(nvidia_api_key=STORED, stt_backend="riva")
check("untouched: blank field still keeps the stored key",
      m1.collect_settings().nvidia_api_key == STORED,
      m1.collect_settings().nvidia_api_key)
check("untouched: the page is complete on the cloud backend",
      m1._api_key_page.isComplete() is True)

m2 = build(nvidia_api_key=STORED, stt_backend="riva")
m2._key_input.setText("nvapi-brand-new")
app.processEvents()
check("replaced: the typed key wins",
      m2.collect_settings().nvidia_api_key == "nvapi-brand-new",
      m2.collect_settings().nvidia_api_key)
check("replaced: typing does not arm a removal", m2._key_removed is False)

m3 = build(nvidia_api_key=STORED, stt_backend="riva")
m3._remove_key_button.click()
app.processEvents()
check("removed: the flag is armed", m3._key_removed is True)
check("removed: collect_settings returns an empty key",
      m3.collect_settings().nvidia_api_key == "",
      repr(m3.collect_settings().nvidia_api_key))
check("removed: the button offers the undo",
      m3._remove_key_button.text() == "Undo", m3._remove_key_button.text())
check("removed: the field says what will happen on save",
      "removed" in m3._key_input.placeholderText().lower(),
      m3._key_input.placeholderText())
check("removed: no fingerprint is dangled once the key is going",
      FINGERPRINT not in m3._key_input.placeholderText(),
      m3._key_input.placeholderText())
check("removed: the customize branch blanks the key too",
      (m3._customize_checkbox.setChecked(True),
       m3.collect_settings().nvidia_api_key)[1] == "",
      repr(m3.collect_settings().nvidia_api_key))
m3._customize_checkbox.setChecked(False)

m4 = build(nvidia_api_key=STORED, stt_backend="riva")
m4._remove_key_button.click()
m4._remove_key_button.click()
app.processEvents()
check("undone: the flag is disarmed", m4._key_removed is False)
check("undone: the stored key is back",
      m4.collect_settings().nvidia_api_key == STORED,
      m4.collect_settings().nvidia_api_key)
check("undone: the button reads 'Remove' again",
      m4._remove_key_button.text() == "Remove", m4._remove_key_button.text())
check("undone: the fingerprint comes back",
      FINGERPRINT in m4._key_input.placeholderText(), m4._key_input.placeholderText())

m5 = build(nvidia_api_key=STORED, stt_backend="riva")
m5._remove_key_button.click()
m5._key_input.setText("nvapi-replacement")
app.processEvents()
check("removed then retyped: typing disarms the removal", m5._key_removed is False)
check("removed then retyped: the typed key wins over both",
      m5.collect_settings().nvidia_api_key == "nvapi-replacement",
      m5.collect_settings().nvidia_api_key)
check("removed then retyped: the button drops back to 'Remove'",
      m5._remove_key_button.text() == "Remove", m5._remove_key_button.text())
m5._key_input.setText("")
app.processEvents()
check("clearing a disarmed field does not silently re-arm the removal",
      m5._key_removed is False and m5.collect_settings().nvidia_api_key == STORED,
      m5.collect_settings().nvidia_api_key)

m6 = build(nvidia_api_key=STORED, stt_backend="riva")
m6._key_input.setText("nvapi-typed-first")
m6._remove_key_button.click()
app.processEvents()
check("typed then removed: the click wins and the field is cleared",
      m6._key_input.text() == "" and m6._key_removed is True,
      repr(m6._key_input.text()))
check("typed then removed: collect_settings returns an empty key",
      m6.collect_settings().nvidia_api_key == "",
      repr(m6.collect_settings().nvidia_api_key))

print("== removal drives the cloud-backend gate exactly like clearing does ==")
g1 = shown(nvidia_api_key=STORED, stt_backend="riva")
check("cloud: Next starts enabled on a stored key",
      next_btn(g1).isEnabled() is True, next_btn(g1).isEnabled())
check("cloud: no red text before the user touches anything",
      not g1._key_error.isVisible())
g1._remove_key_button.click()
app.processEvents()
check("cloud: removing the key disables Next",
      next_btn(g1).isEnabled() is False, next_btn(g1).isEnabled())
check("cloud: the inline error appears, worded as when the field is cleared",
      g1._key_error.isVisible()
      and g1._key_error.text().startswith("Enter an NVIDIA API key"),
      g1._key_error.text())
g1._remove_key_button.click()
app.processEvents()
check("cloud: undo re-enables Next", next_btn(g1).isEnabled() is True,
      next_btn(g1).isEnabled())
check("cloud: undo clears the inline error", not g1._key_error.isVisible(),
      g1._key_error.text())
g1._remove_key_button.click()
app.processEvents()
g1._key_input.setText("nvapi-typed-over-removal")
app.processEvents()
check("cloud: typing over a removal also re-enables Next",
      next_btn(g1).isEnabled() is True, next_btn(g1).isEnabled())
g1._key_input.setText("")
g1._remove_key_button.click()   # armed again, page still blocked
app.processEvents()
check("cloud: a removed key blocks Next even with customize ticked",
      (g1._customize_checkbox.setChecked(True), app.processEvents(),
       next_btn(g1).isEnabled())[2] is False)
g1._backend_combo.setCurrentIndex(1)   # whisper.cpp (local)
app.processEvents()
check("switching to local unblocks a removed key, same as a blank one",
      next_btn(g1).isEnabled() is True, next_btn(g1).isEnabled())
check("and the error goes quiet", not g1._key_error.isVisible())

print("== removal is reported on the Review page ==")
v1 = build(nvidia_api_key=STORED, stt_backend="local")
v1._review_page.initializePage()
check("local + stored key: cleanup reads as ready",
      "Transcript cleanup: ready with" in v1._review_label.text(),
      v1._review_label.text())
check("review never prints the secret", STORED not in v1._review_label.text())
v1._remove_key_button.click()
v1._review_page.initializePage()
check("local + removed key: cleanup reports it cannot run",
      "Transcript cleanup: needs an NVIDIA API key before it can run"
      in v1._review_label.text(), v1._review_label.text())
check("local + removed key: dictation itself is still described",
      "whisper.cpp on this machine" in v1._review_label.text(),
      v1._review_label.text())
v1._remove_key_button.click()
v1._review_page.initializePage()
check("local + undone: the Review page flips back to ready",
      "Transcript cleanup: ready with" in v1._review_label.text(),
      v1._review_label.text())

print("== removal leaves every other field alone ==")
u1 = build(nvidia_api_key=STORED, stt_backend="riva", nemotron_max_tokens=999,
           local_model="small-q5_1", riva_function_id="fn-keep")
u1._customize_checkbox.setChecked(True)
u1._remove_key_button.click()
app.processEvents()
removed_settings = u1.collect_settings()
check("removal blanks only the key",
      removed_settings.nvidia_api_key == ""
      and removed_settings.nemotron_max_tokens == 999
      and removed_settings.local_model == "small-q5_1"
      and removed_settings.riva_function_id == "fn-keep"
      and removed_settings.stt_backend == "riva",
      removed_settings)

print("== mode: setup framing vs settings framing ==")
s_setup = build(nvidia_api_key="k", stt_backend="riva")
check("the default mode is still first-run setup",
      s_setup._wizard.windowTitle() == "WhisperToCode Setup", s_setup._wizard.windowTitle())
s_set = build(ob.MODE_SETTINGS, nvidia_api_key="k", stt_backend="riva")
check("settings mode retitles the window",
      s_set._wizard.windowTitle() == "WhisperToCode Settings", s_set._wizard.windowTitle())
check("settings mode never says Setup",
      "Setup" not in s_set._wizard.windowTitle(), s_set._wizard.windowTitle())
check("an unrecognised mode falls back to the setup framing",
      build("nonsense", nvidia_api_key="k")._wizard.windowTitle() == "WhisperToCode Setup")
check("settings mode changes the framing only, not the pages",
      [s_set._wizard.page(i).title() for i in s_set._wizard.pageIds()]
      == ["Speech backend", "Riva endpoint", "Cleanup model", "Review"],
      [s_set._wizard.page(i).title() for i in s_set._wizard.pageIds()])
check("settings mode still finishes on Save",
      s_set._wizard.buttonText(QtWidgets.QWizard.FinishButton) == "Save",
      s_set._wizard.buttonText(QtWidgets.QWizard.FinishButton))

print("== the customize checkbox reflects what is stored, not a hardcoded False ==")
check("settings + an untouched config: the box stays unticked",
      s_set._customize_checkbox.isChecked() is False)
check("a stored key on its own is not a customisation",
      build(ob.MODE_SETTINGS, nvidia_api_key=STORED)._customize_checkbox.isChecked() is False)
ADVANCED = (
    ("riva_server", "grpc.example.com:443"),
    ("riva_function_id", "fn-custom"),
    ("nemotron_base_url", "https://example.invalid/v1"),
    ("nemotron_model", "vendor/other"),
    ("nemotron_temperature", 0.4),
    ("nemotron_top_p", 0.8),
    ("nemotron_max_tokens", 999),
    ("nemotron_reasoning_budget", 7),
    ("nemotron_enable_thinking", False),
)
for field, value in ADVANCED:
    check(f"a custom {field} opens the box ticked",
          build(ob.MODE_SETTINGS, **{field: value})._customize_checkbox.isChecked() is True)
    check(f"a custom {field} ticks it in setup mode too",
          build(**{field: value})._customize_checkbox.isChecked() is True)
for field, value in (("stt_backend", "local"), ("local_model", "small-q5_1"),
                     ("local_model_dir", "D:/models")):
    check(f"{field} is a first-page choice, so it leaves the box alone",
          build(ob.MODE_SETTINGS, **{field: value})._customize_checkbox.isChecked() is False)

print("== route: a returning user's own values are on the route already ==")
CUSTOM = dict(nvidia_api_key="k", stt_backend="riva",
              riva_server="grpc.example.com:443", nemotron_model="vendor/other")
walk("settings+custom", shown(ob.MODE_SETTINGS, **CUSTOM),
     ["Riva endpoint", "Cleanup model", "Review"])
walk("settings+defaults", shown(ob.MODE_SETTINGS, nvidia_api_key="k", stt_backend="riva"),
     ["Review"])


def clicks_to(w, target):
    """Interactions from the opened dialog to `target`, checkbox hunt included."""
    used = 0
    if not w._customize_checkbox.isChecked():
        w._customize_checkbox.setChecked(True)
        app.processEvents()
        used += 1
    while title(w) != target:
        if not next_btn(w).isEnabled() or used > 8:
            return None
        next_btn(w).click()
        app.processEvents()
        used += 1
    return used


now = shown(ob.MODE_SETTINGS, **CUSTOM)
new_cost = clicks_to(now, "Cleanup model")
was = shown(ob.MODE_SETTINGS, **CUSTOM)
was._customize_checkbox.setChecked(False)   # what every open used to start from
app.processEvents()
old_cost = clicks_to(was, "Cleanup model")
check("reaching the stored cleanup model costs fewer clicks than before",
      new_cost is not None and old_cost is not None and new_cost < old_cost,
      (new_cost, old_cost))
check("and it is exactly the two Next clicks, with no checkbox to remember",
      new_cost == 2, new_cost)
check("the stored model is on screen once there, not hidden behind a checkbox",
      now._nem_model_input.isVisible() and now._nem_model_input.text() == "vendor/other",
      now._nem_model_input.text())
check("the stored endpoint was shown on the way past",
      "Riva endpoint" in [now._wizard.page(i).title() for i in now._wizard.visitedIds()]
      and now._riva_server_input.text() == "grpc.example.com:443",
      now._riva_server_input.text())
check("nothing is lost by not touching the pages: Save keeps the stored values",
      (lambda s: s.riva_server == "grpc.example.com:443"
       and s.nemotron_model == "vendor/other")(now.collect_settings()),
      now.collect_settings())

print("== the mode survives the trip through run_onboarding_with_qt ==")
seen = []
real_wizard = ob._OnboardingWizard


class _ModeRecorder:
    def __init__(self, qt_core, qt_gui, qt_widgets, initial, mode=ob.MODE_SETUP):
        seen.append(mode)

    def exec(self):
        return 0


ob._OnboardingWizard = _ModeRecorder
try:
    ob.run_onboarding_with_qt(QtCore, QtGui, QtWidgets, AppSettings(), ob.MODE_SETTINGS)
    ob.run_onboarding_with_qt(QtCore, QtGui, QtWidgets, AppSettings())
finally:
    ob._OnboardingWizard = real_wizard
check("run_onboarding_with_qt hands the mode straight to the wizard",
      seen == [ob.MODE_SETTINGS, ob.MODE_SETUP], seen)

check("removal never opened a modal popup either", not POPUPS, POPUPS)
check("validation never opened a modal popup", not POPUPS, POPUPS)

print()
if FAILURES:
    print(f"HARNESS FAILED: {len(FAILURES)} check(s): {FAILURES}")
    raise SystemExit(1)
print("HARNESS PASSED")
