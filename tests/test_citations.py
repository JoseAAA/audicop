"""Tests for `app.services.citations` (deterministic [MM:SS] verification).

The fixture transcript and the wrong answers are the real reproduction of a
user-reported bug: Qwen 3B cited [02:28] for facts that live in the [01:55]
line (anchoring on a mark from earlier in the conversation), or dropped the
mark entirely.
"""

from __future__ import annotations

from app.services.citations import CitationFixer, parse_transcript

TRANSCRIPT = """[00:00] Buenas tardes, señor García. Pase, siéntese. ¿Cómo se ha sentido desde que le dieron de alta? Salí del hospital el martes pasado después del stent, pero me viene un dolor en el pecho cuando subo las escaleras.
[00:28] Un minuto, doctor. Me detengo, me siento y se me pasa. Ya me ha pasado tres veces en dos días. ¿Es parecida al dolor del infarto? No, ese fue mucho peor. Este es más leve.
[00:57] pero el lugar es el mismo, en el medio del pecho, opresivo. ¿Está tomando todo lo que le indicaron al alta? Todo, doctor. Mire, aspirina 100, clopidogrel 75, atorvastatina 40,
[01:26] En el april 10 y bisoprolol 5. ¿Antiinflamatorios? No, doctor, nada. Y además soy alérgico a la penicilina. Me da un zarpullido feo. ¿Fuma? No fumo hace 5 años.
[01:55] Vamos a tomarle la presión y a revisarlo. Su presión está en 138 sobre 84. Pulso 68, regular. Saturación 97 en aire ambiente. No tiene fiebre. Los pulmones limpios.
[02:28] Es algo que tenemos que descartar de raíz. Lo más preocupante sería una trombosis del stent o una reistenosis temprana. ¿Eso es grave, doctor? Puede serlo."""


def _fix(line: str) -> str:
    return CitationFixer(TRANSCRIPT).fix_line(line)


def test_parse_transcript_extracts_all_lines() -> None:
    entries = parse_transcript(TRANSCRIPT)
    assert [e.mark for e in entries] == ["00:00", "00:28", "00:57", "01:26", "01:55", "02:28"]


def test_adds_missing_mark_saturation() -> None:
    """Real failed output #1: fact stated with no mark at all."""
    out = _fix("La saturación es de 97 en aire ambiente.")
    assert out.startswith("[01:55] ")


def test_replaces_anchored_wrong_mark_blood_pressure() -> None:
    """Real failed output #2: anchored on [02:28]; the fact lives at [01:55]."""
    out = _fix("[02:28] Su presión arterial está en 138 sobre 84. Pulso 68, regular.")
    assert out.startswith("[01:55] ")
    assert "[02:28]" not in out


def test_adds_missing_mark_medications() -> None:
    out = _fix("El paciente toma aspirina 100, clopidogrel 75, atorvastatina 40.")
    assert out.startswith("[00:57] ")


def test_adds_missing_mark_allergy() -> None:
    out = _fix("El paciente es alérgico a la penicilina.")
    assert out.startswith("[01:26] ")


def test_keeps_correct_mark() -> None:
    line = "[02:28] El médico teme una trombosis del stent o una reistenosis temprana."
    assert _fix(line) == line


def test_bold_synthesis_lines_never_gain_a_mark() -> None:
    """**Tema:**/**Conclusión:** are synthesis — even with strong lexical
    overlap they must not receive a timestamp (real user-reported bug)."""
    line = "**Conclusión:** El médico teme una trombosis del stent tras revisar la presión."
    assert _fix(line) == line
    line2 = "**Tema:** Saturación, presión arterial y medicación del paciente."
    assert _fix(line2) == line2


def test_meta_answer_left_untouched() -> None:
    """Weak overlap (an 'it is not mentioned' answer) must not gain a mark."""
    line = "No se dice el nombre del doctor."
    assert _fix(line) == line


def test_bullet_lines_get_mark_after_bullet() -> None:
    out = _fix("- Saturación 97 en aire ambiente y pulmones limpios.")
    assert out.startswith("- [01:55] ")


def test_ranges_are_left_alone() -> None:
    line = "[00:57–01:26] El paciente repasa su medicación: aspirina y bisoprolol."
    assert _fix(line) == line


def test_streaming_fixes_line_by_line() -> None:
    """Chunks split mid-line are reassembled and corrected at line boundaries."""
    fixer = CitationFixer(TRANSCRIPT)
    chunks = [
        "La saturación es de 97 en",
        " aire ambiente.\n[02:28] Su presión está en 138 ",
        "sobre 84.",
    ]
    out = ""
    for c in chunks:
        out += "".join(fixer.feed(c))
    out += "".join(fixer.flush())
    lines = out.splitlines()
    assert lines[0].startswith("[01:55] ")
    assert lines[1].startswith("[01:55] ")


def test_no_transcript_is_a_noop() -> None:
    fixer = CitationFixer("")
    line = "[09:99] lo que sea"
    assert fixer.fix_line(line) == line
