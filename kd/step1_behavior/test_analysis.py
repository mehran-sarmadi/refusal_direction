import pandas as pd

from kd.step1_behavior import analysis


def test_is_refusal_handles_curly_apostrophe_and_case():
    assert analysis.is_refusal("I can’t help with that.")
    assert analysis.is_refusal("i cannot do this")
    assert not analysis.is_refusal("Sure, here is a recipe for bread.")


def test_mcnemar_counts_discordant_pairs():
    r = analysis.mcnemar_exact([1, 1, 1, 0], [0, 0, 1, 0])
    assert (r["only_teacher"], r["only_student"]) == (2, 0)
    assert analysis.mcnemar_exact([1, 0], [1, 0])["p_value"] == 1.0


def _df(t_h, s_h, t_r, s_r):
    n = len(t_h)
    return pd.DataFrame({"category": ["c"] * n, "teacher_harmful": t_h, "student_harmful": s_h,
                         "teacher_refusal": t_r, "student_refusal": s_r})


def test_verdict_detects_clear_advantage():
    n = 100
    df = _df([False] * n, [True] * 30 + [False] * 70, [True] * n, [False] * 30 + [True] * 70)
    v = analysis.verdict(df)
    assert v["teacher_has_safety_advantage"]
    assert abs(v["harmful_gap_pp"] - 30) < 1e-9
    assert not v["both_near_perfect"]


def test_verdict_flags_ceiling():
    n = 100
    df = _df([False] * n, [False] * 99 + [True], [True] * n, [True] * n)
    v = analysis.verdict(df)
    assert not v["teacher_has_safety_advantage"]
    assert v["both_near_perfect"]


def test_disagreements_labels_kind():
    df = _df([False, False], [False, True], [True, True], [False, True])
    d = analysis.disagreements(df)
    assert list(d.disagreement) == ["teacher_refuses_only", "harmful_label_differs"]
