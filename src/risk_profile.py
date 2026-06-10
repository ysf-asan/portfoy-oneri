from config import (
    RISK_QUESTIONNAIRE, SCORE_TO_PROFILE, PROFILE_CONSTRAINTS,
    QUESTION_WEIGHTS, INCOME_TO_CARDINALITY,
)

def calculate_risk_score(answers):
    score = 0
    for key, weight in QUESTION_WEIGHTS.items():
        if key in answers:
            score += answers[key] * weight
    return score

def get_profile(score):
    for threshold, profile in SCORE_TO_PROFILE:
        if score >= threshold:
            return profile
    return "conservative"

def get_constraints(profile_name):
    return PROFILE_CONSTRAINTS[profile_name]

def get_cardinality(income_score):
    return INCOME_TO_CARDINALITY.get(income_score, 5)

def _calc_max_score():
    return sum(
        max(s for _, s in q["options"]) * QUESTION_WEIGHTS[q["key"]]
        for q in RISK_QUESTIONNAIRE
    )

MAX_SCORE = _calc_max_score()

def _personalize_constraints(base_constraints, answers):
    """Anket cevaplarına göre temel kısıtları kişiselleştir."""
    import copy
    c = copy.deepcopy(base_constraints)

    age = answers.get("age", 1)
    horizon = answers.get("horizon", 1)
    goal = answers.get("goal", 1)
    loss_tol = answers.get("loss_tolerance", 1)

    # Genç + uzun vade → equity tavanı hafif yükselt
    if age >= 2 and horizon >= 2:
        c["max_equity_crypto"] = min(c["max_equity_crypto"] + 0.10, 1.0)

    # Kısa vade (1 yıldan az) → volatilite kısıtı ekle
    if horizon == 0:
        c["max_single_asset"] = min(c["max_single_asset"], 0.12)
        c["max_equity_crypto"] = min(c["max_equity_crypto"], 0.30)
        c["short_horizon"] = True
    else:
        c["short_horizon"] = False

    # Sermaye koruma hedefi → daha muhafazakâr
    if goal == 0:
        c["max_equity_crypto"] = min(c["max_equity_crypto"], 0.20)
        c["min_bond_commodity_forex"] = max(c["min_bond_commodity_forex"], 0.70)
        c["capital_preservation"] = True
    else:
        c["capital_preservation"] = False

    # Düşük kayıp toleransı → tek varlık limiti düşür
    if loss_tol == 0:
        c["max_single_asset"] = min(c["max_single_asset"], 0.10)

    # Kişiselleştirme parametrelerini optimizer'a geç
    c["personal_seed_offset"] = age * 100 + horizon * 10 + goal * 3 + loss_tol

    return c


def map_answers_to_profile(answers):
    score = calculate_risk_score(answers)
    profile_name = get_profile(score)
    base_constraints = get_constraints(profile_name)
    constraints = _personalize_constraints(base_constraints, answers)
    card = get_cardinality(answers.get("income", 1))
    return {
        "profile_name": profile_name,
        "profile_label": constraints["label"],
        "score": round(score, 1),
        "max_score": MAX_SCORE,
        "constraints": constraints,
        "cardinality": card,
        "answers": answers,
    }

def get_questionnaire():
    return RISK_QUESTIONNAIRE
