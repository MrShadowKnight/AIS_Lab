"""Машина виведення: пряме виведення з коефіцієнтами впевненості (за схемою MYCIN)."""
from .knowledge_base import ATTRS, GROUPS, QUESTIONS, RULES, UNKNOWN

QIDS = {q['id'] for q in QUESTIONS}
THRESHOLD = 0.2            # правило спрацьовує, якщо впевненість у передумовах >= 0.2


def combine(a, b):
    """Об'єднання двох коефіцієнтів впевненості для одного висновку (MYCIN)."""
    if a >= 0 and b >= 0:
        return a + b * (1 - a)
    if a <= 0 and b <= 0:
        return a + b * (1 + a)
    return (a + b) / (1 - min(abs(a), abs(b)))


def _rule_levels():
    """Рівень правила = 1 + найбільший рівень атрибутів у його умовах.
    Відповіді замовника мають рівень 0. Так правила виконуються від фактів до висновків."""
    attr_level = {q: 0 for q in QIDS}
    rule_level = {}
    changed = True
    while changed:
        changed = False
        for r in RULES:
            lv = 1 + max(attr_level.get(c[0], 0) for c in r['if'])
            if rule_level.get(r['id']) != lv:
                rule_level[r['id']] = lv
                changed = True
            a = r['then'][0]
            if attr_level.get(a, 0) < lv:
                attr_level[a] = lv
                changed = True
    return rule_level


LEVELS = _rule_levels()
ORDERED = sorted(RULES, key=lambda r: LEVELS[r['id']])


def answer_text(qid, value):
    q = next(q for q in QUESTIONS if q['id'] == qid)
    return dict(q['options']).get(value, 'не знаю')


def infer(facts):
    """facts: {id питання: id відповіді або 'unknown'}. Повертає результат консультації."""
    cf = {}                     # робоча пам'ять: поточна впевненість у кожному висновку
    trace = {}                  # які правила вплинули на висновок
    blocked = {}                # які висновки не врахували через «не знаю»

    for r in ORDERED:
        strengths, unknown_q = [], []
        for attr, op, val in r['if']:
            if op in ('is', 'in'):
                ans = facts.get(attr, UNKNOWN)
                if ans == UNKNOWN:
                    unknown_q.append(attr)
                    continue
                ok = ans == val if op == 'is' else ans in val
                strengths.append(1.0 if ok else 0.0)
            elif op == 'yes':
                strengths.append(cf.get(attr, 0.0))
            else:                                   # 'no' - висновок спростовано
                strengths.append(-cf.get(attr, 0.0))
        if strengths and min(strengths) < THRESHOLD:
            continue                                # якась умова точно не виконана
        if unknown_q:                               # не вистачає відповіді замовника
            for q in unknown_q:
                blocked.setdefault(q, set()).add(r['then'][0])
            continue
        strength = min(strengths)
        attr, rule_cf = r['then']
        delta = rule_cf * strength
        before = cf.get(attr, 0.0)
        cf[attr] = combine(before, delta)
        trace.setdefault(attr, []).append({
            'rule': r['id'], 'strength': round(strength, 2), 'delta': round(delta, 2),
            'after': round(cf[attr], 2), 'why': r['why'],
            'premises': [_premise_text(c, facts, cf) for c in r['if']],
            'sub': [c[0] for c in r['if'] if c[1] in ('yes', 'no')]})
    return _report(facts, cf, trace, blocked)


def _premise_text(cond, facts, cf):
    attr, op, val = cond
    if op in ('is', 'in'):
        q = next(q for q in QUESTIONS if q['id'] == attr)
        return f"«{q['text']}» → {answer_text(attr, facts[attr])}"
    label = ATTRS[attr][0]
    return (f'{label} ({cf.get(attr, 0):+.2f})' if op == 'yes'
            else f'НЕ {label} ({cf.get(attr, 0):+.2f})')


def confidence_label(v):
    if v >= 0.8:
        return 'висока'
    if v >= 0.5:
        return 'середня'
    if v >= THRESHOLD:
        return 'низька'
    return 'немає'


def _report(facts, cf, trace, blocked):
    aspects = {}
    for g, title in GROUPS.items():
        alts = sorted(((a, cf.get(a, 0.0)) for a, (_, grp) in ATTRS.items() if grp == g),
                      key=lambda p: -p[1])
        best, best_cf = alts[0]
        aspects[g] = {
            'title': title,
            'decision': ATTRS[best][0] if best_cf >= THRESHOLD else None,
            'cf': round(best_cf, 2), 'level': confidence_label(best_cf),
            'alternatives': [{'id': a, 'label': ATTRS[a][0], 'cf': round(v, 2)} for a, v in alts],
        }
    advice = sorted(((ATTRS[a][0], round(v, 2)) for a, v in cf.items()
                     if ATTRS[a][1] == 'advice' and v >= 0.5), key=lambda p: -p[1])
    intermediate = [{'id': a, 'label': ATTRS[a][0], 'cf': round(v, 2)}
                    for a, v in cf.items() if ATTRS[a][1] is None]

    # що варто уточнити: невідомі відповіді, через які не враховано висновки
    unknowns = []
    for qid, attrs in blocked.items():
        q = next(q for q in QUESTIONS if q['id'] == qid)
        unknowns.append({'question': q['text'],
                         'affects': sorted({ATTRS[a][0] for a in attrs})})
    weak = [a['title'] for a in aspects.values() if a['cf'] < 0.5]

    parts = [f"{a['title'].lower()} - {a['decision']}" for a in aspects.values() if a['decision']]
    text = '; '.join(parts)
    summary = ('Попередня рекомендація. ' + text[:1].upper() + text[1:] + '.') if parts else \
        'Даних недостатньо для рекомендації.'
    return {'summary': summary, 'aspects': aspects, 'advice': advice,
            'intermediate': intermediate, 'trace': trace, 'unknowns': unknowns,
            'weak': weak,
            'answers': [{'question': q['text'], 'answer': answer_text(q['id'],
                         facts.get(q['id'], UNKNOWN))} for q in QUESTIONS]}


def explain_tree(result, attr, seen=None):
    """Дерево пояснення: правила, що вплинули на висновок, і їхні передумови
    (проміжні висновки розкриваються рекурсивно аж до відповідей замовника)."""
    seen = set() if seen is None else seen
    nodes = []
    for t in result['trace'].get(attr, []):
        children = []
        for sub in t['sub']:
            if sub not in seen:
                seen.add(sub)
                children.append({'attr': ATTRS[sub][0], 'rules': explain_tree(result, sub, seen)})
        nodes.append({**{k: t[k] for k in ('rule', 'delta', 'after', 'why', 'premises')},
                      'children': children})
    return nodes


OPS = {'is': '=', 'in': '∈'}


def rule_text(r):
    """Правило людською мовою: ЯКЩО ... І ... ТО ... (впевненість)."""
    conds = []
    for attr, op, val in r['if']:
        if op in ('is', 'in'):
            q = next(q for q in QUESTIONS if q['id'] == attr)
            opts = dict(q['options'])
            v = opts[val] if op == 'is' else ' або '.join(opts[x] for x in val)
            conds.append(f'{q["text"].rstrip("?")}: {v}')
        else:
            conds.append(('' if op == 'yes' else 'НЕ ') + ATTRS[attr][0])
    attr, c = r['then']
    concl = ATTRS[attr][0]            # знак CF показує, «за» чи «проти» висновку
    return {'id': r['id'], 'if': conds, 'then': concl, 'cf': c, 'why': r['why'],
            'level': LEVELS[r['id']]}
