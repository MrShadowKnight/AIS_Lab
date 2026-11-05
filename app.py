"""Веб-інтерфейс експертної системи. Запуск: python app.py → http://127.0.0.1:5001"""
from flask import Flask, jsonify, request, send_from_directory

from es.engine import explain_tree, infer, rule_text
from es.knowledge_base import ATTRS, QUESTIONS, RULES

app = Flask(__name__, static_folder='static')
app.json.sort_keys = False      # зберігаємо порядок аспектів


@app.route('/')
def index():
    return send_from_directory('static', 'index.html')


@app.get('/api/kb')
def kb():
    """База знань для показу: питання експерта і правила людською мовою."""
    return jsonify({'questions': QUESTIONS, 'rules': [rule_text(r) for r in RULES]})


@app.post('/api/consult')
def consult():
    facts = request.get_json()['facts']
    result = infer(facts)
    # для кожного аспекту - дерево пояснення обраного висновку
    for aspect in result['aspects'].values():
        best = aspect['alternatives'][0]['id']
        aspect['explanation'] = explain_tree(result, best) if aspect['decision'] else []
    result.pop('trace')
    return jsonify(result)


if __name__ == '__main__':
    app.run(port=5001)
