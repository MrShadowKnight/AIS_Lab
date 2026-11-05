"""Скриншоти роботи експертної системи (Playwright проходить консультацію в браузері)."""
import time
from playwright.sync_api import sync_playwright

from es.knowledge_base import QUESTIONS
from scenarios import SCENARIOS

URL = 'http://127.0.0.1:5001'
OUT = 'shots/'


def consult(page, facts, why_at=None, shot_q=None):
    page.goto(URL)
    page.click('#start')
    for i, q in enumerate(QUESTIONS):
        page.wait_for_selector('.opt')
        if why_at == q['id']:
            page.click('#why')
            if shot_q:
                page.screenshot(path=OUT + shot_q)
        page.click(f'.opt[data-v="{facts.get(q["id"], "unknown")}"]')
    page.wait_for_selector('.summary')
    time.sleep(0.3)


with sync_playwright() as p:
    b = p.chromium.launch()
    page = b.new_page(viewport={'width': 1280, 'height': 900})
    page.goto(URL)
    page.wait_for_selector('#start')
    page.screenshot(path=OUT + '01_intro.png')

    consult(page, SCENARIOS['Бухгалтерська фірма (25 осіб)'], why_at='cable',
            shot_q='02_question.png')
    page.screenshot(path=OUT + '03_result_accounting.png', full_page=True)
    # розгорнуте пояснення для середовища передачі
    page.locator('details').nth(2).evaluate('d => d.open = true')
    page.locator('.grid .card').filter(has=page.locator('h3', has_text='Середовище передачі')).screenshot(
        path=OUT + '04_explanation.png')

    consult(page, SCENARIOS['Школа з двома корпусами'])
    page.screenshot(path=OUT + '05_result_school.png', full_page=True)

    consult(page, SCENARIOS['Офіс в історичній будівлі'])
    page.screenshot(path=OUT + '06_result_historic.png', full_page=True)

    consult(page, SCENARIOS['Замовник майже нічого не знає'])
    page.screenshot(path=OUT + '07_result_unknown.png', full_page=True)

    page.click('#nav-kb')
    page.wait_for_selector('table')
    page.screenshot(path=OUT + '08_kb.png')
    b.close()
