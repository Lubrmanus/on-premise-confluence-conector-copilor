import aiohttp
import asyncio
from atlassian import Confluence
import spacy
import hashlib
import time
import os
from bs4 import BeautifulSoup
from diskcache import Cache
from functools import wraps
import logging
import requests

# Nastavení základního logování
logging.basicConfig(level=logging.INFO)

# Načtení NLP modelu
try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load('en_core_web_sm')

# Disk cache pro persistenci výsledků
cache = Cache('/tmp/confluence_cache')

# Rate limiting
last_called = 0
RATE_LIMIT_SECONDS = 1  # Omezení na 1 dotaz za sekundu

def rate_limited(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        global last_called
        now = time.time()
        if now - last_called < RATE_LIMIT_SECONDS:
            time.sleep(RATE_LIMIT_SECONDS - (now - last_called))
        last_called = time.time()
        return func(*args, **kwargs)
    return wrapper

# Připojení k on-premise Confluence
confluence = Confluence(
    url=os.getenv('CONFLUENCE_URL'),
    username=os.getenv('CONFLUENCE_USERNAME'),
    password=os.getenv('CONFLUENCE_PASSWORD')
)

# Paginace pro načítání spaces
@rate_limited
def get_all_spaces():
    start = 0
    limit = 50
    all_spaces = []
    cache_key = "all_spaces"

    # Pokud je cache stále aktuální, vrátíme ji
    if cache_key in cache:
        return cache[cache_key]

    # Paginace pro načítání všech spaces
    while True:
        try:
            spaces = confluence.get_all_spaces(start=start, limit=limit)
            all_spaces.extend(spaces['results'])
            if len(spaces['results']) < limit:
                break
            start += limit
        except Exception as e:
            logging.error(f"Chyba při načítání spaces: {e}")
            break

    # Uložení do cache
    cache[cache_key] = all_spaces
    return all_spaces

# Zpracování dotazu pomocí NLP a Named Entity Recognition
def process_question_with_ner(question):
    doc = nlp(question)
    keywords = [ent.text for ent in doc.ents if ent.label_ in ['PERSON', 'ORG', 'DATE', 'GPE']]
    if not keywords:  # Pokud neexistují žádné pojmy, vrátíme klíčová slova bez stop slov
        keywords = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]
    return ' '.join(keywords)

# Klasifikace otázky
def classify_question(question):
    doc = nlp(question)
    if any(token.text.lower() in ["how", "what", "why", "mám", "je", "jak"] for token in doc):
        return "informative"
    elif any(token.text.lower() in ["list", "which", "give me", "můžu", "mám"] for token in doc):
        return "list"
    return "general"

# Funkce pro extrakci specifického obsahu (tabulky, seznamy, text)
def extract_specific_content(page_content, content_type='text'):
    soup = BeautifulSoup(page_content, 'html.parser')
    if content_type == 'table':
        tables = soup.find_all('table')
        return [str(table) for table in tables]
    elif content_type == 'list':
        lists = soup.find_all(['ul', 'ol'])
        return [str(lst) for lst in lists]
    else:
        return soup.get_text(separator='\n')  # Vrátíme čistý text s novými řádky

# Získání obsahu stránky s timeoutem
async def fetch_page_content(page_id):
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{os.getenv('CONFLUENCE_URL')}/rest/api/content/{page_id}", auth=(os.getenv('CONFLUENCE_USERNAME'), os.getenv('CONFLUENCE_PASSWORD'))) as response:
                if response.status == 200:
                    content = await response.json()
                    return content['body']['storage']['value']
                else:
                    logging.error(f"Chyba při načítání stránky {page_id}: HTTP {response.status}")
                    return None
    except aiohttp.ClientError as e:
        logging.error(f"Chyba při načítání stránky {page_id}: {e}")
        return None

# Získání stránek podle klíčových slov asynchronně
@rate_limited
async def search_pages(query):
    cache_key = hashlib.md5(query.encode('utf-8')).hexdigest()
    if cache_key in cache:
        return cache[cache_key]

    all_results = []
    start = 0
    limit = 50
    while True:
        try:
            results = confluence.cql(f'text ~ "{query}"', start=start, limit=limit)
            all_results.extend(results['results'])
            if len(results['results']) < limit:
                break
            start += limit
        except Exception as e:
            logging.error(f"Chyba při vyhledávání stránek: {e}")
            break

    cache[cache_key] = all_results
    return all_results

# Hlavní asynchronní funkce pro zodpovězení otázky
async def answer_question_async(question):
    processed_question = process_question_with_ner(question)
    search_results = await search_pages(processed_question)

    if search_results:
        response = "Našel jsem tyto relevantní stránky:\n"
        question_type = classify_question(question)
        tasks = [fetch_page_content(result['id']) for result in search_results[:3]]
        page_contents = await asyncio.gather(*tasks)

        for content, result in zip(page_contents, search_results[:3]):
            if content:
                if question_type == "list":
                    extracted_content = extract_specific_content(content, content_type='list')
                else:
                    extracted_content = extract_specific_content(content, content_type='text')
                response += f"- [{result['title']}]({result['_links']['base']}{result['_links']['webui']})\n"
                response += f"Výňatek ze stránky:\n```\n{extracted_content[:500]}...\n```\n"
        # Přidání doporučení na další kroky
        suggestion = suggest_next_steps(question)
        if suggestion:
            response += f"\nDoporučení: {suggestion}"
        return response
    else:
        return "Nenašel jsem žádné relevantní stránky."

# Doporučení na základě dotazu
def suggest_next_steps(question):
    suggestions = {
        "onboarding": "Chcete se podívat i na FAQ o onboardingu?",
        "training": "Máme také videomateriály k tomuto tématu.",
        "checklist": "Potřebujete další pomoc s checklistem?"
    }
    for keyword, suggestion in suggestions.items():
        if keyword in question.lower():
            return suggestion
    return None

# Odeslání odpovědi do MS Teams nebo Slack (Webhook)
def send_to_teams_or_slack(response, webhook_url):
    headers = {'Content-Type': 'application/json'}
    payload = {"text": response}
    try:
        response_post = requests.post(webhook_url, json=payload, headers=headers)
        if response_post.status_code == 200:
            logging.info("Odpověď byla úspěšně odeslána do MS Teams/Slack.")
        else:
            logging.error(f"Chyba při odesílání zprávy do MS Teams/Slack: HTTP {response_post.status_code}")
    except Exception as e:
        logging.error(f"Chyba při odesílání zprávy do MS Teams/Slack: {e}")

# Testovací funkce (základní testy s mockováním)
def run_tests():
    from unittest.mock import patch

    @patch('fetch_page_content')
    def test_get_page_content(mock_fetch_page_content):
        mock_fetch_page_content.return_value = "<html>Test Content</html>"
        response = asyncio.run(fetch_page_content(123))
        assert "Test Content" in response
        print("Test fetch_page_content prošel.")

    @patch('search_pages')
    def test_answer_question(mock_search_pages):
        mock_search_pages.return_value = [{'id': 1, 'title': 'Test Page', '_links': {'base': 'http://base.url', 'webui': '/page1'}}]
        mock_content = "<html><body>Test Content</body></html>"
        with patch('fetch_page_content', return_value=mock_content):
            response = asyncio.run(answer_question_async("Mám nějaký checklist, který musím na své pozici L2 splnit?"))
            assert "Test Page" in response
            assert "Test Content" in response
            print("Test answer_question prošel.")

    test_get_page_content()
    test_answer_question()

# Spuštění testů
if __name__ == '__main__':
    run_tests()
