import json
import pandas as pd
from datetime import datetime
from langchain_anthropic import ChatAnthropic
from langchain.schema import HumanMessage, SystemMessage
from langchain.memory import ConversationBufferMemory
from langchain_core.runnables import RunnableBranch, RunnableLambda, RunnablePassthrough
from operator import itemgetter
from forecast import ProphetForecaster
import warnings
warnings.filterwarnings('ignore')


# ============== КОНФИГУРАЦИЯ ==============
with open('key.txt', 'r') as f:
    ANTHROPIC_API_KEY = f.read().strip()

DATA_FILES = {
    'sales': 'data/sales.csv',
    'price': 'data/price.csv', 
    'trips': 'data/trips.csv'
}

# ============== ЗАГРУЗКА ПРОМПТОВ ==============
def load_prompt(filename):
    with open(f'prompts/{filename}', 'r', encoding='utf-8') as f:
        return f.read().strip()

CLASSIFIER_PROMPT = load_prompt('classifier.txt')
CONVERSATION_PROMPT = load_prompt('chat.txt')
FORECAST_RESPONSE_PROMPT = load_prompt('forecast_response.txt')

# ============== ИНИЦИАЛИЗАЦИЯ ==============
llm = ChatAnthropic(
    model="claude-haiku-4-5",
    temperature=0.7,
    anthropic_api_key=ANTHROPIC_API_KEY,
    #base_url="https://api.proxyapi.ru/anthropic",
    max_tokens=1000
)

# Память для истории диалога
memory = ConversationBufferMemory(return_messages=True)

# ============== ФУНКЦИИ ДЛЯ ПОЛУЧЕНИЯ ДАННЫХ ==============
def process_forecast_request_1(parsed):
    """Основная функция обработки запроса на данные"""
    indicator = parsed['indicator']
    
    # Инициализируем форкастер
    forecaster = ProphetForecaster(indicator)
    
    if parsed.get('date'):
        # Убеждаемся что есть прогноз до нужной даты
        forecaster.ensure_forecast_until(parsed['date'])
        # Получаем данные (прогноз автоматически вернет None для прошлых дат)
        result = forecaster.get_forecast_for_date(parsed['date'])
        
    elif parsed.get('period'):
        # Парсим период "YYYY-MM-DD - YYYY-MM-DD"
        end_date = parsed['period'].split(' - ')[1]
        forecaster.ensure_forecast_until(end_date)
        result = forecaster.get_forecast_for_period(*parsed['period'].split(' - '))
    
    if result:
        print('process_forecast_request')
        print(result)
        # Добавляем тип ответа (история/прогноз определяется внутри forecast_v2)
        return {**result, 'type': 'history' if 'date' in result and pd.to_datetime(result['date']).date() < datetime.now().date() else 'forecast'}
    
    return {'type': 'error', 'message': 'Не удалось получить данные'}

def process_forecast_request(parsed):
    """Основная функция обработки запроса на данные"""
    indicator = parsed['indicator']
    
    # Инициализируем форкастер
    forecaster = ProphetForecaster(indicator)
    
    if parsed.get('date'):
        forecaster.ensure_forecast_until(parsed['date'])
        result = forecaster.get_forecast_for_date(parsed['date'])
        if result:
            # Определяем тип по дате
            target_date = pd.to_datetime(parsed['date']).date()
            is_past = target_date < datetime.now().date()
            return {**result, 'type': 'history' if is_past else 'forecast'}
    
    elif parsed.get('period'):
        start_date, end_date = parsed['period'].split(' - ')
        end = pd.to_datetime(end_date).date()
        forecaster.ensure_forecast_until(end_date)
        result = forecaster.get_forecast_for_period(start_date, end_date)
        if result:
            # Определяем тип по концу периода
            is_past = end < datetime.now().date()
            return {**result, 'type': 'history_period' if is_past else 'forecast_period'}
    
    return {'type': 'error', 'message': 'Не удалось получить данные'}

# ============== ОСНОВНЫЕ ОБРАБОТЧИКИ ==============
def classifier_step(inputs):
    """Классифицирует запрос пользователя"""
    query = inputs['query']
    today_str = datetime.now().strftime("%Y-%m-%d")
    day_of_week_ru = ['понедельник', 'вторник', 'среда', 'четверг', 'пятница', 'суббота', 'воскресенье'][datetime.now().weekday()]
    
    enhanced_prompt = CLASSIFIER_PROMPT + f"\n\nСегодня: {today_str} ({day_of_week_ru}). Учитывай это при определении 'сегодня', 'завтра', 'вчера'."
    
    messages = [
        SystemMessage(content=enhanced_prompt),
        HumanMessage(content=query)
    ]
    
    response = llm.invoke(messages)
    result_text = response.content
        
    # Очищаем ответ от возможных markdown
    result_text = result_text.strip().strip('`').replace('json\n', '').replace('\n', '')

    try:
        result = json.loads(result_text)
        print('classifier_step')
        print(result)
        print('return')
        print({**inputs, 'intent': result['type'], 'parsed': result})
        return {**inputs, 'intent': result['type'], 'parsed': result}
    except json.JSONDecodeError:
        print(f"Не удалось распарсить JSON: {result_text[:100]}...")

    
def handle_forecast(inputs):
    """Обработчик прогнозов/истории"""
    parsed = inputs['parsed']
    result = process_forecast_request(parsed)
    print('handle_forecast')
    print(result)
    
    if result['type'] == 'error':
        return {**inputs, 'response_data': result, 'need_llm_response': False}
    
    return {**inputs, 'response_data': result, 'need_llm_response': True}

def handle_conversation(inputs):
    """Обработчик беседы"""
    history = memory.load_memory_variables({})['history']
    
    messages = [
        SystemMessage(content=CONVERSATION_PROMPT),
        HumanMessage(content=f"История диалога: {history}\n\nЗапрос пользователя: {inputs['query']}")
    ]
    
    response = llm.invoke(messages)
    return {**inputs, 'direct_response': response.content, 'need_llm_response': False}

def format_forecast_response(inputs):
    """Формирует ответ на основе данных прогноза/истории"""
    response_data = inputs['response_data']
    
    messages = [
        SystemMessage(content=FORECAST_RESPONSE_PROMPT),
        HumanMessage(content=json.dumps(response_data, ensure_ascii=False, indent=2))
    ]
    
    response = llm.invoke(messages)
    return response.content

# ============== ПОСТРОЕНИЕ ЦЕПОЧКИ ==============
def build_chain():
    """Строит цепочку с роутингом"""
    
    # Прогноз/история -> получение данных -> формирование ответа
    forecast_chain = (
        RunnablePassthrough.assign(
            processed=RunnableLambda(handle_forecast)
        )
        | RunnableLambda(lambda x: format_forecast_response(x['processed']) 
                         if x['processed']['need_llm_response'] 
                         else x['processed']['response_data']['message'])
    )
    
    # Беседа -> прямой ответ
    conversation_chain = RunnableLambda(handle_conversation) | itemgetter('direct_response')
    
    # Fallback для неизвестных интентов
    fallback_chain = RunnableLambda(lambda x: "Извините, не могу обработать этот запрос.")
    
    # Основной роутер
    router = RunnableBranch(
        (lambda x: x['intent'] in ['forecast', 'history'], forecast_chain),
        (lambda x: x['intent'] in ['greeting', 'small_talk'], conversation_chain),
        fallback_chain
    )
    
    # Полная цепочка
    return classifier_step | router

# ============== ОСНОВНАЯ ФУНКЦИЯ ==============
def process_query(query):
    """Главная функция обработки запроса"""
    
    # Сохраняем запрос в историю
    memory.chat_memory.add_user_message(query)
    
    # Запускаем цепочку
    chain = build_chain()
    response = chain.invoke({'query': query})
    
    # Сохраняем ответ в историю
    memory.chat_memory.add_ai_message(response)
    
    return response


# ============== ТОЧКА ВХОДА ==============
if __name__ == "__main__":
    print("Умный ассистент по прогнозированию")
    print("="*60)
    print("Я могу:")
    print("  • Прогнозировать продажи/цену/поездки")
    print("  • Отвечать на вопросы о прошлых данных")
    print("  • Поддерживать диалог и помнить историю")
    print("="*60)
    
    while True:
        query = input("\nВы: ")
        
        if query.lower() in ['выход', 'exit', 'quit', 'q']:
            print("До свидания!")
            break
        
        if query.strip():
            response = process_query(query)
            print(f"\nАссистент: {response}")
