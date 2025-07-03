import requests
import ccxt
import pandas as pd

# Configura il tuo token del bot
BOT_TOKEN = '7669555617:AAECVrKJ20HdbJPN7DzSDImh0LBTMGJCK18'

# Inserisci i due chat_id a cui vuoi inviare il messaggio
chat_ids = [
    '132642281',
    '514488413'
]

def fetch_live_data(symbol, timeframe='1h', lookback_hours=72, limit=1000):
    exch = ccxt.binance()
    since = exch.milliseconds() - lookback_hours * 60 * 60 * 1000
    try:
        ohlcv = exch.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
    except Exception as e:
        print(f"Errore fetch OHLCV: {e}")
        return None
    
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

def send_message(chat_id, text):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {
        'chat_id': chat_id,
        'text': text
    }
    response = requests.post(url, data=payload)
    if response.status_code == 200:
        print(f"Messaggio inviato a {chat_id}")
    else:
        print(f"Errore nell'invio a {chat_id}: {response.text}")

# Invia il messaggio a entrambi i chat_id
for chat_id in chat_ids:
    send_message(chat_id, fetch_live_data('SOL/USDT', timeframe='1h', lookback_hours=72))
