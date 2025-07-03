import requests

# Configura il tuo token del bot
BOT_TOKEN = '7669555617:AAECVrKJ20HdbJPN7DzSDImh0LBTMGJCK18'

# Inserisci i due chat_id a cui vuoi inviare il messaggio
chat_ids = [
    '132642281',
    '514488413'
]

def get_btt_ticker():
    url = "https://api.binance.com/api/v3/ticker/24hr"
    params = {
        "symbol": "BTTUSDT"
    }
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        data = response.json()
        last_price = data['lastPrice']
        return last_price
        print("Prezzo attuale BTT/USDT:", last_price)
    else:
        print("Errore nella richiesta:", response.status_code)

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
    send_message(chat_id, get_btt_ticker())
