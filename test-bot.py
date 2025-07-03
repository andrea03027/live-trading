import requests

# Configura il tuo token del bot
BOT_TOKEN = 'YOUR_BOT_TOKEN_HERE'

# Inserisci i due chat_id a cui vuoi inviare il messaggio
chat_ids = [
    'CHAT_ID_1',
    'CHAT_ID_2'
]

# Messaggio da inviare
message = "Ciao! Questo è un messaggio automatico dal bot."

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
    send_message(chat_id, message)
