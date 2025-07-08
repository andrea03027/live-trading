import ccxt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import itertools
import time
import os
import requests

# === PARAMETRI INIZIALI ===
symbols = ['SOL/USDT']
timeframe = '1h'
capital_per_trade = 5000
commission = 6.0

# Intervallo backtest
start_date = '2025-01-01'
end_date = '2025-06-30'

# Configura il tuo token del bot
BOT_TOKEN = '7669555617:AAECVrKJ20HdbJPN7DzSDImh0LBTMGJCK18'

# Inserisci i due chat_id a cui vuoi inviare il messaggio
chat_ids = [
    '132642281',
    '514488413'
]

def send_telegram(chat_id, message):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    r = requests.post(url,
                      data={
                          'chat_id': chat_id,
                          'text': message,
                          'parse_mode': 'Markdown'
                      })
    if not r.ok:
        print("Errore invio Telegram:", r.text)

def send_telegram_photo(chat_id, photo_path, cSOLion=""):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto"
    with open(photo_path, "rb") as photo:
        requests.post(url, data={"chat_id": chat_id, "cSOLion": cSOLion}, files={"photo": photo})


# === FUNZIONI INDICATORI ===
def ema(series, span):
    return series.ewm(span=int(span), adjust=False).mean()

def atr(df, period=14):
    period = int(period)
    hl = df['high'] - df['low']
    hc = abs(df['high'] - df['close'].shift())
    lc = abs(df['low'] - df['close'].shift())
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def rsi(series, period=14):
    period = int(period)
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = -delta.clip(upper=0).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# === FETCH DATI COMPLETI ===
def fetch_data_full(symbol, start_date, end_date):
    exch = ccxt.binance()
    since = exch.parse8601(start_date + 'T00:00:00Z')
    end_dt = pd.to_datetime(end_date)
    all_chunks = []

    while True:
        ohlcv = exch.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=1000)
        if not ohlcv:
            break
        dfc = pd.DataFrame(ohlcv, columns=['timestamp','open','high','low','close','volume'])
        dfc['timestamp'] = pd.to_datetime(dfc['timestamp'], unit='ms')
        all_chunks.append(dfc)
        last_time = dfc['timestamp'].iloc[-1]
        if last_time >= end_dt:
            break
        since = int(last_time.timestamp() * 1000) + 1

    df = pd.concat(all_chunks, ignore_index=True)
    df = df[(df['timestamp'] >= pd.to_datetime(start_date)) & (df['timestamp'] <= pd.to_datetime(end_date))]
    return df.reset_index(drop=True)

# === CALCOLO INDICATORI ===
def calculate_indicators(df, ema_fast_len, ema_slow_len, rsi_len, atr_period):
    df['EMA_fast'] = ema(df['close'], ema_fast_len)
    df['EMA_slow'] = ema(df['close'], ema_slow_len)
    df['RSI'] = rsi(df['close'], rsi_len)
    df['ATR'] = atr(df, atr_period)
    return df.dropna().reset_index(drop=True)

# === BACKTEST CON POSIZIONI MULTIPLE ===
def backtest(df, symbol, atr_sl_factor, atr_tp_factor, min_ema_diff_pct):
    trades = []
    positions = []  # lista posizioni aperte: dict con info
    capital = capital_per_trade

    for i in range(1, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i-1]

        price = row['close']
        atr_val = row['ATR']
        ema_diff_pct = abs(row['EMA_fast'] - row['EMA_slow']) / price

        sl = atr_sl_factor * atr_val
        tp = atr_tp_factor * atr_val
        potential_tp_usd = tp * capital / price

        # Verifica chiusura posizioni aperte
        still_positions = []
        for pos in positions:
            pos_type = pos['type']
            entry_price = pos['entry_price']
            entry_time = pos['entry_time']

            pnl = None
            if pos_type == 'long':
                if price >= entry_price + tp or price <= entry_price - sl:
                    pnl = price - entry_price
            elif pos_type == 'short':
                if price <= entry_price - tp or price >= entry_price + sl:
                    pnl = entry_price - price

            if pnl is not None:
                pnl_pct = pnl / entry_price * 100
                gross_usd = pnl_pct / 100 * capital
                net_usd = gross_usd - commission
                duration = (row['timestamp'] - entry_time).total_seconds() / 60
                #print(f"🔻 EXIT {pos_type.upper()} | Entry: {entry_price:.2f} @ {entry_time} -> Exit: {price:.2f} @ {row['timestamp']}")
                #print(f"     PnL: {pnl:.2f} USD | PnL%: {pnl_pct:.2f}% | Profit (net): {net_usd:.2f} USD | Duration: {duration:.1f} min")
                trades.append({
                    'symbol': symbol,
                    'type': pos_type,
                    'entry_time': entry_time,
                    'exit_time': row['timestamp'],
                    'entry_price': entry_price,
                    'exit_price': price,
                    'pnl': pnl,
                    'pnl_%': pnl_pct,
                    'profit_usd': net_usd,
                    'duration_min': duration
                })
            else:
                still_positions.append(pos)
        positions = still_positions

        # Condizioni apertura nuove posizioni (SOLo se differenza EMA sufficiente e profitto potenziale > commissioni)
        if ema_diff_pct >= min_ema_diff_pct and potential_tp_usd > commission * 3:
            # Long entry
            if prev['EMA_fast'] < prev['EMA_slow'] and row['EMA_fast'] > row['EMA_slow']:
                #print(f"🟢 ENTER LONG @ {price:.2f} on {row['timestamp']}")
                positions.append({
                    'type': 'long',
                    'entry_price': price,
                    'entry_time': row['timestamp']
                })
            # Short entry
            elif prev['EMA_fast'] > prev['EMA_slow'] and row['EMA_fast'] < row['EMA_slow']:
                #print(f"🔴 ENTER SHORT @ {price:.2f} on {row['timestamp']}")
                positions.append({
                    'type': 'short',
                    'entry_price': price,
                    'entry_time': row['timestamp']
                })

    return pd.DataFrame(trades)

# === METRICHE DI PERFORMANCE ===
def calculate_metrics(trades_df):
    total_profit = trades_df['profit_usd'].sum()
    win_rate = (trades_df['pnl'] > 0).mean()
    if trades_df.empty:
        max_drawdown = np.nan
        sharpe_ratio = np.nan
    else:
        equity_curve = trades_df['profit_usd'].cumsum()
        max_drawdown = (equity_curve.cummax() - equity_curve).max()
        returns = trades_df['profit_usd'] / capital_per_trade
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() != 0 else np.nan
    return {
        'total_profit_usd': total_profit,
        'win_rate': win_rate,
        'max_drawdown_usd': max_drawdown,
        'sharpe_ratio': sharpe_ratio
    }

# === DIVISIONE IN MESI ===
def generate_month_range(start_date: str, end_date: str):
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    current = start.replace(day=1)
    
    months = []
    while current <= end:
        next_month = (current.replace(day=28) + timedelta(days=4)).replace(day=1)
        month_end = (next_month - timedelta(days=1)).strftime("%Y-%m-%d")
        months.append((current.strftime("%Y-%m-%d"), month_end))
        current = next_month
    return months

# === MAIN CON CICLO MENSILE ===
def main():
    ema_fast_list = [8, 9, 10]
    ema_slow_list = [20, 21, 22]
    rsi_len_list = [13, 14, 15]
    atr_period_list = [13, 14, 15]
    atr_sl_factor_list = [0.4, 0.5, 0.6]
    atr_tp_factor_list = [2.5, 3.0, 3.5]
    min_ema_diff_pct = 0.001

    base_folder = "results"
    os.makedirs(base_folder, exist_ok=True)

    # Itera su ogni mese
    for month_start, month_end in generate_month_range(start_date, end_date):
        print(f"\n📅 Analisi mensile: {month_start} -> {month_end}")
        monthly_results = []

        for symbol in symbols:
            print(f"Fetching data {symbol} for {month_start} to {month_end}...")
            df = fetch_data_full(symbol, month_start, month_end)
            if df.empty:
                print(f"No data for {symbol}")
                continue

            for ema_fast_len, ema_slow_len, rsi_len, atr_period, atr_sl_factor, atr_tp_factor in itertools.product(
                ema_fast_list, ema_slow_list, rsi_len_list, atr_period_list, atr_sl_factor_list, atr_tp_factor_list):

                df_ind = calculate_indicators(df.copy(), ema_fast_len, ema_slow_len, rsi_len, int(atr_period))
                trades = backtest(df_ind, symbol, atr_sl_factor, atr_tp_factor, min_ema_diff_pct)

                if not trades.empty:
                    metrics = calculate_metrics(trades)
                    monthly_results.append({
                        'symbol': symbol,
                        'start': month_start,
                        'end': month_end,
                        'ema_fast_len': ema_fast_len,
                        'ema_slow_len': ema_slow_len,
                        'rsi_len': rsi_len,
                        'atr_period': atr_period,
                        'atr_sl_factor': atr_sl_factor,
                        'atr_tp_factor': atr_tp_factor,
                        **metrics
                    })

        # Salvataggio risultati del mese
        if monthly_results:
            df_results = pd.DataFrame(monthly_results)
            best = df_results.loc[df_results['total_profit_usd'].idxmax()]
            print(f"\n🏆 Best params {symbol} | {month_start}–{month_end}:")
            print(best)

            # Ricalcolo indicatori con best strategy e salvataggio dei grafici
            df_ind = calculate_indicators(df.copy(), best['ema_fast_len'], best['ema_slow_len'], best['rsi_len'], int(best['atr_period']))
            trades = backtest(df_ind, symbol, best['atr_sl_factor'], best['atr_tp_factor'], min_ema_diff_pct)

            df_ind['entry_signal'] = False
            df_ind['exit_signal'] = False
            for _, t in trades.iterrows():
                entry_idx = df_ind.index[df_ind['timestamp'] == t['entry_time']].tolist()
                exit_idx = df_ind.index[df_ind['timestamp'] == t['exit_time']].tolist()
                if entry_idx:
                    df_ind.at[entry_idx[0], 'entry_signal'] = True
                if exit_idx:
                    df_ind.at[exit_idx[0], 'exit_signal'] = True

            # === CARTELLA DI SALVATAGGIO ===
            out_dir = os.path.join(base_folder, f"{symbol.replace('/', '_')}_{month_start}_to_{month_end}")
            os.makedirs(out_dir, exist_ok=True)

            # Salva file Excel con i trade
            trades.to_excel(os.path.join(out_dir, "trades.xlsx"), index=False)

            # Salva equity curve
            trades_sorted = trades.sort_values(by='exit_time').reset_index(drop=True)
            trades_sorted['cumulative_profit'] = trades_sorted['profit_usd'].cumsum()
            plt.figure(figsize=(12, 6))
            plt.plot(trades_sorted['exit_time'], trades_sorted['cumulative_profit'], label='Equity Curve', color='blue')
            plt.title(f"{symbol} - Equity Curve\n{month_start} to {month_end}")
            plt.xlabel("Exit Time")
            plt.ylabel("Cumulative Profit (USD)")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "equity_curve.png"))
            plt.close()

            # Salva grafico segnali
            plt.figure(figsize=(14,7))
            plt.plot(df_ind['timestamp'], df_ind['close'], label='Close Price')
            plt.plot(df_ind['timestamp'], df_ind['EMA_fast'], label=f'EMA {int(best["ema_fast_len"])}')
            plt.plot(df_ind['timestamp'], df_ind['EMA_slow'], label=f'EMA {int(best["ema_slow_len"])}')
            plt.scatter(df_ind.loc[df_ind['entry_signal'], 'timestamp'], df_ind.loc[df_ind['entry_signal'], 'close'], marker='^', color='green', label='Entry')
            plt.scatter(df_ind.loc[df_ind['exit_signal'], 'timestamp'], df_ind.loc[df_ind['exit_signal'], 'close'], marker='v', color='red', label='Exit')
            plt.title(f"{symbol} - Entry/Exit Signals\n{month_start} to {month_end}")
            plt.xlabel("Time")
            plt.ylabel("Price")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "entry_exit_signals.png"))
            plt.close()

            # Salva strategia vincente in CSV cumulativo
            df_results.to_csv(os.path.join(base_folder, "all_monthly_results.csv"), mode='a', header=not os.path.exists(os.path.join(base_folder, "all_monthly_results.csv")), index=False)

def load_strategies_from_csv(path="results_SOL/all_monthly_results.csv"):
    df = pd.read_csv(path)
    grouped = df.groupby(['ema_fast_len', 'ema_slow_len', 'rsi_len', 'atr_period', 'atr_sl_factor', 'atr_tp_factor']).mean(numeric_only=True)
    return grouped.reset_index()

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

def test_strategies_live(df_live, strategies_df, symbol):
    results = []

    for _, row in strategies_df.iterrows():
        df_ind = calculate_indicators(df_live.copy(),
                                       int(row['ema_fast_len']),
                                       int(row['ema_slow_len']),
                                       int(row['rsi_len']),
                                       int(row['atr_period']))
        
        trades = backtest(df_ind, symbol,
                          row['atr_sl_factor'],
                          row['atr_tp_factor'],
                          min_ema_diff_pct=0.001)

        if trades.empty:
            # crea un DataFrame con le colonne previste ma vuoto, o imposta metriche di default
            metrics = {
            'total_profit_usd': 0,
            'win_rate': 0,
            'max_drawdown_usd': 0,
            'sharpe_ratio': 0
                }
        else:
            metrics = calculate_metrics(trades)
        results.append({
                **row,
                **metrics
        })

    return pd.DataFrame(results).sort_values(by="total_profit_usd", ascending=False)

def evaluate_best_live_strategy():
    symbol = 'SOL/USDT'
    strategies_df = load_strategies_from_csv()
    df_live = fetch_live_data(symbol)
    
    print("Testing strategies on live data...")
    ranked = test_strategies_live(df_live, strategies_df, symbol)
    
    print("\n📈 Top strategy to apply now:")
    print(ranked.head(1).T)

    return ranked

def apply_and_plot_best_live_strategy():

    # Carica la strategia migliore live
    path = "results_SOL/live_strategy_ranking.csv"
    if not os.path.exists(path):
        print("⚠️ File live_strategy_ranking.csv non trovato.")
        return

    ranked = pd.read_csv(path)
    if ranked.empty:
        print("⚠️ Nessuna strategia live trovata.")
        return

    best = ranked.iloc[0]
    print("\n📊 Migliore strategia live trovata:")
    print(best)

    # Fetch live data
    symbol = 'SOL/USDT'
    df_live = fetch_live_data(symbol)

    # Calcolo indicatori
    df_ind = calculate_indicators(
        df_live.copy(),
        int(best['ema_fast_len']),
        int(best['ema_slow_len']),
        int(best['rsi_len']),
        int(best['atr_period'])
    )

    # Applica la strategia (backtest live per generare entry/exit anche ora)
    trades = backtest(
        df_ind,
        symbol,
        best['atr_sl_factor'],
        best['atr_tp_factor'],
        min_ema_diff_pct=0.001
    )

    # Aggiungi colonne segnali
    df_ind['entry_signal'] = False
    df_ind['exit_signal'] = False
    df_ind['entry_type'] = None
    df_ind['exit_type'] = None

    for i in range(1, len(df_ind)):
        prev = df_ind.iloc[i - 1]
        curr = df_ind.iloc[i]

        # Golden Cross
        if prev['EMA_fast'] < prev['EMA_slow'] and curr['EMA_fast'] > curr['EMA_slow']:
            df_ind.at[i, 'cross_type'] = 'Golden Cross'

        # Death Cross
        elif prev['EMA_fast'] > prev['EMA_slow'] and curr['EMA_fast'] < curr['EMA_slow']:
            df_ind.at[i, 'cross_type'] = 'Death Cross'
        

    for _, t in trades.iterrows():
        entry_idx = df_ind.index[df_ind['timestamp'] == t['entry_time']].tolist()
        exit_idx = df_ind.index[df_ind['timestamp'] == t['exit_time']].tolist() if pd.notna(t['exit_time']) else []

        if entry_idx:
            df_ind.at[entry_idx[0], 'entry_signal'] = True
            df_ind.at[entry_idx[0], 'entry_type'] = t['type'].upper()
        if exit_idx:
            df_ind.at[exit_idx[0], 'exit_signal'] = True
            df_ind.at[exit_idx[0], 'exit_type'] = t['type'].upper()

    # Plot
    plt.figure(figsize=(14, 7))
    plt.plot(df_ind['timestamp'], df_ind['close'], label='Close Price', color='black')
    plt.plot(df_ind['timestamp'], df_ind['EMA_fast'], label=f"EMA {int(best['ema_fast_len'])}", color='blue')
    plt.plot(df_ind['timestamp'], df_ind['EMA_slow'], label=f"EMA {int(best['ema_slow_len'])}", color='orange')

    # Entry signals
    for idx, row in df_ind[df_ind['entry_signal']].iterrows():
        plt.scatter(row['timestamp'], row['close'], marker='^', color='green', s=120, label='Entry' if idx == 0 else "")
        plt.text(row['timestamp'], row['close'] * 1.01, row['entry_type'], color='green', fontsize=9, ha='center')

    # Exit signals
    for idx, row in df_ind[df_ind['exit_signal']].iterrows():
        plt.scatter(row['timestamp'], row['close'], marker='v', color='red', s=120, label='Exit' if idx == 0 else "")
        plt.text(row['timestamp'], row['close'] * 0.99, row['exit_type'], color='red', fontsize=9, ha='center')

    # Frecce per trade chiusi
    for _, trade in trades.iterrows():
        entry_time = trade['entry_time']
        entry_price = trade['entry_price']
        pnl_pct = trade['pnl_%']

        if pd.notna(trade['exit_time']):
            exit_time = trade['exit_time']
            exit_price = trade['exit_price']
            color = 'green' if pnl_pct > 0 else 'red'

            plt.annotate(
                '', 
                xy=(exit_time, exit_price), 
                xytext=(entry_time, entry_price),
                arrowprops=dict(facecolor=color, edgecolor=color, arrowstyle='->', lw=2, alpha=0.6)
            )

            # Testo vicino alla freccia
            mid_time = entry_time + (exit_time - entry_time) / 2
            mid_price = entry_price + (exit_price - entry_price) / 2
            plt.text(
                mid_time, 
                mid_price * (1.01 if pnl_pct > 0 else 0.99), 
                f"{pnl_pct:.2f}%", 
                color=color, 
                fontsize=9, 
                ha='center'
            )

    # Frecce per trade aperti (exit_time = NaN)
    if 'exit_time' in trades.columns:
        open_trades = trades[trades['exit_time'].isna()]
    else:
        print("Column 'exit_time' not found in trades DataFrame.")
        open_trades = pd.DataFrame()  # or handle however makes sense
    if not open_trades.empty:
        last_time = df_ind['timestamp'].iloc[-1]
        last_price = df_ind['close'].iloc[-1]

        for _, trade in open_trades.iterrows():
            entry_time = trade['entry_time']
            entry_price = trade['entry_price']
            trade_type = trade['type']
            # Colore blu per trade aperti
            color = 'blue'

            plt.annotate(
                '', 
                xy=(last_time, last_price), 
                xytext=(entry_time, entry_price),
                arrowprops=dict(facecolor=color, edgecolor=color, arrowstyle='-|>', linestyle='dashed', lw=2, alpha=0.6)
            )

            # Testo vicino alla freccia
            mid_time = entry_time + (last_time - entry_time) / 2
            mid_price = entry_price + (last_price - entry_price) / 2
            plt.text(
                mid_time, 
                mid_price * 1.01, 
                f"OPEN {trade_type.upper()}", 
                color=color, 
                fontsize=9, 
                ha='center'
            )
    latest_cross = df_ind.iloc[-1]
    if 'cross_type' in latest_cross.index:
        if pd.notnull(latest_cross['cross_type']):
            message = (
            f"🔔 Segnale Live SOL!\n"
            f"Tipo posizione: {latest_cross['cross_type']}\n"
            f"⏰ Orario: {latest_cross['timestamp'].strftime('%Y-%m-%d %H:%M')}\n"
            f"💰 Prezzo: {latest_cross['close']:.2f}"
            )
            plt.title("🔍 Live SignalSOLTEVL/USDT (Long/Short, incl. Open Trades)")
            plt.xlabel("Time")
            plt.ylabel("Price (USDT)")
            plt.grid(True)
            plt.savefig("grafico.png")
            for chat_id in chat_ids:
                send_telegram_photo(chat_id, "grafico.png", cSOLion=message)

            color = 'green' if latest_cross['cross_type'] == 'Golden Cross' else 'red'
            label = f"{latest_cross['cross_type']}\n{latest_cross['timestamp'].strftime('%H:%M')}"
            plt.axvline(x=latest_cross['timestamp'], color=color, linestyle='--', alpha=0.3)
            plt.text(
                latest_cross['timestamp'], latest_cross['close'], label,
                color=color, fontsize=8, ha='left', va='bottom', rotation=90
            )
    
    plt.legend()
    plt.tight_layout()
    plt.show()
    #plt.close()


if __name__ == "__main__":
    while True:
        ranked_live = evaluate_best_live_strategy()
        ranked_live.to_csv("results_SOL/live_strategy_ranking.csv", index=False)
        apply_and_plot_best_live_strategy()
        print("Attesa 15 minuti per il prossimo aggiornamento...\n")
        time.sleep(900)
