# Copyright 2026, MetaQuotes Ltd.
# https://www.mql5.com

import MetaTrader5 as mt5
import pandas as pd
import ta, logging, time, os, json, requests, traceback
from logging.handlers import RotatingFileHandler


SYMBOL = "XAUUSD_i"
TIMEFRAME_M5 = mt5.TIMEFRAME_M5
TIMEFRAME_M15 = mt5.TIMEFRAME_M15
TIMEFRAME_H1 = mt5.TIMEFRAME_H1
START_POS = 0
COUNT = 300
FAST = 5
SLOW = 20
#TAKE_PROFIT = 4000
#STOP_L0SS = 2000
DEVIATION = 20
LOT_SIZE = 0.01
POLL_SEC = 2
TG_BOT_TOKEN = "8647396123:AAGF2EjBEQqw8N0DuIwpt0c-m3qhS5xoOUU"
CHAT_ID = "@SignalCamptf" 

m5_index = None 
m5_open = None
m5_close = None
last_m5_open_time = None
x = None 

LOG_FILE = "mt5_bot.log"
STATE_FILE = "live_state.json"
HFT_STATE_FILE = "hft_live_state.json"


# ====== Logging configuration =========
log = logging.getLogger("mt5_bot_logger")
log.setLevel(logging.INFO)
fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
fh = RotatingFileHandler(LOG_FILE, maxBytes=1_000_000, backupCount=3)
fh.setFormatter(fmt)
log.addHandler(fh)
sh = logging.StreamHandler()
sh.setFormatter(fmt)
log.addHandler(sh)

def send_telegram_message(message="signal", bot_token=TG_BOT_TOKEN, chat_id=CHAT_ID):
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message,
        
    }
    try:
        response = requests.post(url, json=payload)
        if response.status_code != 200:
            log.error(f"Failed to send Telegram message: {response.text}")
    except Exception as e:
        log.error(f"Exception while sending Telegram message: {e}")

#========= PERSISTENCE =========
def save_state(state, path=STATE_FILE):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f)
    os.replace(tmp, path)
    log.info(f"State saved: {state}")

def load_state(path=STATE_FILE):
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r") as f:
            state = json.load(f)
        log.info(f"State loaded: {state}")
        return state
    except Exception as e:
        log.error(f"Failed to load state: {e}")
        return None

def clear_state(path=STATE_FILE):
    try:
        if os.path.exists(path): os.remove(path)
        log.info("State cleared.")
    except Exception as e:
        log.error(f"Failed to clear state: {e}")

# ========= MetaTrader 5 Initialization and Data Fetching =========
def mt5_initialize():

    if not mt5.initialize():
        log.error("Failed to initialize MetaTrader 5")
        return False
    return True

# ======== Trade Strategies ========
def m5_long():

    # === calculate 5mins timeframe indicators ===
    m5_fast_ma, m5_slow_ma, m5_rsi, m5_open, m5_high, m5_low, m5_close, m5_volume, _, m5_stoch_k, m5_stoch_d, m5_adx, m5_adx_pos, m5_adx_neg, m5_candle_body, m5_avg_body = fetch_df(SYMBOL, TIMEFRAME_M5, START_POS, COUNT, FAST, SLOW)
    #_, _, h1_rsi, h1_open, h1_high, h1_low, h1_close, h1_volume, _, h1_stoch_k, h1_stoch_d, h1_adx, h1_adx_pos, h1_adx_neg = fetch_df(SYMBOL, TIMEFRAME_H1, START_POS, COUNT, FAST, SLOW)
    #_, _, m15_rsi, m15_open, m15_high, _, m15_close, m15_volume, _, m15_stoch_k, m15_stoch_d, m15_adx, m15_adx_pos, m15_adx_neg = fetch_df(SYMBOL, TIMEFRAME_M15, START_POS, COUNT, FAST, SLOW)

    # === the strategy ===
    #if m5_fast_ma.iloc[-2] > m5_slow_ma.iloc[-2] and m5_stoch_k.iloc[-2] < 80 and m5_stoch_k.iloc[-2] > 20 and m5_rsi.iloc[-2] < 70 and m5_rsi.iloc[-2] > 30 and m5_close.iloc[-1] > m5_open.iloc[-1] and m5_candle_body.iloc[-1] > 1.5 * m5_avg_body.iloc[-1] and m5_stoch_k.iloc[-1] > m5_stoch_d.iloc[-1] and m5_volume.iloc[-1] > m5_volume.rolling(14).mean().iloc[-1]: 
    if m5_fast_ma.iloc[-2] > m5_slow_ma.iloc[-2] and m5_rsi.iloc[-2] < 70 and m5_rsi.iloc[-2] > 30 and m5_adx.iloc[-2] > 20 and m5_close.iloc[-1] > m5_open.iloc[-1] and m5_candle_body.iloc[-1] > 1.5 * m5_avg_body.iloc[-1] and m5_stoch_k.iloc[-1] > m5_stoch_d.iloc[-1] and m5_volume.iloc[-1] > m5_volume.rolling(14).mean().iloc[-1]: 

        return True
    
    else: return False 

def m5_short():

    # === calculate 5mins timeframe indicators ===
    m5_fast_ma, m5_slow_ma, m5_rsi, m5_open, _, m5_low, m5_close, m5_volume, _, m5_stoch_k, m5_stoch_d, m5_adx, _, _, m5_candle_body, m5_avg_body = fetch_df(SYMBOL, TIMEFRAME_M5, START_POS, COUNT, FAST, SLOW)
    #_, _, h1_rsi, _, _, _, _, _, _, _, _, _, _, _ = fetch_df(SYMBOL, TIMEFRAME_H1, START_POS, COUNT, FAST, SLOW)
    #_, _, _, m15_open, _, m15_low, m15_close, m15_volume, _, m15_stoch_k, m15_stoch_d, m15_adx, m15_adx_pos, m15_adx_neg = fetch_df(SYMBOL, TIMEFRAME_M15, START_POS, COUNT, FAST, SLOW)

    # === the strategy === 
    #if m5_fast_ma.iloc[-2] < m5_slow_ma.iloc[-2] and m5_stoch_k.iloc[-2] < 80 and m5_stoch_k.iloc[-2] > 20 and m5_rsi.iloc[-2] < 70 and m5_rsi.iloc[-2] > 30 and m5_close.iloc[-1] < m5_open.iloc[-1] and m5_candle_body.iloc[-1] > 1.5 * m5_avg_body.iloc[-1] and m5_stoch_k.iloc[-1] < m5_stoch_d.iloc[-1] and m5_volume.iloc[-1] > m5_volume.rolling(14).mean().iloc[-1]: 
    if m5_fast_ma.iloc[-2] < m5_slow_ma.iloc[-2] and m5_rsi.iloc[-2] < 70 and m5_rsi.iloc[-2] > 30 and m5_adx.iloc[-2] > 20 and m5_close.iloc[-1] < m5_open.iloc[-1] and m5_candle_body.iloc[-1] > 1.5 * m5_avg_body.iloc[-1] and m5_stoch_k.iloc[-1] < m5_stoch_d.iloc[-1] and m5_volume.iloc[-1] > m5_volume.rolling(14).mean().iloc[-1]: 

        return True
    
    else: return False 

# ======== Execution Functions ========
def fetch_df(symbol, timeframe, start_pos, count, fast, slow):
        
        # get rates for the last 60 minutes    
        rates = mt5.copy_rates_from_pos(symbol, timeframe, start_pos, count)

        if rates is None:
            print(mt5.last_error())
            return None

        if len(rates) == 0:
            print("No bars returned")
            return None
        
        # create DataFrame out of the obtained data
        df = pd.DataFrame(rates)    
        df['time'] = pd.to_datetime(df['time'], unit='s') # convert time in seconds into the datetime format
        df.set_index('time', inplace=True) # set time as index
        df = df[['open', 'high', 'low', 'close', 'tick_volume']] # keep only necessary columns

        index = df.index
        open = df['open']
        high = df['high']
        low = df['low']
        close = df['close']
        volume = df['tick_volume']
        fast_ma = df['close'].rolling(fast).mean()
        slow_ma = df['close'].rolling(slow).mean()
        #rsi = ta.momentum.RSIIndicator(df['close'], window=14).rsi()

        delta = df['close'].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        stoch_indicator = ta.momentum.StochasticOscillator( df['high'], df['low'], df['close'], window=5, smooth_window=1)
        raw_k = stoch_indicator.stoch()
        
        # MT5 Slowing = 3 (SMA)
        stoch_k = raw_k.rolling(3).mean()

        # MT5 %D = SMA(3) of slowed %K
        stoch_d = stoch_k.rolling(3).mean()

        #ADX 
        adx = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=14, fillna=False).adx()
        adx_pos = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=14, fillna=False).adx_pos()
        adx_neg = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=14, fillna=False).adx_neg()

        candle_range = df['high'] - df['low'] 
        candle_body = abs(df['close'] - df['open'])
        strength = round( (candle_body / candle_range) * 100,2)
        avg_body = candle_body.rolling(3).mean()


        return fast_ma, slow_ma, rsi, open, high, low, close, volume, index, stoch_k, stoch_d, adx, adx_pos, adx_neg, candle_body, avg_body

def send_buy_order(
        symbol, 
        order_type, 
        volume=LOT_SIZE, 
        deviation=DEVIATION, 
        timeframe=TIMEFRAME_M5, 
        start_pos=0, 
        count=60, 
        fast=None, 
        slow=None,
        hft=False):

    for _ in range(3): 

        tick = mt5.symbol_info_tick(SYMBOL)

        if tick is not None:

            entry_price = tick.ask
            break

        time.sleep(0.5)

    if tick is None:
            
            log.error("Failed to get symbol info after 3 attempts")
            return mt5.last_error 

    if hft is False or None:

        _, _, _, _, _, low_ma, _, _, _, _, _, _, _, _, _, _ = fetch_df(symbol, timeframe, start_pos, count, fast, slow)
        stop_loss = low_ma.iloc[-3]
        take_profit = entry_price + ((entry_price - low_ma.iloc[-3]) * 1.4)

    if hft is True:

        _, _, _, open, high, low, _, _, _, _, _, _, _, _, _, _ = fetch_df(symbol, timeframe, start_pos, count, fast, slow)
        #stop_loss = min(low.iloc[-1], low.iloc[-2], low.iloc[-3])
        stop_loss = low.iloc[-1]
        take_profit = entry_price + ((entry_price - stop_loss) * 1.3)   
        #take_profit = open.iloc[-1] + ((open.iloc[-1] - stop_loss) * 0.5)

    print(f"Buy_Order Entry : {entry_price}")
    print(f"Buy_Order SL    : {stop_loss}")
    print(f"Buy_Order TP    : {take_profit}")

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": volume,
        "type": order_type,
        "price": entry_price,
        "sl": stop_loss,
        "tp": take_profit, 
        "deviation": deviation,
        "magic": 234000,
        "comment": "python script open",
        "type_time": mt5.ORDER_TIME_GTC,  # Good till canceled
        "type_filling": mt5.ORDER_FILLING_FOK, #info.filling_mode #mt5.ORDER_FILLING_RETURN,  # Immediate or Cancel
    }
    return request, entry_price, stop_loss, take_profit, tick

def send_sell_order(
        symbol, 
        order_type, 
        volume=LOT_SIZE, 
        deviation=DEVIATION, 
        timeframe=None, 
        start_pos=0, 
        count=60, 
        fast=None, 
        slow=None,
        hft=False):

    for _ in range(3): 

        tick = mt5.symbol_info_tick(symbol)

        if tick is not None:

            entry_price = tick.bid
            break

        time.sleep(0.5)

    if tick is None:
            
            log.error("Failed to get symbol info after 3 attempts")
            return mt5.last_error
    
    if hft is False or None:

        _, _, _, _, high_ma, _, _, _, _, _, _, _, _, _, _, _ = fetch_df(symbol, timeframe, start_pos, count, fast, slow)
        stop_loss = high_ma.iloc[-3]
        take_profit = entry_price - ((high_ma.iloc[-3] - entry_price) * 1.4)

    if hft is True:

        _, _, _, open, high, low, _, _, _, _, _, _, _, _, _, _ = fetch_df(symbol, timeframe, start_pos, count, fast, slow)
        #stop_loss = max(high.iloc[-1], high.iloc[-2], high.iloc[-3])
        stop_loss = high.iloc[-1]
        take_profit = entry_price - ((stop_loss - entry_price) * 1.3)
        #take_profit = open.iloc[-1] - ((stop_loss -  open.iloc[-1]) * 1.5)


    print(f"Sell_Order Entry : {entry_price}")
    print(f"Sell_Order SL    : {stop_loss}")
    print(f"Sell_Order TP    : {take_profit}")

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": volume,
        "type": order_type,
        "price": entry_price,
        "sl": stop_loss,
        "tp": take_profit, 
        "deviation": deviation,
        "magic": 234000,
        "comment": "python script open",
        "type_time": mt5.ORDER_TIME_GTC,  # Good till canceled
        "type_filling": mt5.ORDER_FILLING_FOK, #info.filling_mode #mt5.ORDER_FILLING_RETURN,  # Immediate or Cancel
    }
    return request, entry_price, stop_loss, take_profit, tick

def execute_buy(symbol, timeframe, hft=False):

    #current_hour = time.localtime().tm_hour

   # if 13 <= current_hour < 16:   log.info("News trading window. No trades.") return None

    send_telegram_message(f"BUY signal detected for {symbol}")
    log.info("Buy signal detected")

    # === send a buy order trading request ===
    request, entry_price, stop_loss, take_profit, tick = send_buy_order(symbol=symbol, order_type=mt5.ORDER_TYPE_BUY, timeframe=timeframe, 
        start_pos=0, 
        count=60, 
        fast=FAST, 
        slow=SLOW,
        hft=hft)
    
    if take_profit > entry_price:

        result = mt5.order_send(request)
        send_telegram_message(f"BUY {SYMBOL} | Price: {entry_price} | TP: {take_profit:.2f} | SL: {stop_loss:.2f}")
        
        if result.retcode != mt5.TRADE_RETCODE_DONE or result is None:

            for _ in range(3): 

                time.sleep(0.5)
                result = mt5.order_send(request)

                if result.retcode == mt5.TRADE_RETCODE_DONE:

                    break

        if result.retcode == mt5.TRADE_RETCODE_DONE:

            log.info(f"Buy order sent @ {entry_price} | TP: {take_profit:.2f} | SL: {stop_loss:.2f}")
            
            current_position = {
            "type": "buy",
            "price": entry_price,
            "take_profit": take_profit,
            "stop_loss": stop_loss,
            "time": pd.to_datetime(tick.time_msc, unit='ms', utc=True).isoformat()
            }
            save_state(current_position)

            if hft is False:

                send_telegram_message(f"LFT buy order sent @ {entry_price} | TP: {take_profit:.2f} | SL: {stop_loss:.2f}")
                return current_position
            
            if hft is True:

                send_telegram_message(f"HFT buy order sent @ {entry_price} | TP: {take_profit:.2f} | SL: {stop_loss:.2f} ")
                return current_position
        
        else:

            log.error(f"Failed to send order after 3 attempts: | retcode={result.retcode} | comment={result.comment}")
            return None
    else: return None

def execute_sell(symbol, timeframe, hft=False):

    #current_hour = time.localtime().tm_hour

   # if 13 <= current_hour < 16: log.info("News trading window. No trades.") return None

    send_telegram_message(f"SELL signal detected for {symbol}")
    log.info("Sell signal detected")
    
    # === send a sell order trading request ===
    request, entry_price, stop_loss, take_profit, tick = send_sell_order(symbol, mt5.ORDER_TYPE_SELL, timeframe=timeframe, 
        start_pos=0, 
        count=60, 
        fast=FAST, 
        slow=SLOW,
        hft=hft)
    
    if take_profit < entry_price:

        result = mt5.order_send(request)
        send_telegram_message(f"SELL {symbol} | Price: {entry_price} | TP: {take_profit:.2f} | SL: {stop_loss:.2f}")

        if result.retcode != mt5.TRADE_RETCODE_DONE or result is None:

            for _ in range(3): 

                time.sleep(0.5)
                result = mt5.order_send(request)
                if result.retcode == mt5.TRADE_RETCODE_DONE:

                    break
        
        if result.retcode == mt5.TRADE_RETCODE_DONE:

            log.info(f"SELL order sent @ {entry_price} | TP: {take_profit:.2f} | SL: {stop_loss:.2f}")
            
            current_position = {
            "type": "sell",
            "price": entry_price,
            "take_profit": take_profit,
            "stop_loss": stop_loss,
            "time": pd.to_datetime(tick.time_msc, unit='ms', utc=True).isoformat()
            }
            save_state(current_position)
            
            if hft is False:

                send_telegram_message(f"LFT Sell order sent @ {entry_price} | TP: {take_profit:.2f} | SL: {stop_loss:.2f}")
                return current_position
            
            if hft is True:

                send_telegram_message(f"HFT Sell order sent @ {entry_price} | TP: {take_profit:.2f} | SL: {stop_loss:.2f} ")
                return current_position

        else:

            log.error(f"Failed to send order after 3 attempts: | retcode={result.retcode} | comment={result.comment}") 
            return None

    else: return None
    
def mt5_shutdown():

    mt5.shutdown()
    log.info("MetaTrader 5 shutdown")

def main():

    if not mt5_initialize():
        return
    else:
        loop()
    mt5_shutdown()

def loop():
   
    last_candle_time = None

    hft_current_position = load_state(path=HFT_STATE_FILE)
    #current_position = load_state(path=STATE_FILE)
    while True:

        try:

            # === determine up/down trend by 1hr timeframe ===
            h1_fast_ma, h1_slow_ma, h1_rsi, _, _, _, _, _, _, _, _, _, _, _, _, _ = fetch_df(SYMBOL, TIMEFRAME_H1, START_POS, COUNT, FAST, SLOW)
            h1_uptrend = h1_fast_ma.iloc[-2] > h1_slow_ma.iloc[-2] and h1_rsi.iloc[-2] > 51
            h1_downtrend = h1_fast_ma.iloc[-2] < h1_slow_ma.iloc[-2] and h1_rsi.iloc[-2] < 49 

            # === determine up/down trend by 15m timeframe ===
            #m15_fast_ma, m15_slow_ma, m15_rsi, _, _, _, _, _ = fetch_df(SYMBOL, TIMEFRAME_M15, START_POS, COUNT, FAST, SLOW)
            #m15_uptrend = m15_fast_ma.iloc[-2] > m15_slow_ma.iloc[-2] and m15_rsi.iloc[-2] > 51
            #m15_downtrend = m15_fast_ma.iloc[-2] < m15_slow_ma.iloc[-2] and m15_rsi.iloc[-2] < 49

            # === calculate moving averages and RSI for 5mins timeframe ===
            _, _, _, _, _, _, _, _, m5_index, _, _, _, _, _, _, _ = fetch_df(SYMBOL, TIMEFRAME_M5, START_POS, COUNT, FAST, SLOW)
            current_candle_time = m5_index[-1]
            
            #m5_uptrend = m5_fast_ma.iloc[-2] > m5_slow_ma.iloc[-2] and m5_rsi.iloc[-2] > 51
            #m5_downtrend = m5_fast_ma.iloc[-2] < m5_slow_ma.iloc[-2] and m5_rsi.iloc[-2] < 49
            log.info(f"Fast MA: {h1_fast_ma.iloc[-1]:.2f}, Slow MA: {h1_slow_ma.iloc[-1]:.2f}, RSI: {h1_rsi.iloc[-1]:.2f}")

            # === determine 15min buy/sell signals ===
            #m15_buy = h1_uptrend and m15_fast_ma.iloc[-3] < m15_slow_ma.iloc[-3] and m15_fast_ma.iloc[-2] > m15_slow_ma.iloc[-2] and m15_rsi.iloc[-2] >= 55 
            #m15_sell = h1_downtrend and m15_fast_ma.iloc[-3] > m15_slow_ma.iloc[-3] and m15_fast_ma.iloc[-2] < m15_slow_ma.iloc[-2] and m15_rsi.iloc[-2] <= 45 

            # check if there is an htf open position
            if hft_current_position is None:

                print('checking for htf setup...')

                if current_candle_time != last_candle_time:
                
                    buy = m5_long()
                    sell = m5_short()

                    print('Here we go')
                    if buy: # and h1_rsi.iloc[-4] > h1_rsi.loc[-2] :  # hft buy strategy

                            hft_current_position = execute_buy(SYMBOL, TIMEFRAME_M5, hft=True)
                            last_candle_time = current_candle_time
                            save_state(hft_current_position, path=HFT_STATE_FILE)

                    elif sell: # and h1_rsi.iloc[-4] > h1_rsi.loc[-2]: # hft sell str

                            hft_current_position = execute_sell(SYMBOL, TIMEFRAME_M5, hft=True)
                            last_candle_time = current_candle_time
                            save_state(hft_current_position, path=HFT_STATE_FILE)
                        
            else:

                log.info("Checking for HFT take-profit or stop-loss")
                tick = mt5.symbol_info_tick(SYMBOL)
                
                if tick is None: 

                    log.error("Failed to get symbol info")
                    continue

                else:

                    if hft_current_position["type"] == "buy":

                        if tick.bid >= hft_current_position["take_profit"]:

                            log.info(f"Take profit hit for HFT buy position at {tick.bid}")
                            send_telegram_message(f"Take profit hit for HFT buy position at {tick.ask}")
                            last_candle_time = current_candle_time
                            clear_state(path=HFT_STATE_FILE)
                            hft_current_position = None
                            
                                                  

                        elif tick.bid <= hft_current_position["stop_loss"]:

                            log.info(f"Stop loss hit for HFT buy position at {tick.bid}")
                            send_telegram_message(f"Stop loss hit for HFT buy position at {tick.bid}")
                            last_candle_time = current_candle_time
                            clear_state(path=HFT_STATE_FILE)
                            hft_current_position = None
                            
                    elif hft_current_position["type"] == "sell":

                        if tick.ask <= hft_current_position["take_profit"]:

                            log.info(f"Take profit hit for HFT sell position at {tick.ask}")
                            send_telegram_message(f"Take profit hit for HFT sell position at {tick.bid}")
                            last_candle_time = current_candle_time
                            skip_next_candle = True
                            clear_state(path=HFT_STATE_FILE)
                            hft_current_position = None                           

                        elif tick.ask >= hft_current_position["stop_loss"]:

                            log.info(f"Stop loss hit for HFT sell position at {tick.ask}")
                            send_telegram_message(f"Stop loss hit for HFT sell position at {tick.ask}")
                            last_candle_time = current_candle_time
                            clear_state(path=HFT_STATE_FILE)
                            hft_current_position = None
                            hft_open_position = False
            
            #if hft_current_position is not None:


           
        except Exception:
            
            traceback.print_exc()
        
        time.sleep(POLL_SEC)

if __name__ == "__main__":
    
    hft_current_position = load_state(path=HFT_STATE_FILE)
    #current_position = load_state(path=STATE_FILE)
    #clear_state(path=HFT_STATE_FILE)


    log.info(f"Start {SYMBOL} | TIMEFRAME={TIMEFRAME_M5}MINS | FAST={FAST} SLOW={SLOW}")
    send_telegram_message(f" {SYMBOL} BOT STARTED @ {time.strftime('%Y-%m-%d %H:%M:%S')} ")

    try:

        main()

    except KeyboardInterrupt:

        log.info("Script interrupted by user")

    except Exception as e:

        log.error(f"Error in main: {e}")
