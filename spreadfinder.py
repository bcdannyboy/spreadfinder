import argparse
import datetime
import yfinance as yf
import numpy as np
import pandas as pd
from scipy.stats import norm
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from itertools import islice
import os
import sys

def get_stock_data(symbol, mindte, maxdte):
    print(f"Fetching stock data for {symbol} with min DTE {mindte} and max DTE {maxdte}")
    ticker = yf.Ticker(symbol)
    expirations = ticker.options

    valid_exp = []
    for exp in expirations:
        exp_date = datetime.datetime.strptime(exp, '%Y-%m-%d')
        dte = (exp_date - datetime.datetime.now()).days
        if mindte <= dte <= maxdte:
            valid_exp.append(exp)

    puts = []
    calls = []
    
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(fetch_option_chain, ticker, exp): exp for exp in valid_exp}
        for future in as_completed(futures):
            exp, opt_puts_filtered, opt_calls_filtered = future.result()
            puts.append((exp, opt_puts_filtered))
            calls.append((exp, opt_calls_filtered))
    
    print(f"Found {len(puts)} put chains and {len(calls)} call chains for {symbol} with min DTE {mindte} and max DTE {maxdte}")
    return puts, calls

def fetch_option_chain(ticker, exp):
    opt = ticker.option_chain(exp)
    opt_puts = opt.puts
    opt_calls = opt.calls
    opt_puts_filtered = opt_puts[(opt_puts['bid'] > 0) & (opt_puts['ask'] > 0)]
    opt_calls_filtered = opt_calls[(opt_calls['bid'] > 0) & (opt_calls['ask'] > 0)]
    return exp, opt_puts_filtered, opt_calls_filtered

def process_bull_put_spread(exp, put_chain, underlying_price, min_ror, max_strike_dist):
    def process_single_spread(i, j):
        short_put = put_chain.iloc[j]
        long_put = put_chain.iloc[i]

        if short_put.strike > underlying_price * (1 + max_strike_dist) or short_put.strike < underlying_price * (1 - max_strike_dist):
            return []

        if short_put.strike > long_put.strike:
            credit = short_put.bid - long_put.ask
            max_loss = (short_put.strike - long_put.strike) - credit
            if max_loss <= 0:
                return []

            return_on_risk = credit / max_loss

            if return_on_risk < min_ror:
                return []

            time_to_expiration = (datetime.datetime.strptime(exp, '%Y-%m-%d') - datetime.datetime.now()).days / 365.0
            iv = short_put.impliedVolatility
            if iv is None or iv <= 0:
                return []
            
            d1 = (np.log(underlying_price / short_put.strike) + (0.5 * iv**2) * time_to_expiration) / (iv * np.sqrt(time_to_expiration))
            probability_of_success = norm.cdf(d1)
            
            return [(exp, short_put.strike, long_put.strike, credit, max_loss, return_on_risk, probability_of_success)]
        return []

    results = []
    put_chain = put_chain.sort_values(by='strike')
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = [executor.submit(process_single_spread, i, j) for i in range(len(put_chain)) for j in range(i + 1, len(put_chain))]
        total_spreads = len(futures)
        completed_spreads = 0

        for future in as_completed(futures):
            results.extend(future.result())
            completed_spreads += 1
            if completed_spreads % 100 == 0 or completed_spreads == total_spreads:
                print(f"\rProcessed {completed_spreads}/{total_spreads} bull put spreads", end='', flush=True)

    print()
    return results

def batch_futures(iterator, batch_size):
    """Helper function to batch futures."""
    while True:
        batch = list(islice(iterator, batch_size))
        if not batch:
            break
        yield batch

def find_bull_put_spreads(puts, underlying_price, min_ror, max_strike_dist, batch_size):
    print(f"Finding bull put spreads for underlying price {underlying_price} using batch size {batch_size}")
    bull_put_spreads = []

    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures_iterator = (
            executor.submit(process_bull_put_spread, exp, put_chain, underlying_price, min_ror, max_strike_dist)
            for exp, put_chain in puts
        )
        batches_processed = 0
        for batch in batch_futures(futures_iterator, batch_size):
            for future in as_completed(batch):
                bull_put_spreads.extend(future.result())
            batches_processed += 1
            print(f"\rProcessed batch {batches_processed}", end='', flush=True)

    print()
    return bull_put_spreads

def process_iron_condor(exp, put_chain, call_chain, underlying_price, min_ror, max_strike_dist):
    def process_single_condor(i, j, k, l):
        short_put = put_chain.iloc[j]
        long_put = put_chain.iloc[i]
        short_call = call_chain.iloc[k]
        long_call = call_chain.iloc[l]

        if (short_put.strike > underlying_price * (1 + max_strike_dist) or short_put.strike < underlying_price * (1 - max_strike_dist) or
            short_call.strike > underlying_price * (1 + max_strike_dist) or short_call.strike < underlying_price * (1 - max_strike_dist)):
            return []

        if short_put.strike > long_put.strike and short_call.strike < long_call.strike:
            credit = short_put.bid - long_put.ask + short_call.bid - long_call.ask
            max_loss_put = (short_put.strike - long_put.strike) - credit
            max_loss_call = (long_call.strike - short_call.strike) - credit
            max_loss = max(max_loss_put, max_loss_call)
            if max_loss <= 0:
                return []
            
            return_on_risk = credit / max_loss
            
            if return_on_risk < min_ror:
                return []

            time_to_expiration = (datetime.datetime.strptime(exp, '%Y-%m-%d') - datetime.datetime.now()).days / 365.0
            iv_put = short_put.impliedVolatility
            iv_call = short_call.impliedVolatility
            if iv_put is None or iv_put <= 0 or iv_call is None or iv_call <= 0:
                return []
            
            d1_put = (np.log(underlying_price / short_put.strike) + (0.5 * iv_put**2) * time_to_expiration) / (iv_put * np.sqrt(time_to_expiration))
            d1_call = (np.log(underlying_price / short_call.strike) + (0.5 * iv_call**2) * time_to_expiration) / (iv_call * np.sqrt(time_to_expiration))
            probability_of_success_put = norm.cdf(d1_put)
            probability_of_success_call = 1 - norm.cdf(d1_call)
            probability_of_success = (probability_of_success_put + probability_of_success_call) / 2
            
            return [(exp, short_put.strike, long_put.strike, short_call.strike, long_call.strike, credit, max_loss, return_on_risk, probability_of_success)]
        return []

    results = []
    put_chain = put_chain.sort_values(by='strike')
    call_chain = call_chain.sort_values(by='strike')
    
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = [executor.submit(process_single_condor, i, j, k, l)
                   for i in range(len(put_chain)) for j in range(i + 1, len(put_chain))
                   for k in range(len(call_chain)) for l in range(k + 1, len(call_chain))]
        total_condors = len(futures)
        completed_condors = 0

        for future in as_completed(futures):
            results.extend(future.result())
            completed_condors += 1
            if completed_condors % 100 == 0 or completed_condors == total_condors:
                print(f"\rProcessed {completed_condors}/{total_condors} iron condors", end='', flush=True)

    print()
    return results

def find_iron_condors(puts, calls, underlying_price, min_ror, max_strike_dist, batch_size):
    print(f"Finding iron condors for underlying price {underlying_price} using batch size {batch_size}")
    iron_condors = []

    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures_iterator = (
            executor.submit(process_iron_condor, exp, put_chain, call_chain, underlying_price, min_ror, max_strike_dist)
            for (exp, put_chain), (exp_c, call_chain) in zip(puts, calls)
            if exp == exp_c
        )
        batches_processed = 0
        for batch in batch_futures(futures_iterator, batch_size):
            for future in as_completed(batch):
                iron_condors.extend(future.result())
            batches_processed += 1
            print(f"\rProcessed batch {batches_processed}", end='', flush=True)

    print()
    return iron_condors

def find_best_spreads(puts, calls, underlying_price, top_n, min_ror, max_strike_dist, batch_size, include_iron_condors):
    bull_put_spreads = find_bull_put_spreads(puts, underlying_price, min_ror, max_strike_dist, batch_size)
    combined_spreads = []

    for spread in bull_put_spreads:
        combined_spreads.append((*spread, 'bull_put'))

    if include_iron_condors:
        iron_condors = find_iron_condors(puts, calls, underlying_price, min_ror, max_strike_dist, batch_size)
        for spread in iron_condors:
            combined_spreads.append((*spread, 'iron_condor'))

    combined_spreads.sort(key=lambda x: (x[-2], x[-3]), reverse=True)

    return combined_spreads[:top_n]

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description="Find and rank option spreads")
    
    argparser.add_argument('-symbol', '-s', type=str, required=True, help='Stock symbol to fetch data for')
    argparser.add_argument('-mindte', '-m', type=int, required=True, help='Minimum days to expiration')
    argparser.add_argument('-maxdte', '-l', type=int, required=True, help='Maximum days to expiration')
    argparser.add_argument('-top_n', '-n', type=int, default=10, help='Number of top spreads to display')
    argparser.add_argument('-min_ror', '-r', type=float, default=0.15, help='Minimum return on risk')
    argparser.add_argument('-max_strike_dist', '-d', type=float, default=0.2, help='Maximum strike distance percentage')
    argparser.add_argument('-output', '-o', type=str, default='best_spreads.csv', help='Output CSV file path')
    argparser.add_argument('-batch_size', '-b', type=int, default=10, help='Batch size for processing tasks')
    argparser.add_argument('--include_iron_condors', action='store_true', help='Include iron condors in the search')
    
    args = argparser.parse_args()
    
    ticker = yf.Ticker(args.symbol)
    puts, calls = get_stock_data(args.symbol, args.mindte, args.maxdte)
    
    underlying_price = ticker.history(period="1d")['Close'][0]
    print(f"Underlying price: {underlying_price}")

    best_spreads = find_best_spreads(puts, calls, underlying_price, args.top_n, args.min_ror, args.max_strike_dist, args.batch_size, args.include_iron_condors)
    
    print("Best Spreads:")
    spread_data = []
    for spread in best_spreads:
        exp = spread[0]
        if spread[-1] == 'bull_put':
            short_strike = spread[1]
            long_strike = spread[2]
            credit = spread[3]
            max_loss = spread[4]
            return_on_risk = spread[5]
            probability_of_success = spread[6]
            spread_type = spread[7]
            spread_data.append([spread_type, exp, short_strike, long_strike, None, None, credit, max_loss, return_on_risk, probability_of_success])
        elif spread[-1] == 'iron_condor':
            short_put_strike = spread[1]
            long_put_strike = spread[2]
            short_call_strike = spread[3]
            long_call_strike = spread[4]
            credit = spread[5]
            max_loss = spread[6]
            return_on_risk = spread[7]
            probability_of_success = spread[8]
            spread_type = spread[9]
            spread_data.append([spread_type, exp, short_put_strike, long_put_strike, short_call_strike, long_call_strike, credit, max_loss, return_on_risk, probability_of_success])

    df = pd.DataFrame(spread_data, columns=['Type', 'Expiration', 'Short Put Strike', 'Long Put Strike', 'Short Call Strike', 'Long Call Strike', 'Credit', 'Max Loss', 'Return on Risk', 'Probability of Success'])
    df.to_csv(args.output, index=False)
    print(f"Results saved to {args.output}")
