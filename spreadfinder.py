import argparse
import datetime
import yfinance as yf
import numpy as np
import pandas as pd
from scipy.stats import norm
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Value

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
    
    for exp in valid_exp:
        opt = ticker.option_chain(exp)
        opt_puts_filtered = opt.puts[(opt.puts['bid'] > 0) & (opt.puts['ask'] > 0)]
        opt_calls_filtered = opt.calls[(opt.calls['bid'] > 0) & (opt.calls['ask'] > 0)]
        puts.append((exp, opt_puts_filtered))
        calls.append((exp, opt_calls_filtered))
    
    print(f"Found {len(puts)} put chains and {len(calls)} call chains for {symbol} with min DTE {mindte} and max DTE {maxdte}")
    return puts, calls

def process_bull_put_spread(exp, put_chain, underlying_price, min_ror, max_strike_dist, progress, total):
    results = []
    put_chain = put_chain.sort_values(by='strike')
    for i in range(len(put_chain)):
        for j in range(i + 1, len(put_chain)):
            short_put = put_chain.iloc[j]
            long_put = put_chain.iloc[i]

            if short_put.strike > underlying_price * (1 + max_strike_dist) or short_put.strike < underlying_price * (1 - max_strike_dist):
                continue

            if short_put.strike > long_put.strike:
                credit = short_put.bid - long_put.ask
                max_loss = (short_put.strike - long_put.strike) - credit
                if max_loss <= 0:
                    continue
                
                return_on_risk = credit / max_loss
                
                if return_on_risk < min_ror:
                    continue

                time_to_expiration = (datetime.datetime.strptime(exp, '%Y-%m-%d') - datetime.datetime.now()).days / 365.0
                iv = short_put.impliedVolatility
                if iv is None or iv <= 0:
                    continue
                
                d1 = (np.log(underlying_price / short_put.strike) + (0.5 * iv**2) * time_to_expiration) / (iv * np.sqrt(time_to_expiration))
                probability_of_success = norm.cdf(d1)
                
                results.append((exp, short_put.strike, long_put.strike, credit, max_loss, return_on_risk, probability_of_success))
            
            with progress.get_lock():
                progress.value += 1

    return results

def find_bull_put_spreads(puts, underlying_price, min_ror, max_strike_dist, max_workers):
    print(f"Finding bull put spreads for underlying price {underlying_price}")
    bull_put_spreads = []

    total_iterations = sum(len(put_chain) * (len(put_chain) - 1) // 2 for _, put_chain in puts)
    progress = Value('i', 0)

    with tqdm(total=total_iterations, desc="Processing Bull Put Spreads") as pbar:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(process_bull_put_spread, exp, put_chain, underlying_price, min_ror, max_strike_dist, progress, total_iterations)
                for exp, put_chain in puts
            ]
            while not all(f.done() for f in futures):
                with progress.get_lock():
                    pbar.update(progress.value - pbar.n)
                futures = [f for f in futures if not f.done()]

            for future in as_completed(futures):
                bull_put_spreads.extend(future.result())
                with progress.get_lock():
                    pbar.update(progress.value - pbar.n)

    return bull_put_spreads

def process_iron_condor(exp, put_chain, call_chain, underlying_price, min_ror, max_strike_dist, progress, total):
    results = []
    put_chain = put_chain.sort_values(by='strike')
    call_chain = call_chain.sort_values(by='strike')

    for i in range(len(put_chain)):
        for j in range(i + 1, len(put_chain)):
            for k in range(len(call_chain)):
                for l in range(k + 1, len(call_chain)):
                    short_put = put_chain.iloc[j]
                    long_put = put_chain.iloc[i]
                    short_call = call_chain.iloc[k]
                    long_call = call_chain.iloc[l]

                    if (short_put.strike > underlying_price * (1 + max_strike_dist) or short_put.strike < underlying_price * (1 - max_strike_dist) or
                        short_call.strike > underlying_price * (1 + max_strike_dist) or short_call.strike < underlying_price * (1 - max_strike_dist)):
                        continue

                    if short_put.strike > long_put.strike and short_call.strike < long_call.strike:
                        credit = short_put.bid - long_put.ask + short_call.bid - long_call.ask
                        max_loss_put = (short_put.strike - long_put.strike) - credit
                        max_loss_call = (long_call.strike - short_call.strike) - credit
                        max_loss = max(max_loss_put, max_loss_call)
                        if max_loss <= 0:
                            continue
                        
                        return_on_risk = credit / max_loss
                        
                        if return_on_risk < min_ror:
                            continue

                        time_to_expiration = (datetime.datetime.strptime(exp, '%Y-%m-%d') - datetime.datetime.now()).days / 365.0
                        iv_put = short_put.impliedVolatility
                        iv_call = short_call.impliedVolatility
                        if iv_put is None or iv_put <= 0 or iv_call is None or iv_call <= 0:
                            continue
                        
                        d1_put = (np.log(underlying_price / short_put.strike) + (0.5 * iv_put**2) * time_to_expiration) / (iv_put * np.sqrt(time_to_expiration))
                        d1_call = (np.log(underlying_price / short_call.strike) + (0.5 * iv_call**2) * time_to_expiration) / (iv_call * np.sqrt(time_to_expiration))
                        probability_of_success_put = norm.cdf(d1_put)
                        probability_of_success_call = 1 - norm.cdf(d1_call)
                        probability_of_success = (probability_of_success_put + probability_of_success_call) / 2
                        
                        results.append((exp, short_put.strike, long_put.strike, short_call.strike, long_call.strike, credit, max_loss, return_on_risk, probability_of_success))
            
            with progress.get_lock():
                progress.value += 1

    return results

def find_iron_condors(puts, calls, underlying_price, min_ror, max_strike_dist, max_workers):
    print(f"Finding iron condors for underlying price {underlying_price}")
    iron_condors = []

    total_iterations = sum(
        len(put_chain) * (len(put_chain) - 1) // 2 * len(call_chain) * (len(call_chain) - 1) // 2
        for (exp, put_chain), (exp_c, call_chain) in zip(puts, calls)
        if exp == exp_c
    )
    progress = Value('i', 0)

    with tqdm(total=total_iterations, desc="Processing Iron Condors") as pbar:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(process_iron_condor, exp, put_chain, call_chain, underlying_price, min_ror, max_strike_dist, progress, total_iterations)
                for (exp, put_chain), (exp_c, call_chain) in zip(puts, calls)
                if exp == exp_c
            ]
            while not all(f.done() for f in futures):
                with progress.get_lock():
                    pbar.update(progress.value - pbar.n)
                futures = [f for f in futures if not f.done()]

            for future in as_completed(futures):
                iron_condors.extend(future.result())
                with progress.get_lock():
                    pbar.update(progress.value - pbar.n)

    return iron_condors

def find_best_spreads(puts, calls, underlying_price, top_n, min_ror, max_strike_dist, max_workers):
    bull_put_spreads = find_bull_put_spreads(puts, underlying_price, min_ror, max_strike_dist, max_workers)
    iron_condors = find_iron_condors(puts, calls, underlying_price, min_ror, max_strike_dist, max_workers)
    
    combined_spreads = []
    
    for spread in bull_put_spreads:
        combined_spreads.append((*spread, 'bull_put'))
    
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
    argparser.add_argument('-max_workers', '-w', type=int, default=4, help='Maximum number of worker processes to use')
    
    args = argparser.parse_args()
    
    ticker = yf.Ticker(args.symbol)
    puts, calls = get_stock_data(args.symbol, args.mindte, args.maxdte)
    
    underlying_price = ticker.history(period="1d")['Close'][0]
    print(f"Underlying price: {underlying_price}")

    best_spreads = find_best_spreads(puts, calls, underlying_price, args.top_n, args.min_ror, args.max_strike_dist, args.max_workers)
    
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
