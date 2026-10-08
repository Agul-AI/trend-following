#!/usr/bin/env python3
"""Independent v4 arithmetic audit. Run ONLY in the custodian's audit sandbox.

Reads the new released/test packet and sealed new output files, not a project
module, old report, network source, engine, ledger, rate helper, or policy import.
The policy formula is separately transcribed from the fixed proposal. Tax checks
are event/account/return identities; this is not a full tax-lot certification.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import traceback
import numpy as np
import pandas as pd

CHECKS = []
READS = []
STUDY = None

def check(name, passed, details=None):
    CHECKS.append({'check': name, 'passed': bool(passed), 'details': details})
    return bool(passed)

def read_csv(path):
    path = Path(path); READS.append(str(path.relative_to(STUDY)))
    return pd.read_csv(path)

def read_json(path):
    path = Path(path); READS.append(str(path.relative_to(STUDY)))
    return json.loads(path.read_text())

def compare(name, observed, expected, atol=1e-8, rtol=2e-10):
    a = np.asarray(observed, dtype=float); b = np.asarray(expected, dtype=float)
    if a.shape != b.shape:
        return check(name, False, {'observed_shape': list(a.shape), 'expected_shape': list(b.shape)})
    ok = np.isclose(a, b, atol=atol, rtol=rtol, equal_nan=True)
    delta = np.abs(a-b)
    finite = delta[np.isfinite(delta)]
    bad = np.flatnonzero(~ok)
    return check(name, bool(ok.all()), {'n': int(a.size), 'max_abs_error': float(finite.max()) if len(finite) else 0., 'mismatches': int(len(bad)), 'first_bad_positions': bad[:10].tolist(), 'atol':atol,'rtol':rtol})

def date_series(v):
    return pd.to_datetime(v, utc=True)

def col(df, choices, required=True):
    for c in choices:
        if c in df.columns:
            return c
    if required:
        raise KeyError(f'Missing column from {choices}; actual {list(df.columns)}')
    return None

def stable_json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), default=str).encode()

def integrate_known_rate(frame, sessions, daycount, spread=0.):
    """Simple piecewise ACT accrual, left-continuous rate knowledge; no helper."""
    times=pd.to_datetime(frame['available_at'],utc=True).astype('int64').to_numpy()
    order=np.argsort(times,kind='stable'); times=times[order]
    rates=frame['annual_rate'].to_numpy(dtype=float)[order]+spread
    observations=pd.to_datetime(frame['observation_date'],utc=True).astype('int64').to_numpy()[order]
    results=np.zeros(len(sessions)); worst_age=0.
    for i in range(1,len(sessions)):
        left=sessions[i-1].value;right=sessions[i].value
        j=int(np.searchsorted(times,left,side='right')-1)
        if j<0: raise ValueError('No known rate at '+str(sessions[i-1]))
        breakpoints=times[(times>left)&(times<right)]
        edges=np.unique(np.r_[left,breakpoints,right]);total=0.
        for l,r in zip(edges[:-1],edges[1:]):
            j=int(np.searchsorted(times,l,side='right')-1)
            total+=rates[j]*float(r-l)/(1e9*86400.*daycount)
            worst_age=max(worst_age,float(l-observations[j])/(1e9*86400.))
        results[i]=total
    check('rate max observation age '+str(frame['series_id'].iloc[0]),worst_age<=10.,{'worst_calendar_days':worst_age})
    return results


def raw_returns(price):
    close = price['close'].to_numpy(dtype=float)
    split = price['split_coefficient'].to_numpy(dtype=float)
    div = price['dividend_amount'].to_numpy(dtype=float)
    ret = np.zeros(len(price))
    ret[1:] = split[1:]*(close[1:]+div[1:])/close[:-1]-1.
    return ret

def selected_targets(ret, params):
    """No policy import: fixed positive two-horizon daily-reset log-growth gate."""
    out = np.zeros(len(ret)); long = int(params['long_window']); short = int(params['short_window'])
    need = max(long, short, int(params['vol_window']))+1
    for i in range(need-1, len(ret)):
        leveraged = 3.*ret[i-long+1:i+1]
        if not np.isfinite(leveraged).all() or np.any(leveraged <= -1.):
            continue
        log = np.log1p(leveraged)
        if log.sum() > 0. and log[-short:].sum() > 0.:
            if params['volatility_ceiling'] is not None:
                raise ValueError('Audit permits only the actual sealed no-volatility-ceiling selected method')
            out[i] = float(params['cap'])
    return out

def independent_ledger(market, targets, kind, layer, costs):
    """Independent pre-cost nominal target sizing; no external ledger."""
    fee=costs['fee_bps']/10000. if layer!='gross' else 0.
    slip=costs['slippage_bps']/10000. if layer!='gross' else 0.
    c=fee+slip;cash=float(costs['initial_cash']);qty=0.;last_target=0.;prev_equity=cash
    rows=[];orders=0;total_fee=total_slip=total_turnover=0.
    for i,row in market.iterrows():
        interest=cash*float(row['cash_return']) if i else 0.;cash+=interest
        if kind=='QQQ' and i: qty*=float(row['split_coefficient'])
        price=float(row['close']) if kind=='QQQ' else float(row['nav_'+layer])
        dividend=qty*float(row['dividend_amount']) if kind=='QQQ' and i else 0.
        cash+=dividend
        w=float(targets[i-1]) if i else 0.
        risk=qty*price;pre=cash+risk;f=sl=nt=0.;day_orders=0
        if abs(w-last_target)>1e-12 or (kind=='QQQ' and i):
            numerator=w*pre-risk
            if numerator>1e-12:
                nt=numerator;nt=min(nt,max(0.,cash)/(1.+c))
                qty+=nt/price;f=nt*fee;sl=nt*slip;cash-=nt+f+sl
                day_orders=int(nt>1e-12)
            elif numerator < -1e-12:
                nt=-numerator;nt=min(nt,risk)
                qty-=nt/price;f=nt*fee;sl=nt*slip;cash+=nt-f-sl
                day_orders=int(nt>1e-12)
            last_target=w
        equity=cash+qty*price;orders+=day_orders;total_fee+=f;total_slip+=sl;total_turnover+=nt
        rows.append({'cash':cash,'equity':equity,'quantity':qty,'price':price,'return':equity/prev_equity-1.,'target':w,'fee':f,'slippage':sl,'orders':day_orders,'turnover':nt,'cash_interest':interest,'distribution':dividend,'effective_exposure':qty*price/equity*(3. if kind=='synthetic3x' else 1.),'cash_weight':cash/equity})
        prev_equity=equity
    mark=cash+qty*price;nt=qty*price;tf=nt*fee;ts=nt*slip;liquid=mark-tf-ts
    terminal={'marked_equity':mark,'liquidated_equity':liquid,'fees':tf,'slippage':ts,'taxes':0.,'orders':int(qty>1e-12),'turnover':nt}
    frame=pd.DataFrame(rows);returns=frame['return'].to_numpy().copy()
    if len(frame): returns[-1]=liquid/(float(costs['initial_cash']) if len(frame)==1 else frame['equity'].iloc[-2])-1.
    aggregate={'filled_orders':orders+terminal['orders'],'fees':total_fee+tf,'slippage':total_slip+ts,'taxes':0.,'turnover':total_turnover+nt}
    return frame,returns,terminal,aggregate


def metrics(returns, cash_returns, ledger, terminal, aggregate, targets, sessions, initial):
    r=np.asarray(returns,dtype=float); rf=np.asarray(cash_returns,dtype=float); excess=r-rf
    years=len(r)/252.
    equity=np.cumprod(1.+r)*initial
    drawdown=equity/np.maximum.accumulate(np.r_[initial,equity])[1:]-1.
    std=np.std(r,ddof=1); exstd=np.std(excess,ddof=1)
    return {'sessions':len(r),'CAGR':(terminal['liquidated_equity']/initial)**(1./years)-1.,'total_return_pct':(terminal['liquidated_equity']/initial-1.)*100.,'terminal_equity':terminal['liquidated_equity'],'max_drawdown':float(drawdown.min()),'annualized_volatility':float(std*np.sqrt(252.)),'cash_excess_Sharpe':float(excess.mean()/exstd*np.sqrt(252.)) if exstd>1e-12 else np.nan,'zero_RF_Sharpe':float(r.mean()/std*np.sqrt(252.)) if std>1e-12 else np.nan,'target_changes':int(np.sum(np.abs(np.diff(np.r_[0.,targets]))>1e-12)),'filled_orders':aggregate['filled_orders'],'fees':aggregate['fees'],'slippage':aggregate['slippage'],'taxes':aggregate['taxes'],'mean_effective_exposure':float(ledger['effective_exposure'].mean()),'max_effective_exposure':float(ledger['effective_exposure'].max()),'mean_cash_weight':float(ledger['cash_weight'].mean())}

def tax_identities(base, account, events, holdings, daily, terminal, reported_metrics, costs):
    """Accounting conservation and tax schedule identities, not a tax-lot engine."""
    qtycol=col(account,['quantity','qty','shares'],False)
    cashcol=col(account,['cash']); equitycol=col(account,['equity','marked_equity'])
    pricecol=col(account,['price','asset_price','close'],False)
    if qtycol and pricecol:
        compare(base+'/cash+quantity*price=equity',account[cashcol]+account[qtycol]*account[pricecol],account[equitycol])
    elif len(holdings):
        q=col(holdings,['quantity','qty','shares']); p=col(holdings,['price','asset_price','close']); session=col(holdings,['timestamp','session','date']); acsession=col(account,['timestamp','session','date'])
        mv=(holdings[q]*holdings[p]).groupby(pd.to_datetime(holdings[session],utc=True)).sum()
        mapped=pd.to_datetime(account[acsession],utc=True).map(mv).fillna(0.)
        compare(base+'/cash+holdings=equity',account[cashcol]+mapped,account[equitycol])
    compare(base+'/terminal mark',account[equitycol].iloc[-1],terminal['marked_equity'])
    compare(base+'/terminal liquidation',terminal['marked_equity']-terminal['fees']-terminal['slippage']-terminal['taxes'],terminal['liquidated_equity'])
    initial=float(costs['initial_cash']); observed=np.asarray(daily,dtype=float)
    expected=account[equitycol].to_numpy()/np.r_[initial,account[equitycol].to_numpy()[:-1]]-1.
    expected[-1]=terminal['liquidated_equity']/(initial if len(account)==1 else account[equitycol].iloc[-2])-1.
    compare(base+'/daily-return identity',observed,expected)
    compare(base+'/continuous account return identity',account['returns'],account[equitycol].to_numpy()/np.r_[initial,account[equitycol].to_numpy()[:-1]]-1.,atol=1e-10)
    if len(holdings):
        valuecol=col(holdings,['value','market_value']);qcol=col(holdings,['quantity']);pcol=col(holdings,['price'])
        compare(base+'/holdings quantity-price identity',holdings[qcol]*holdings[pcol],holdings[valuecol])
        if 'basis' in holdings:
            check(base+'/nonnegative lot-basis structure',bool((holdings['basis']>=-1e-7).all()))
    compare(base+'/terminal return product',np.prod(1.+observed)*initial,terminal['liquidated_equity'])
    for key in ('fees','slippage','taxes'):
        eventcol=col(events,[key,'cash_flow' if key=='taxes' else ('fee' if key=='fees' else key)],False)
        if eventcol:
            compare(base+'/event '+key,(-events.loc[events['event'].isin(['tax_payment','tax_refund']),eventcol].fillna(0.).sum() if key=='taxes' else events[eventcol].fillna(0.).sum())+terminal[key],reported_metrics[key])
    check(base+'/nonnegative cash/equity',bool((account[cashcol]>=-1e-7).all() and (account[equitycol]>0).all()))
    taxcols=[c for c in events if 'tax' in c.lower()]
    check(base+'/tax-event structure',bool(taxcols) or 'event' in events,{'tax_columns':taxcols,'event_columns':list(events.columns),'scope':'Not independent full tax-lot, wash-sale or holding-period certification'})

def audit_custody(study, check):
    study = Path(study)
    custodian = study / 'custodian'
    reads = set()
    cache = {}

    def raw(path):
        path = Path(path)
        reads.add(str(path))
        if str(path) not in cache:
            cache[str(path)] = path.read_bytes()
        return cache[str(path)]

    def sha(path):
        return hashlib.sha256(raw(path)).hexdigest()

    def load(path):
        return json.loads(raw(path))

    manifest = load(custodian / 'manifest.json')
    development = load(custodian / 'DEVELOPMENT_SEALED.json')
    proposal = load(custodian / 'proposal' / 'proposal.json')
    development_log = load(custodian / 'proposal' / 'development_log.json')
    seals = {name: load(custodian / 'seals' / (name + '.json')) for name in
             ('protocol', 'validation_request', 'validation_lock', 'selection',
              'test_request', 'test_lock', 'completion')}
    protocol = seals['protocol']
    selection = seals['selection']
    completion = seals['completion']
    stages = {stage: load(custodian / ('STAGE_' + stage.upper() + '_COMPLETED.json'))
              for stage in ('validation', 'test')}
    audit = [json.loads(line) for line in raw(custodian / 'audit.jsonl').decode().splitlines()
             if line.strip()]
    manifest_hash = sha(custodian / 'manifest.json')
    development_hash = sha(custodian / 'DEVELOPMENT_SEALED.json')
    proposal_hash = sha(custodian / 'proposal' / 'proposal.json')
    log_hash = sha(custodian / 'proposal' / 'development_log.json')
    hashes = {name: sha(custodian / 'seals' / (name + '.json')) for name in seals}

    check('custody_manifest_seal_link', development.get('manifest_sha256') == manifest_hash,
          {'actual_manifest_sha256': manifest_hash,
           'declared': development.get('manifest_sha256')})
    check('custody_initial_hash_declarations',
          development.get('development_input_hashes') == manifest.get('developer_input_hashes')
          and development.get('held_hashes') == manifest.get('held_hashes')
          and development.get('engine_hashes') == manifest.get('engine_hashes'),
          {'verified': 'metadata declaration equality only; original input and frozen code bytes not read'})
    check('custody_prepare_links', len(audit) > 0 and
          audit[0].get('event') == 'prepare'
          and audit[0].get('manifest_sha256') == manifest_hash
          and audit[0].get('partition_hashes') == manifest.get('held_hashes')
          and audit[0].get('source_hashes') ==
              {k: v.get('sha256') for k, v in manifest.get('sources', {}).items()},
          {'event_count': len(audit)})

    bad_hashes = []
    bad_links = []
    regressions = []
    previous = '0' * 64
    for number, event in enumerate(audit, 1):
        payload = {k: v for k, v in event.items() if k != 'event_sha256'}
        actual = hashlib.sha256((json.dumps(payload, sort_keys=True, indent=2) + '\n').encode()).hexdigest()
        if actual != event.get('event_sha256'):
            bad_hashes.append({'line': number, 'actual': actual,
                               'declared': event.get('event_sha256')})
        if event.get('previous_sha256') != previous:
            bad_links.append(number)
        previous = event.get('event_sha256')
        if number > 1 and audit[number - 2].get('timestamp_utc', '') > event.get('timestamp_utc', ''):
            regressions.append({'line': number, 'preceding_event': audit[number - 2].get('event'),
                                'preceding_timestamp': audit[number - 2].get('timestamp_utc'),
                                'event': event.get('event'), 'stage': event.get('stage'),
                                'timestamp': event.get('timestamp_utc')})
    check('custody_audit_event_hashes', not bad_hashes,
          {'count': len(audit), 'encoding': 'sort_keys=True, indent=2, trailing newline, exclude event_sha256',
           'mismatches': bad_hashes})
    check('custody_audit_chain_pointers', not bad_links, {'mismatch_lines': bad_links})
    check('custody_audit_timestamp_monotonicity', not regressions,
          {'regressions': regressions,
           'interpretation': 'held_stage_exit reuses start timestamp; append order is reliable but these values are not completion times'})

    def events(name, stage=None):
        return [(i, event) for i, event in enumerate(audit)
                if event.get('event') == name and (stage is None or event.get('stage') == stage)]

    development_events = events('development_seal')
    proposal_events = events('proposal_lock')
    winner_events = events('winner_lock')
    completion_events = events('test_complete')
    counts_ok = all(len(items) == 1 for items in
                    (development_events, proposal_events, winner_events, completion_events))
    check('custody_single_primary_seals', counts_ok,
          {'development_seal': len(development_events), 'proposal_lock': len(proposal_events),
           'winner_lock': len(winner_events), 'test_complete': len(completion_events)})
    if counts_ok:
        di, de = development_events[0]
        pi, pe = proposal_events[0]
        wi, we = winner_events[0]
        ci, ce = completion_events[0]
        check('custody_development_audit_head', de.get('seal_sha256') == development_hash
              and de.get('previous_sha256') == development.get('audit_head_sha256'),
              {'development_seal_sha256': development_hash})
        check('custody_protocol_event_links', pe.get('protocol_sha256') == hashes['protocol']
              and we.get('selection_sha256') == hashes['selection']
              and ce.get('summary_sha256') == hashes['completion'],
              {'protocol_sha256': hashes['protocol'], 'selection_sha256': hashes['selection'],
               'completion_sha256': hashes['completion']})

    grid = proposal.get('candidates', [])
    grid_hash = hashlib.sha256(json.dumps(grid, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    ids = [candidate.get('candidate_id') for candidate in grid]
    expected_pairs = {(84, 63), (126, 42), (126, 63), (126, 84), (189, 63), (252, 63)}
    expected_configs = {(long_window, short_window, cap) for long_window, short_window in expected_pairs
                        for cap in (1.0 / 3.0, 2.0 / 3.0)}
    actual_configs = {(c.get('parameters', {}).get('long_window'),
                       c.get('parameters', {}).get('short_window'),
                       c.get('parameters', {}).get('cap')) for c in grid}
    candidate_fields_ok = all(
        c.get('parameters', {}).get('mode') == 'agreement'
        and c.get('parameters', {}).get('vol_window') == 63
        and c.get('parameters', {}).get('volatility_ceiling') is None
        and c.get('history_sessions') == max(c['parameters']['long_window'],
                                            c['parameters']['short_window'], 63) + 1
        and c.get('complexity_rank') == 2 for c in grid)
    check('custody_12_candidate_grid', len(grid) == len(set(ids)) == 12
          and protocol.get('frozen_candidates') == 12
          and development_log.get('final_candidate_count') == 12
          and development_log.get('final_candidate_ids') == ids
          and actual_configs == expected_configs and candidate_fields_ok,
          {'candidate_ids': ids, 'frozen_count': protocol.get('frozen_candidates')})
    check('custody_candidate_grid_hash', grid_hash == protocol.get('candidate_grid_sha256')
          == selection.get('candidate_grid_sha256')
          and all(event.get('candidate_grid_sha256') == grid_hash for _, event in proposal_events),
          {'actual': grid_hash, 'encoding': "json.dumps(candidates, sort_keys=True, separators=(',', ':')), no newline"})
    check('custody_development_configuration_count',
          development_log.get('explored_distinct_configurations') == 28
          and len(development_log.get('experiments', [])) == 28
          and protocol.get('explored_configurations') == 28
          and development_log.get('configuration_limit') == 32,
          {'explored': development_log.get('explored_distinct_configurations'),
           'limit': development_log.get('configuration_limit')})

    proposal_file_declarations = protocol.get('proposal_files', {})
    policy_hash = proposal.get('policy_sha256')
    check('custody_proposal_metadata_hashes',
          proposal_file_declarations.get('proposal.json') == proposal_hash
          and proposal_file_declarations.get('development_log.json') == log_hash
          and proposal.get('development_log_sha256') == log_hash
          and development.get('source_hashes', {}).get('proposal.json') == proposal_hash
          and development.get('source_hashes', {}).get('development_log_final.json') == log_hash,
          {'proposal_sha256': proposal_hash, 'development_log_sha256': log_hash})
    check('custody_policy_hash_declaration_consistency',
          policy_hash == proposal_file_declarations.get('policy.py')
          == development.get('source_hashes', {}).get('policy.py')
          == development_log.get('policy_sha256') == selection.get('policy_sha256'),
          {'policy_sha256': policy_hash, 'limit': 'Actual frozen policy bytes deliberately not read'})
    code_hashes = {item.get('path'): item.get('sha256') for item in protocol.get('code_files', [])}
    expected_engine = {'engine.py': code_hashes.get('src/trend_following/research_blind_engine_v4.py'),
                       'ledger_kernel.py': code_hashes.get('src/trend_following/research_ledger.py'),
                       'rate_kernel.py': code_hashes.get('src/trend_following/research_rates.py')}
    check('custody_engine_hash_declaration_consistency',
          expected_engine == manifest.get('engine_hashes') == development.get('engine_hashes')
          and development.get('guard_source_sha256') ==
              code_hashes.get('src/trend_following/research_blind_guard_v4.py')
          and all(stages[stage].get('script_sha256') == expected_engine['engine.py']
                  for stage in stages),
          {'engine_hashes': expected_engine, 'limit': 'Actual frozen engine bytes deliberately not read'})
    check('custody_protocol_seal_links',
          protocol.get('development_seal_sha256') == development_hash
          and selection.get('protocol_sha256') == hashes['protocol']
          and completion.get('protocol_sha256') == hashes['protocol']
          and completion.get('selection_sha256') == hashes['selection'],
          {'development_seal_sha256': development_hash})
    check('custody_selection_rule_consistency', protocol.get('selection') == selection.get('selection_rule')
          and protocol.get('selection', {}).get('annualization') == 252
          and protocol.get('selection', {}).get('standard_deviation_ddof') == 1
          and protocol.get('selection', {}).get('freeze_selected_rule_before_test') is True
          and selection.get('test_used') is False and completion.get('no_test_retuning') is True,
          {'rule': protocol.get('selection'),
           'limit': 'Declarations and chronology verified; cannot erase prior human knowledge'})

    chosen = selection.get('selected')
    check('custody_selected_only_test_request', chosen in grid
          and seals['test_request'].get('candidates') == [chosen]
          and completion.get('selected') == chosen
          and seals['validation_request'].get('candidates') == grid,
          {'selected_id': chosen.get('candidate_id') if isinstance(chosen, dict) else None})
    check('custody_fixed_request_settings',
          seals['validation_request'].get('layers') == ['expense']
          and seals['validation_request'].get('benchmarks') is False
          and seals['test_request'].get('layers') == ['gross', 'trading', 'financing', 'expense', 'tax']
          and seals['test_request'].get('benchmarks') is True
          and seals['validation_request'].get('costs') == seals['test_request'].get('costs')
          and seals['validation_request'].get('start') == '2012-01-01'
          and seals['validation_request'].get('end') == '2015-12-31'
          and seals['test_request'].get('start') == '2016-01-01'
          and seals['test_request'].get('end') == '2026-06-01',
          {'costs': seals['test_request'].get('costs')})

    primary_indices = {}
    result_inventory = []
    for stage in ('validation', 'test'):
        record = stages[stage]
        request = seals[stage + '_request']
        lock = seals[stage + '_lock']
        starts = events('held_stage_start', stage)
        packets = events('held_packet_sealed', stage)
        exits = events('held_stage_exit', stage)
        single = len(starts) == len(packets) == len(exits) == 1
        check('custody_' + stage + '_single_run', single,
              {'starts': len(starts), 'packets': len(packets), 'exits': len(exits)})
        if single:
            primary_indices[stage] = (starts[0][0], packets[0][0], exits[0][0])
            exit_body = {k: v for k, v in exits[0][1].items()
                         if k not in ('event', 'event_sha256', 'previous_sha256', 'input_access_events')}
            check('custody_' + stage + '_stage_record_equals_audit_exit', record == exit_body,
                  {'record_timestamp': record.get('timestamp_utc'), 'run_id': record.get('run_id')})
            check('custody_' + stage + '_packet_hash_declarations',
                  record.get('input_hashes') == packets[0][1].get('input_hashes'),
                  {'input_hashes': record.get('input_hashes')})
        check('custody_' + stage + '_successful_process_exit', record.get('returncode') == 0
              and record.get('timed_out') is False, {'returncode': record.get('returncode')})
        check('custody_' + stage + '_seal_links',
              lock.get('stage') == record.get('stage') == stage
              and record.get('development_seal_sha256') == lock.get('development_seal_sha256') == development_hash
              and record.get('lock_sha256') == hashes[stage + '_lock']
              and record.get('request_sha256') == hashes[stage + '_request']
              and lock.get('proposal_sha256') == record.get('proposal_sha256') == proposal_hash
              and record.get('declared_file_hashes') == lock.get('files'),
              {'lock_sha256': hashes[stage + '_lock'], 'request_sha256': hashes[stage + '_request']})
        expected_lock_files = {
            str(custodian / 'proposal' / 'policy.py'): policy_hash,
            str(custodian / 'proposal' / 'proposal.json'): proposal_hash,
            str(custodian / 'proposal' / 'development_log.json'): log_hash,
            str(custodian / 'seals' / (stage + '_request.json')): hashes[stage + '_request']}
        if stage == 'test':
            expected_lock_files[str(custodian / 'seals' / 'selection.json')] = hashes['selection']
        check('custody_' + stage + '_lock_file_declarations', lock.get('files') == expected_lock_files,
              {'limit': 'JSON metadata bytes verified; policy hash checked by declaration only'})
        # Warmup packet QQQ and manifest legitimately have different declarations
        # from the pure held partition. The rate files should be unchanged.
        check('custody_' + stage + '_held_rate_hash_continuity', all(
              record.get('input_hashes', {}).get(name) == manifest.get('held_hashes', {}).get(stage + '/' + name)
              for name in ('DFF_available.csv', 'DTB3_available.csv')),
              {'QQQ_and_manifest': 'not equated to pure held partition because released packet includes warmup'})
        model_ids = ids if stage == 'validation' else [chosen['candidate_id'], 'QQQ_buy_hold', 'cash', 'synthetic3x_buy_hold']
        layers = ['expense'] if stage == 'validation' else ['gross', 'trading', 'financing', 'expense', 'tax']
        suffixes = ['metrics.csv', 'daily_returns.csv', 'targets.csv', 'terminal.json']
        suffixes += [layer + '_' + suffix for layer in layers
                     for suffix in ('account.csv', 'events.csv', 'holdings.parquet')]
        expected_outputs = {'results/all_metrics.csv'} | {
            'results/' + model + '/' + suffix for model in model_ids for suffix in suffixes}
        declared = record.get('output_hashes', {})
        safe_paths = all(Path(rel).parts and Path(rel).parts[0] == 'results'
                         and '..' not in Path(rel).parts and not Path(rel).is_absolute()
                         for rel in declared)
        check('custody_' + stage + '_complete_declared_output_inventory',
              safe_paths and set(declared) == expected_outputs,
              {'count': len(declared), 'expected_count': len(expected_outputs),
               'missing': sorted(expected_outputs - set(declared)),
               'unexpected': sorted(set(declared) - expected_outputs)})
        mismatches = []
        hashed = 0
        if safe_paths:
            for rel, expected_hash in sorted(declared.items()):
                path = study / 'operator_output' / stage / rel
                result_inventory.append(str(path))
                try:
                    actual = sha(path)
                    hashed += 1
                except OSError as error:
                    mismatches.append({'path': str(path), 'error': str(error)})
                    continue
                if actual != expected_hash:
                    mismatches.append({'path': str(path), 'actual': actual, 'declared': expected_hash})
        check('custody_' + stage + '_all_output_file_hashes',
              safe_paths and hashed == len(declared) and not mismatches,
              {'hashed': hashed, 'declared_count': len(declared), 'mismatches': mismatches})

    check('custody_validation_metric_receipt_hash',
          selection.get('validation_table_sha256') == stages['validation'].get('output_hashes', {}).get('results/all_metrics.csv')
          and selection.get('validation_gate_sha256') == hashes['validation_lock'],
          {'validation_table_sha256': selection.get('validation_table_sha256')})
    check('custody_test_metric_completion_hash',
          completion.get('test_metrics_sha256') == stages['test'].get('output_hashes', {}).get('results/all_metrics.csv'),
          {'test_metrics_sha256': completion.get('test_metrics_sha256')})
    check('custody_test_lock_selection_link',
          seals['test_lock'].get('validation_lock_sha256') == hashes['validation_lock']
          and seals['test_lock'].get('selection_receipt_sha256') == hashes['selection'],
          {'selection_sha256': hashes['selection']})

    expected_inputs = stages['test'].get('input_hashes', {})
    test_input_safe = set(expected_inputs) == {'QQQ.csv', 'DFF_available.csv', 'DTB3_available.csv', 'input_manifest.json'}
    bad_input_hashes = []
    if test_input_safe:
        for name, expected_hash in sorted(expected_inputs.items()):
            path = study / 'released' / 'test' / 'input' / name
            try:
                actual = sha(path)
            except OSError as error:
                bad_input_hashes.append({'path': str(path), 'error': str(error)})
                continue
            if actual != expected_hash:
                bad_input_hashes.append({'path': str(path), 'actual': actual, 'declared': expected_hash})
    check('custody_released_test_input_hashes', test_input_safe and not bad_input_hashes,
          {'count': len(expected_inputs), 'mismatches': bad_input_hashes})

    chronology = counts_ok and len(primary_indices) == 2
    if chronology:
        vi, vpi, vei = primary_indices['validation']
        ti, tpi, tei = primary_indices['test']
        chronology = di < pi < vi < vpi < vei < wi < ti < tpi < tei < ci
        chronology = chronology and (
            development.get('timestamp_utc', '') <= protocol.get('locked_at_utc', '')
            < audit[vi].get('timestamp_utc', '')
            < selection.get('selected_at_utc', '') <= audit[wi].get('timestamp_utc', '')
            < audit[ti].get('timestamp_utc', '') < completion.get('completed_at_utc', ''))
    check('custody_selection_before_test_chronology', chronology,
          {'protocol_locked_at': protocol.get('locked_at_utc'),
           'selected_at': selection.get('selected_at_utc'),
           'test_start': stages['test'].get('timestamp_utc'),
           'completion_at': completion.get('completed_at_utc'),
           'limit': 'Audit append ordering used for exits; exit timestamps are copied start values'})

    probes = [event for event in audit if event.get('event') == 'kernel_probe']
    final_probe_ok = bool(probes) and probes[-1].get('passed') is True and probes[-1].get('returncode') == 0 \
        and probes[-1].get('guard_source_sha256') == development.get('guard_source_sha256')
    check('custody_final_guard_probe_metadata', final_probe_ok,
          {'probe_count': len(probes), 'final_probe_timestamp': probes[-1].get('timestamp_utc') if probes else None,
           'limit': 'Metadata only; supplemental Python open/import audit is not full native syscall tracing'})
    return {'read_inventory': sorted(reads), 'result_inventory': sorted(result_inventory),
            'audit_event_count': len(audit), 'audit_timestamp_regressions': regressions,
            'selected_id': chosen.get('candidate_id') if isinstance(chosen, dict) else None,
            'limitations': [
                'Internal coordinator-controlled hashes and append chain are consistency evidence, not externally anchored signatures.',
                'Actual frozen policy/engine/source bytes and original raw files were deliberately not read.',
                'Validation released input bytes were not read; their packet declarations were cross-checked only.',
                'Held exit timestamps are copied from start timestamps, not recorded completion times.',
                manifest.get('audit_limit'), manifest.get('tool_limit'),
                manifest.get('stages', {}).get('test', {}).get('rate_limitation'),
                manifest.get('stages', {}).get('test', {}).get('availability_limitation')]}


def run(study, output):
    global STUDY
    STUDY=Path(study).resolve(); out=Path(output).resolve()
    if out.exists():
        raise ValueError('Output must be a NEW directory')
    out.mkdir(parents=True)
    schemas={}; summary={}
    try:
        if 'audit_custody' in globals():
            custody=audit_custody(STUDY,check)
            READS.extend(str(Path(p).relative_to(STUDY)) for p in custody['read_inventory'])
        policy=STUDY/'custodian/proposal/policy.py';READS.append(str(policy.relative_to(STUDY)))
        check('custody_actual_proposal_policy_hash',hashlib.sha256(policy.read_bytes()).hexdigest()==read_json(STUDY/'custodian/seals/selection.json')['policy_sha256'])
        selection=read_json(STUDY/'custodian/seals/selection.json')['selected']
        request=read_json(STUDY/'custodian/seals/test_request.json'); costs=request['costs']
        check('selected method fixed',selection['candidate_id']=='agree_l252_s63_vnone_c67' and selection['parameters']=={'mode':'agreement','long_window':252,'short_window':63,'vol_window':63,'volatility_ceiling':None,'cap':2./3.})
        raw=read_csv(STUDY/'released/test/input/QQQ.csv')
        dff=read_csv(STUDY/'released/test/input/DFF_available.csv'); dtb=read_csv(STUDY/'released/test/input/DTB3_available.csv')
        schemas['input']={k:list(f.columns) for k,f in [('QQQ',raw),('DFF',dff),('DTB3',dtb)]}
        # A market session is a date, not a midnight accrual timestamp.
        dates=pd.to_datetime(raw['session'].astype(str).str[:10],format='%Y-%m-%d')
        sessions=(dates+pd.Timedelta(hours=16)).dt.tz_localize('America/New_York').dt.tz_convert('UTC')
        check('raw sessions unique increasing',sessions.is_unique and sessions.is_monotonic_increasing)
        ret=raw_returns(raw); decisions=selected_targets(ret,selection['parameters'])
        scoring=(sessions>=pd.Timestamp(request['start'],tz='America/New_York')) & (sessions<=pd.Timestamp(request['end']+' 23:59:59',tz='America/New_York'))
        ix=np.flatnonzero(scoring.to_numpy()); check('test session count',len(ix)==2617,{'observed':len(ix)})
        check('test starts/ends',str(sessions.iloc[ix[0]].tz_convert('America/New_York').date())=='2016-01-04' and str(sessions.iloc[ix[-1]].tz_convert('America/New_York').date())=='2026-06-01')
        # Full-history synthetic NAV is normalized once, at first released input.
        fulls=pd.DatetimeIndex(sessions)
        fullcf=integrate_known_rate(dtb,fulls,365.)
        fullff=2.*integrate_known_rate(dff,fulls,360.,costs['financing_spread_bps']/10000.)
        elapsed=np.r_[0.,np.diff(fulls.asi8)/(1e9*86400.*365.)]
        fullexp=(costs['expense_ratio']+costs['extra_drag'])*elapsed
        fullnav={layer:100.*np.cumprod(1.+3.*ret-(fullff if layer in ('financing','expense') else 0.)-(fullexp if layer=='expense' else 0.)) for layer in ('gross','trading','financing','expense')}
        s=pd.DatetimeIndex(sessions.iloc[ix]);cf=fullcf[ix].copy();cf[0]=0.
        market=raw.iloc[ix].reset_index(drop=True).copy();market['total_return']=ret[ix];market['cash_return']=cf;market['funding_return']=fullff[ix];market['expense_return']=fullexp[ix]
        for layer,nav in fullnav.items():market['nav_'+layer]=nav[ix]
        check('positive full-history synthetic NAV',all(np.all(v>0.) for v in fullnav.values()))
        market.to_csv(out/'independent_market_reconstruction.csv',index=False)
        allmetrics=read_csv(STUDY/'operator_output/test/results/all_metrics.csv')
        tax_rf=read_csv(STUDY/'operator_output/test/results/cash/daily_returns.csv')['tax'].to_numpy(dtype=float)
        validation_metrics=read_csv(STUDY/'operator_output/validation/results/all_metrics.csv')
        validation_mask=(sessions>=pd.Timestamp('2012-01-01',tz='America/New_York')) & (sessions<pd.Timestamp('2016-01-01',tz='America/New_York'))
        validation_cf=fullcf[np.flatnonzero(validation_mask.to_numpy())].copy();validation_cf[0]=0.
        validation_scores=[]
        proposal=read_json(STUDY/'custodian/proposal/proposal.json')
        for candidate in proposal['candidates']:
            cid=candidate['candidate_id']
            vret=read_csv(STUDY/'operator_output/validation/results'/cid/'daily_returns.csv')['expense'].to_numpy(dtype=float)
            ex=vret-validation_cf;score=float(ex.mean()/ex.std(ddof=1)*np.sqrt(252.))
            expected=float(validation_metrics.loc[validation_metrics['candidate_id']==cid,'cash_excess_Sharpe'].iloc[0])
            compare('validation/'+cid+'/cash-excess Sharpe',expected,score,atol=1e-10)
            validation_scores.append((cid,score,candidate['complexity_rank']))
        finite=[item for item in validation_scores if np.isfinite(item[1])]
        maximum=max(item[1] for item in finite);tied=[item for item in finite if abs(item[1]-maximum)<=1e-10]
        winner=sorted(tied,key=lambda x:(x[2],x[0]))[0][0]
        check('validation independent fixed-objective winner',winner==selection['candidate_id'],{'winner':winner,'scores':validation_scores})

        candidates=[(selection['candidate_id'],'synthetic3x',decisions[ix]),('QQQ_buy_hold','QQQ',np.ones(len(ix))),('synthetic3x_buy_hold','synthetic3x',np.ones(len(ix))),('cash','QQQ',np.zeros(len(ix)))]
        for cid,kind,targets in candidates:
            folder=STUDY/'operator_output/test/results'/cid
            observed_targets=read_csv(folder/'targets.csv'); daily=read_csv(folder/'daily_returns.csv'); terminal=read_json(folder/'terminal.json')
            schemas[cid]={'targets':list(observed_targets.columns),'daily_returns':list(daily.columns)}
            tc=col(observed_targets,['target','target_weight','weight','decision_target','0'])
            compare(cid+'/one-session-lagged fill targets',observed_targets[tc],np.r_[0.,targets[:-1]],atol=1e-12,rtol=0.)
            candidate_metrics=read_csv(folder/'metrics.csv')
            check(cid+'/declared accounting asset',bool((candidate_metrics['asset']==kind).all()),{'asset':kind,'note':'Cash uses a zero-quantity QQQ placeholder; quote marks remain independently checked.'})
            summary[cid]={}
            for layer in ('gross','trading','financing','expense'):
                account=read_csv(folder/(layer+'_account.csv')); events=read_csv(folder/(layer+'_events.csv'))
                parquet=folder/(layer+'_holdings.parquet'); READS.append(str(parquet.relative_to(STUDY))); holdings=pd.read_parquet(parquet)
                schemas[cid][layer]={'account':list(account.columns),'events':list(events.columns),'holdings':list(holdings.columns)}
                ledger, rets, term, aggregates=independent_ledger(market,targets,kind,layer,costs)
                ledger.insert(0,'session',s.astype(str)); ledger.to_csv(out/(cid+'_'+layer+'_independent.csv'),index=False)
                for key,choices in {'cash':['cash'],'equity':['equity','marked_equity'],'quantity':['quantity','qty','shares'],'price':['price','asset_price','close']}.items():
                    actualcol=col(account,choices,False)
                    if actualcol:
                        compare(cid+'/'+layer+'/daily '+key,account[actualcol],ledger[key])
                    elif key in ('quantity','price'):
                        hc=col(holdings,choices,False); hs=col(holdings,['timestamp','session','date'],False); ass=col(account,['timestamp','session','date'],False)
                        if hc and hs and ass:
                            grouped=holdings.groupby(pd.to_datetime(holdings[hs],utc=True))[hc].sum() if key=='quantity' else holdings.groupby(pd.to_datetime(holdings[hs],utc=True))[hc].last()
                            merged=pd.to_datetime(account[ass],utc=True).map(grouped).fillna(0. if key=='quantity' else np.nan)
                            compare(cid+'/'+layer+'/daily holdings '+key,merged if key=='quantity' else merged.dropna(),ledger[key] if key=='quantity' else ledger.loc[merged.notna(),key])
                        else:
                            check(cid+'/'+layer+'/daily '+key+' exposed',False)
                dc=col(daily,[layer,layer+'_return','return_'+layer])
                compare(cid+'/'+layer+'/daily returns',daily[dc],rets,atol=1e-10)
                compare(cid+'/'+layer+'/continuous account returns',account['returns'],ledger['return'],atol=1e-10)
                compare(cid+'/'+layer+'/daily normalized turnover',account['turnover'],ledger['turnover'].to_numpy()/np.r_[float(costs['initial_cash']),ledger['equity'].to_numpy()[:-1]],atol=1e-10)
                compare(cid+'/'+layer+'/pretax account taxes',account['taxes'],np.zeros(len(account)),atol=1e-12)
                compare(cid+'/'+layer+'/events fees',events['fee'].fillna(0.).sum(),aggregates['fees']-term['fees'])
                compare(cid+'/'+layer+'/events slippage',events['slippage'].fillna(0.).sum(),aggregates['slippage']-term['slippage'])
                for key in term:
                    compare(cid+'/'+layer+'/terminal '+key,terminal[layer][key],term[key])
                calc=metrics(rets,cf,ledger,term,aggregates,targets,s,costs['initial_cash'])
                observed=candidate_metrics.loc[candidate_metrics['layer']==layer].iloc[0]
                for key,value in calc.items():
                    compare(cid+'/'+layer+'/metric '+key,observed[key],value,atol=1e-8)
                summary[cid][layer]=calc
            account=read_csv(folder/'tax_account.csv'); events=read_csv(folder/'tax_events.csv')
            parquet=folder/'tax_holdings.parquet'; READS.append(str(parquet.relative_to(STUDY))); holdings=pd.read_parquet(parquet)
            schemas[cid]['tax']={'account':list(account.columns),'events':list(events.columns),'holdings':list(holdings.columns)}
            dc=col(daily,['tax','tax_return','return_tax']); tax_metric=candidate_metrics.loc[candidate_metrics['layer']=='tax'].iloc[0]
            tax_identities(cid+'/tax',account,events,holdings,daily[dc],terminal['tax'],tax_metric,costs)
            # Pure structural tax metric identities: exported matched tax-cash RF, not a lot engine.
            taxeq=account['equity'].to_numpy(dtype=float); taxcash=account['cash'].to_numpy(dtype=float)
            taxledger=pd.DataFrame({'equity':taxeq,'effective_exposure':(taxeq-taxcash)/taxeq*(1. if kind=='QQQ' else 3.),'cash_weight':taxcash/taxeq})
            taxagg={'filled_orders':int(events['event'].isin(['buy','sell']).sum())+terminal['tax']['orders'],'fees':events['fee'].fillna(0.).sum()+terminal['tax']['fees'],'slippage':events['slippage'].fillna(0.).sum()+terminal['tax']['slippage'],'taxes':-events.loc[events['event'].isin(['tax_payment','tax_refund']),'cash_flow'].fillna(0.).sum()+terminal['tax']['taxes']}
            taxcalc=metrics(daily[dc],tax_rf,taxledger,terminal['tax'],taxagg,targets,s,costs['initial_cash'])
            for key,value in taxcalc.items():compare(cid+'/tax/structural metric '+key,tax_metric[key],value,atol=1e-8)
            summary[cid]['tax_structural_metrics']=taxcalc
        sel=summary[selection['candidate_id']]['expense']; qqq=summary['QQQ_buy_hold']['expense']
        summary['primary_cash_excess_sharpe_difference']=sel['cash_excess_Sharpe']-qqq['cash_excess_Sharpe']
        check('primary superiority not demonstrated',summary['primary_cash_excess_sharpe_difference']<0.,{'selected_minus_QQQ':summary['primary_cash_excess_sharpe_difference']})
    except Exception as exc:
        check('audit execution',False,{'exception':repr(exc),'traceback':traceback.format_exc()})
    finally:
        (out/'schemas.json').write_text(json.dumps(schemas,indent=2))
        (out/'read_files.json').write_text(json.dumps(sorted(set(READS)),indent=2))
        report={'arithmetic_passed':all(x['passed'] for x in CHECKS if not x['check'].startswith('custody_')),'custody_passed':all(x['passed'] for x in CHECKS if x['check'].startswith('custody_')),'audit':'Independent new-v4 arithmetic reconstruction','passed':all(x['passed'] for x in CHECKS),'checks':CHECKS,'independent_metrics':summary,'scope':'All four pretax layers; tax event/account/return/metric identities only, not full tax-lot certification; latest-vintage rate and assumed availability limitations retained. Targets are nominal fractions of PRE-COST marked NAV; fees/slippage are paid after target sizing, so partial actual weights can exceed nominal caps. Full-weight buys are scaled to cash affordability. Cash is a zero-quantity QQQ placeholder; its raw marks are independently checked, with no equity exposure.'}
        (out/'audit.json').write_text(json.dumps(report,indent=2,allow_nan=True))
        failures=[x for x in CHECKS if not x['passed']]
        lines=['# Independent v4 post-evaluation audit','',f"Checks passed: {len(CHECKS)-len(failures)}/{len(CHECKS)}.", '', 'This is an independent reconstruction, not a policy reoptimization or a tax-lot certification.', '', 'Nominal targets use PRE-COST marked NAV. Fees/slippage reduce post-trade NAV, so partial actual weights can exceed nominal caps; full-weight buys are cash-affordability-scaled. The audit uses this frozen mechanical convention, not the initially misstated post-cost abstraction.', '', '## Failures']
        lines += [f"- {x['check']}: `{json.dumps(x['details'],default=str)}`" for x in failures]
        if not failures: lines.append('- None.')
        (out/'audit.md').write_text('\n'.join(lines)+'\n')
        print(json.dumps({'passed':report['passed'],'checks':len(CHECKS),'failures':len(failures),'output':str(out)}))
    return 0 if all(x['passed'] for x in CHECKS) else 1

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--study',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();raise SystemExit(run(args.study,args.output))
