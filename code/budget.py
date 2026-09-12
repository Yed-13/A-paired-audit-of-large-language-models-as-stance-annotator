"""Durable shared request-budget reservations. Amounts are USD, not token counts."""
import fcntl
import json
import math
import os
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def locked(path):
    path=Path(path)
    with path.open('r+',encoding='utf-8') as f:
        fcntl.flock(f,fcntl.LOCK_EX)
        obj=json.load(f)
        yield obj
        f.seek(0);json.dump(obj,f,indent=2,allow_nan=False);f.truncate();f.flush();os.fsync(f.fileno())


def reserve(path,key,amount):
    if not math.isfinite(amount) or amount<=0:
        raise ValueError('Invalid cost reservation')
    with locked(path) as b:
        if b.get('authorization')!='user_confirmed_numeric_limit':
            raise ValueError('Budget needs explicit numeric user authorization')
        if key in b['requests']:
            raise ValueError('Request already reserved; resolve ledger rather than replay')
        used=sum(r['accounted_usd'] for r in b['requests'].values())
        if used+amount>b['limit_usd']:
            raise ValueError('Budget ceiling reached before request')
        b['requests'][key]={'reserved_usd':amount,'accounted_usd':amount,'status':'reserved'}


def settle(path,key,cost):
    with locked(path) as b:
        r=b['requests'][key]
        if cost is None:
            r['status']='unknown_cost_reservation_retained'
            return
        if not isinstance(cost,(int,float)) or not math.isfinite(cost) or cost<0:
            raise ValueError('Invalid provider cost')
        r.update(accounted_usd=float(cost),status='settled',actual_usd=float(cost))
        if cost>r['reserved_usd']:
            b['halt_reason']='Provider cost exceeded conservative reservation; review prices before resuming'


def request_bound(model,messages):
    guard=model['cost_guard']
    text_bytes=len(json.dumps(messages,ensure_ascii=False).encode('utf-8'))
    if text_bytes+512>guard['max_input_tokens_bound']:
        raise ValueError('Prompt exceeds conservative byte-based input allowance')
    return (guard['max_input_tokens_bound']*guard['input_usd_per_million']+
            model['parameters']['max_tokens']*guard['output_usd_per_million'])/1e6
