# fake_pool.py - minimal bntpool-like stratum server (protocol of seine's pool.rs tests), cloud check.
import socket, json, sys, threading
s = socket.socket(); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1); s.bind(('127.0.0.1', 13333)); s.listen(1)
c, _ = s.accept(); f = c.makefile('rw')
login = json.loads(f.readline()); print('LOGIN', login, flush=True)
f.write(json.dumps({"id": 1, "status": "ok", "result": {"protocol_version": 2, "capabilities": ["submit_claimed_hash"], "required_capabilities": []}}) + '\n')
f.write(json.dumps({"method": "job", "params": {"job_id": "j1", "header_base": "11" * 100, "target": "ff" * 32, "difficulty": 1, "height": 5, "nonce_start": 1000, "nonce_end": 1003}}) + '\n'); f.flush()
n = 0
while n < 3:
    line = f.readline()
    if not line: break
    m = json.loads(line); print('SUBMIT', m, flush=True); n += 1
    f.write(json.dumps({"id": m["id"], "status": "ok", "result": {"accepted": True, "difficulty": 1}}) + '\n'); f.flush()
c.close()
