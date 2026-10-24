// BntMiner - GPU miner for Blocknet (BNT, blocknetcrypto.com), Argon2id 2 GiB t=1 p=1,
// bntpool.com stratum.  No developer fee.  Apache-2.0.
//
//   bntminer.exe                     mine with bntminer.conf
//   bntminer.exe --test              CPU reference + GPU engine against the Rust argon2 vectors
//   bntminer.exe --bench [seconds]   H/s benchmark on the selected GPUs
//   bntminer.exe --profile           time of the Argon2i half and the Argon2d half of one hash
//
// Options (also key=value lines in bntminer.conf, without "--"):
//   -o / --pool URL (repeatable, failover order)   -u / --user ADDRESS   --worker RIG
//   --gpus 0,1|all   --lanes N (hashes in parallel per GPU, 0 = as many as the memory allows)
//   --reserve-mb N (GPU memory left free, default 160)   --chunk N (blocks per kernel launch)
//   --stats N (seconds)   --log FILE   --args-file FILE   --cafile FILE
//
// Stratum of bntpool.com (JSON lines, protocol of the seine miner, checked in its pool.rs):
//   -> {"id":1,"method":"login","params":{"address","worker","protocol_version":2,"capabilities":[...]}}
//   <- {"id":1,"status":"ok","result":{"protocol_version","capabilities","required_capabilities"}}
//   <- {"method":"job","params":{"job_id","header_base" (hex, header without nonce),"target"
//       (32 bytes BE hex),"difficulty","network_target"?,"height","nonce_start","nonce_end"}}
//   -> {"id":n,"method":"submit","params":{"job_id","nonce" (u64),"claimed_hash" (hex),"miner_version"}}
//   <- {"id":n,"status":"ok","result":{"accepted":true,"difficulty":d}}  or  {"id":n,"error":"..."}
//   <- {"method":"notification","params":{"kind","message"}}
// PoW: hash = Argon2id(password = nonce LE, salt = header_base), share if hash <= target.
#include <stdio.h>
#include <stdlib.h>
#include <cstdlib>
#include <string.h>
#include <stdarg.h>
#include <time.h>
#include <math.h>
#include <string>
#include <vector>
#include <deque>
#include <map>
#include <mutex>
#include <thread>
#include <atomic>
#include <chrono>
#include <memory>
#include <random>
#include <signal.h>
#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif

#include "argon_ref.h"
#include "bnt_vectors.h"
#include "json.h"
#include "tls_conn.h"
#include "../gpu/bnt_gpu.h"

#define BNTMINER_VERSION "0.1.0"
#define BNT_M_BLOCKS 2097152u

using Clock = std::chrono::steady_clock;

// ---------------------------------------------------------------------------
// logging
// ---------------------------------------------------------------------------
static std::mutex g_log_mtx;
static FILE* g_logf = nullptr;

static void logf_(const char* fmt, ...)
{
    char msg[2048];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(msg, sizeof msg, fmt, ap);
    va_end(ap);
    time_t t = time(nullptr);
    struct tm tmv;
#ifdef _WIN32
    localtime_s(&tmv, &t);
#else
    localtime_r(&t, &tmv);
#endif
    char ts[32];
    strftime(ts, sizeof ts, "%Y-%m-%d %H:%M:%S", &tmv);
    std::lock_guard<std::mutex> lk(g_log_mtx);
    printf("[%s] %s\n", ts, msg);
    fflush(stdout);
    if (g_logf) { fprintf(g_logf, "[%s] %s\n", ts, msg); fflush(g_logf); }
}
#define LOG(...) logf_(__VA_ARGS__)

// ---------------------------------------------------------------------------
// hex / target helpers
// ---------------------------------------------------------------------------
static int hexval(char c)
{
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}
static std::string strip0x(const std::string& s)
{
    if (s.size() >= 2 && s[0] == '0' && (s[1] == 'x' || s[1] == 'X')) return s.substr(2);
    return s;
}
static bool hex_to_vec(const std::string& h0, std::vector<uint8_t>& out)
{
    std::string h = strip0x(h0);
    if (h.size() % 2) return false;
    out.resize(h.size() / 2);
    for (size_t i = 0; i < out.size(); i++) {
        int a = hexval(h[2 * i]), b = hexval(h[2 * i + 1]);
        if (a < 0 || b < 0) return false;
        out[i] = (uint8_t)(a * 16 + b);
    }
    return true;
}
static bool hex_to_bytes(const std::string& h, uint8_t* out, size_t n)
{
    std::vector<uint8_t> v;
    if (!hex_to_vec(h, v) || v.size() != n) return false;
    memcpy(out, v.data(), n);
    return true;
}
static std::string to_hex(const uint8_t* p, size_t n)
{
    static const char* d = "0123456789abcdef";
    std::string s(2 * n, '0');
    for (size_t i = 0; i < n; i++) { s[2 * i] = d[p[i] >> 4]; s[2 * i + 1] = d[p[i] & 15]; }
    return s;
}
// expected hashes per hit for a big-endian target (2^256 / (target + 1))
static double target_hashes(const uint8_t t[32])
{
    double v = 0;
    for (int i = 0; i < 32; i++) v = v * 256.0 + t[i];
    return pow(2.0, 256) / (v + 1.0);
}

// ---------------------------------------------------------------------------
// configuration
// ---------------------------------------------------------------------------
struct Config {
    std::vector<std::string> pools;
    std::string user, worker = "rig";
    std::vector<int> gpus;          // empty = all
    int lanes = 0;                  // per GPU, 0 = auto
    int reserve_mb = 160;
    int chunk = 0;
    std::string cafile;
    int stats = 60;
    std::string logfile;
    bool test = false, bench = false, profile = false;
    int bench_seconds = 120;
    std::string exe_dir = ".";
};

static std::string trim(const std::string& s)
{
    size_t a = s.find_first_not_of(" \t\r\n"), b = s.find_last_not_of(" \t\r\n");
    return a == std::string::npos ? "" : s.substr(a, b - a + 1);
}

static bool apply_option(Config& c, const std::string& key, const std::string& val)
{
    std::string k = key;
    while (!k.empty() && k[0] == '-') k.erase(0, 1);
    if (k == "o" || k == "pool" || k == "url") c.pools.push_back(val);
    else if (k == "u" || k == "user" || k == "wallet" || k == "address") c.user = val;
    else if (k == "p" || k == "pass") { /* unused by bntpool */ }
    else if (k == "worker" || k == "rig") c.worker = val;
    else if (k == "gpus" || k == "devices" || k == "d") {
        c.gpus.clear();
        if (val != "all" && !val.empty()) {
            size_t p = 0;
            while (p <= val.size()) {
                size_t q = val.find(',', p);
                std::string one = trim(val.substr(p, q == std::string::npos ? std::string::npos : q - p));
                if (!one.empty()) c.gpus.push_back(atoi(one.c_str()));
                if (q == std::string::npos) break;
                p = q + 1;
            }
        }
    }
    else if (k == "lanes") c.lanes = atoi(val.c_str());
    else if (k == "reserve-mb") c.reserve_mb = atoi(val.c_str());
    else if (k == "chunk") c.chunk = atoi(val.c_str());
    else if (k == "cafile") c.cafile = val;
    else if (k == "stats") c.stats = atoi(val.c_str());
    else if (k == "log") c.logfile = val;
    else return false;
    return true;
}

static bool load_config_file(Config& c, const std::string& path)
{
    FILE* f = fopen(path.c_str(), "r");
    if (!f) return false;
    char line[1024];
    while (fgets(line, sizeof line, f)) {
        std::string l = trim(line);
        if (l.empty() || l[0] == '#' || l[0] == ';') continue;
        size_t eq = l.find('=');
        if (eq == std::string::npos) continue;
        std::string k = trim(l.substr(0, eq)), v = trim(l.substr(eq + 1));
        if (!apply_option(c, k, v)) fprintf(stderr, "config: unknown key '%s'\n", k.c_str());
    }
    fclose(f);
    return true;
}

static std::string exe_directory(const char* argv0)
{
#ifdef _WIN32
    char buf[MAX_PATH];
    DWORD n = GetModuleFileNameA(NULL, buf, MAX_PATH);
    std::string p = n ? std::string(buf, n) : std::string(argv0);
#else
    std::string p = argv0;
    char buf[4096];
    ssize_t n = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (n > 0) p.assign(buf, (size_t)n);
#endif
    size_t s = p.find_last_of("\\/");
    return s == std::string::npos ? "." : p.substr(0, s);
}

static bool file_exists(const std::string& p)
{
    FILE* f = fopen(p.c_str(), "rb");
    if (!f) return false;
    fclose(f);
    return true;
}

// ---------------------------------------------------------------------------
// shared mining state
// ---------------------------------------------------------------------------
struct PoolJob {
    uint64_t seq = 0;
    std::string job_id;
    uint64_t height = 0;
    std::vector<uint8_t> header;     // header_base
    uint8_t target[32];              // share target, big-endian
    uint8_t net_target[32];          // block target (zero if unknown)
    uint64_t nonce_start = 0, nonce_end = 0;
    uint64_t difficulty = 0;
    mutable std::atomic<uint64_t> next{ 0 };   // next nonce to hand out
};

struct Share {
    std::string job_id, hash_hex;
    uint64_t nonce;
    int gpu;
    bool block;
};

static std::atomic<bool> g_running(true);
static std::atomic<int> g_gpu_failed(0);
static std::mutex g_mtx;
static std::shared_ptr<const PoolJob> g_job;
static bool g_paused = true;
static uint64_t g_job_seq = 0;
static std::deque<Share> g_shares;

struct GpuStat {
    std::atomic<uint64_t> hashes{ 0 };
    std::atomic<uint64_t> aborted{ 0 };
    std::atomic<uint64_t> shares{ 0 };
    std::atomic<int> run_ms{ 0 };
    std::atomic<int> lanes{ 0 };
    std::atomic<bool> alive{ false };
    std::atomic<long long> beat_ms{ 0 };
    volatile int abort = 0;
    std::string name;
    int dev = -1;
};
static std::vector<std::unique_ptr<GpuStat>> g_stats;

static long long now_ms()
{
    return std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now().time_since_epoch()).count();
}

#define BM_STUCK_MS 60000
static void deadline_watchdog(int seconds, const char* what)
{
    std::thread([seconds, what]() {
        for (int i = 0; i < seconds * 10; i++) std::this_thread::sleep_for(std::chrono::milliseconds(100));
        LOG("%s TIMEOUT after %d s: a GPU is stuck (unstable clocks?), exiting with code 5", what, seconds);
        fflush(nullptr);
        std::_Exit(5);
    }).detach();
}

static std::atomic<uint64_t> g_accepted(0), g_rejected(0), g_blocks(0);
static void on_signal(int) { g_running = false; }

// blocks 0 and 1 of one lane
static void make_init(const std::vector<uint8_t>& header, uint64_t nonce, uint32_t m, uint64_t* init256)
{
    bnt::initial_blocks(header.data(), header.size(), nonce, m, init256, init256 + 128);
}

// ---------------------------------------------------------------------------
// GPU worker
// ---------------------------------------------------------------------------
static void gpu_worker(const Config& cfg, int idx, int dev)
{
    GpuStat& st = *g_stats[idx];
    void* g = bg_create(dev, BNT_M_BLOCKS, cfg.lanes, cfg.reserve_mb, cfg.chunk);
    if (!g) {
        LOG("GPU %d: init failed: %s", dev, bg_last_error(nullptr));
        g_gpu_failed = dev + 1;
        g_running = false;
        return;
    }
    const int L = bg_lanes(g);
    st.name = bg_name(g);
    st.lanes = L;
    st.beat_ms = now_ms();
    st.alive = true;
    LOG("GPU %d: %s, sm_%d, %d SMs, %.0f MB, %d lane(s) of 2 GiB (%.0f MB used), %s", dev, st.name.c_str(), bg_compute_capability(g), bg_sm_count(g),
        bg_memory_total(g) / 1048576.0, L, bg_memory_used(g) / 1048576.0, bg_kernel_info());
    std::vector<uint64_t> init((size_t)L * 256), last((size_t)L * 128);
    std::vector<uint64_t> nonces(L);

    while (g_running) {
        st.beat_ms = now_ms();
        std::shared_ptr<const PoolJob> job;
        bool paused;
        {
            std::lock_guard<std::mutex> lk(g_mtx);
            job = g_job;
            paused = g_paused;
        }
        if (paused || !job) { std::this_thread::sleep_for(std::chrono::milliseconds(50)); continue; }
        int n = 0;
        for (; n < L; n++) {
            uint64_t v = job->next.fetch_add(1);
            if (v > job->nonce_end || v < job->nonce_start) break;
            nonces[n] = v;
            make_init(job->header, v, BNT_M_BLOCKS, init.data() + (size_t)n * 256);
        }
        if (n == 0) {   // nonce range used up: wait for the next job
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
            continue;
        }
        st.abort = 0;
        double ms = 0;
        int r = bg_hash(g, init.data(), last.data(), n, &st.abort, &ms);
        st.beat_ms = now_ms();
        if (r < 0) {
            LOG("GPU %d: CUDA error: %s", dev, bg_last_error(g));
            g_gpu_failed = dev + 1;
            g_running = false;
            break;
        }
        if (r == 1) { st.aborted += n; continue; }   // new block: these hashes are stale
        st.hashes += n;
        st.run_ms = (int)ms;
        for (int l = 0; l < n; l++) {
            uint8_t h[32];
            bnt::final_hash(last.data() + (size_t)l * 128, h);
            if (!bnt::hash_le_target(h, job->target)) continue;
            Share sh;
            sh.job_id = job->job_id;
            sh.nonce = nonces[l];
            sh.hash_hex = to_hex(h, 32);
            sh.gpu = dev;
            uint8_t zero[32] = { 0 };
            sh.block = memcmp(job->net_target, zero, 32) && bnt::hash_le_target(h, job->net_target);
            st.shares++;
            if (sh.block) LOG("GPU %d: BLOCK candidate at height %llu (hash %s)", dev, (unsigned long long)job->height, sh.hash_hex.c_str());
            std::lock_guard<std::mutex> lk(g_mtx);
            g_shares.push_back(sh);
        }
    }
    st.alive = false;
    bg_destroy(g);
}

// ---------------------------------------------------------------------------
// pool client (bntpool.com)
// ---------------------------------------------------------------------------
struct Pending { int gpu; std::string job_id, hash; bool block; uint64_t nonce; };

static std::string jstr(const json::Value* o, const char* k)
{
    const json::Value* v = o ? o->get(k) : nullptr;
    return (v && (v->is_str() || v->is_num())) ? v->s : std::string();
}
static uint64_t ju64(const json::Value* o, const char* k, uint64_t def = 0)
{
    std::string s = jstr(o, k);
    if (s.empty()) return def;
    return strtoull(s.c_str(), nullptr, 10);
}

static bool parse_job(const json::Value* p, PoolJob& j, std::string& why)
{
    if (!p || !p->is_obj()) { why = "params is not an object"; return false; }
    j.job_id = jstr(p, "job_id");
    if (j.job_id.empty()) { why = "missing job_id"; return false; }
    if (!hex_to_vec(jstr(p, "header_base"), j.header) || j.header.size() < 8 || j.header.size() > 1024) { why = "bad header_base"; return false; }
    if (!hex_to_bytes(jstr(p, "target"), j.target, 32)) { why = "bad target (32 bytes hex expected)"; return false; }
    std::string nt = jstr(p, "network_target");
    if (nt.empty() || !hex_to_bytes(nt, j.net_target, 32)) memset(j.net_target, 0, 32);
    j.height = ju64(p, "height");
    j.difficulty = ju64(p, "difficulty");
    j.nonce_start = ju64(p, "nonce_start");
    j.nonce_end = ju64(p, "nonce_end", UINT64_MAX);
    if (j.nonce_end < j.nonce_start) { why = "empty nonce range"; return false; }
    j.next = j.nonce_start;
    return true;
}

static void pool_worker(const Config& cfg)
{
    TlsConn conn;
    size_t pool_idx = 0;
    int backoff = 2;
    while (g_running) {
        std::string url = cfg.pools[pool_idx % cfg.pools.size()];
        LOG("pool: connecting to %s", url.c_str());
        if (!conn.connect(url, cfg.cafile, 10000)) {
            LOG("pool: %s", conn.error().c_str());
            pool_idx++;
            for (int i = 0; i < backoff * 10 && g_running; i++) std::this_thread::sleep_for(std::chrono::milliseconds(100));
            backoff = backoff * 2 > 60 ? 60 : backoff * 2;
            continue;
        }
        LOG("pool: connected to %s", conn.host().c_str());
        int next_id = 2;
        std::map<int, Pending> pending;
        bool logged_in = false, send_claimed = true;
        int raw_lines = 0;
        auto t_conn = Clock::now();
        auto t_rx = Clock::now();
        std::string login = std::string("{\"id\":1,\"method\":\"login\",\"params\":{\"address\":") + json::quote(cfg.user) +
                            ",\"worker\":" + json::quote(cfg.worker) +
                            ",\"protocol_version\":2,\"capabilities\":[\"login_negotiation\",\"share_validation_status\",\"submit_claimed_hash\"]}}";
        conn.send_line(login);
        bool lost = false;

        while (g_running && !lost) {
            std::vector<std::string> lines;
            if (!conn.poll_lines(lines, 100)) { LOG("pool: %s", conn.error().c_str()); lost = true; break; }
            if (!lines.empty()) t_rx = Clock::now();
            for (auto& line : lines) {
                if (raw_lines < 4) { raw_lines++; LOG("pool <- %.500s", line.c_str()); }
                json::Value v;
                if (!json::parse(line, v) || !v.is_obj()) { LOG("pool: unparsable line: %.200s", line.c_str()); continue; }
                const json::Value* method = v.get("method");
                const json::Value* params = v.get("params");
                if (method && method->is_str()) {
                    const std::string& m = method->s;
                    if (m == "job") {
                        auto j = std::make_shared<PoolJob>();
                        std::string why;
                        if (!parse_job(params, *j, why)) { LOG("pool: ignoring job: %s: %.300s", why.c_str(), line.c_str()); continue; }
                        bool new_header;
                        {
                            std::lock_guard<std::mutex> lk(g_mtx);
                            j->seq = ++g_job_seq;
                            new_header = !g_job || g_job->header != j->header;
                            g_job = j;
                            g_paused = false;
                        }
                        if (new_header)
                            for (auto& s : g_stats) s->abort = 1;   // hashes of the old header are stale
                        LOG("pool: new job %s height %llu%s, share difficulty %llu (~%.0f hashes per share), nonces %llu..%llu", j->job_id.c_str(),
                            (unsigned long long)j->height, new_header ? " (new template)" : "", (unsigned long long)j->difficulty,
                            target_hashes(j->target), (unsigned long long)j->nonce_start, (unsigned long long)j->nonce_end);
                    } else if (m == "notification") {
                        std::string msg = params && params->is_str() ? params->s : jstr(params, "message");
                        std::string kind = params && params->is_obj() ? jstr(params, "kind") : "pool_notice";
                        LOG("pool notification (%s): %s", kind.c_str(), msg.c_str());
                    } else {
                        LOG("pool: unhandled method %s: %.300s", m.c_str(), line.c_str());
                    }
                    continue;
                }
                long long id = v.num("id", -1);
                const json::Value* result = v.get("result");
                const json::Value* error = v.get("error");
                bool has_err = error && !error->is_null();
                std::string status = v.str("status");
                if (id == 1) {
                    if (has_err || (status != "ok" && status != "OK")) {
                        LOG("pool: login refused: %s", line.c_str());
                        lost = true;
                        backoff = 60;
                        break;
                    }
                    logged_in = true;
                    backoff = 2;
                    // claimed hash only for pools that announce it (protocol 2)
                    send_claimed = false;
                    const json::Value* caps = result ? result->get("capabilities") : nullptr;
                    if (caps && caps->is_arr())
                        for (auto& c : caps->arr) if (c.is_str() && c.s == "submit_claimed_hash") send_claimed = true;
                    LOG("pool: logged in as %s (worker %s), protocol %lld%s", cfg.user.c_str(), cfg.worker.c_str(),
                        result ? result->num("protocol_version", 1) : 1LL, send_claimed ? ", claimed hash" : "");
                    continue;
                }
                auto it = pending.find((int)id);
                if (it == pending.end()) { LOG("pool: unexpected reply: %.300s", line.c_str()); continue; }
                Pending pd = it->second;
                pending.erase(it);
                bool acc = false;
                if (!has_err) {
                    const json::Value* a = result ? result->get("accepted") : nullptr;
                    if (a && a->is_bool()) acc = a->b;
                    else acc = status == "ok" || status == "OK";
                }
                if (acc) {
                    g_accepted++;
                    if (pd.block) g_blocks++;
                    LOG("share accepted (GPU %d)%s nonce %llu hash %s  [%llu/%llu]", pd.gpu, pd.block ? " BLOCK" : "", (unsigned long long)pd.nonce,
                        pd.hash.substr(0, 16).c_str(), (unsigned long long)g_accepted.load(), (unsigned long long)(g_accepted.load() + g_rejected.load()));
                } else {
                    g_rejected++;
                    std::string etext = has_err ? json::dump(*error) : line;
                    LOG("share REJECTED (GPU %d) job %s nonce %llu: %s  [%llu/%llu]", pd.gpu, pd.job_id.c_str(), (unsigned long long)pd.nonce,
                        etext.c_str(), (unsigned long long)g_accepted.load(), (unsigned long long)(g_accepted.load() + g_rejected.load()));
                }
            }
            if (lost) break;
            if (logged_in) {
                std::deque<Share> q;
                {
                    std::lock_guard<std::mutex> lk(g_mtx);
                    q.swap(g_shares);
                }
                for (auto& s : q) {
                    int id = next_id++;
                    pending[id] = Pending{ s.gpu, s.job_id, s.hash_hex, s.block, s.nonce };
                    std::string msg = "{\"id\":" + std::to_string(id) + ",\"method\":\"submit\",\"params\":{\"job_id\":" + json::quote(s.job_id) +
                                      ",\"nonce\":" + std::to_string(s.nonce);
                    if (send_claimed) msg += ",\"claimed_hash\":" + json::quote(s.hash_hex);
                    msg += ",\"miner_version\":" + json::quote(std::string("bntminer/") + BNTMINER_VERSION) + "}}";
                    if (g_accepted.load() + g_rejected.load() < 2) LOG("pool -> %.400s", msg.c_str());
                    if (!conn.send_line(msg)) { LOG("pool: send failed: %s", conn.error().c_str()); lost = true; break; }
                }
            } else if (std::chrono::duration_cast<std::chrono::seconds>(Clock::now() - t_conn).count() > 60) {
                LOG("pool: no login answer within 60 s");
                lost = true;
            }
            if (lost) break;
            if (Clock::now() - t_rx > std::chrono::seconds(900)) { LOG("pool: nothing received for 900 s, reconnecting"); lost = true; }
        }
        conn.close();
        {
            std::lock_guard<std::mutex> lk(g_mtx);
            g_paused = true;
            g_job.reset();
            g_shares.clear();
        }
        for (auto& s : g_stats) s->abort = 1;
        if (!g_running) break;
        pool_idx++;
        LOG("pool: disconnected, retrying in %d s", backoff);
        for (int i = 0; i < backoff * 10 && g_running; i++) std::this_thread::sleep_for(std::chrono::milliseconds(100));
        backoff = backoff * 2 > 60 ? 60 : backoff * 2;
    }
}

// ---------------------------------------------------------------------------
// tests and benchmark
// ---------------------------------------------------------------------------
static std::vector<int> selected_devices(const Config& cfg)
{
    int n = bg_device_count();
    std::vector<int> d;
    if (cfg.gpus.empty()) { for (int i = 0; i < n; i++) d.push_back(i); }
    else { for (int g : cfg.gpus) if (g >= 0 && g < n) d.push_back(g); }
    return d;
}

static const size_t NV = sizeof bnt_vectors / sizeof bnt_vectors[0];

// GPU hashes of the vectors of memory size m (all at once, one lane each); returns failures
static int gpu_vectors(int dev, uint32_t m, int chunk, int reserve_mb, int& checked, double* ms_out)
{
    std::vector<const BntVector*> vs;
    for (size_t i = 0; i < NV; i++) if (bnt_vectors[i].m == m) vs.push_back(&bnt_vectors[i]);
    if (vs.empty()) return 0;
    void* g = bg_create(dev, m, (int)vs.size(), reserve_mb, chunk);
    if (!g) { LOG("TEST GPU %d m=%u: create failed: %s", dev, m, bg_last_error(nullptr)); return 1; }
    int L = bg_lanes(g);
    if (L < (int)vs.size()) LOG("TEST GPU %d m=%u: only %d lane(s) for %zu vectors", dev, m, L, vs.size());
    int fails = 0;
    for (size_t base = 0; base < vs.size(); base += L) {
        int n = (int)std::min<size_t>(L, vs.size() - base);
        std::vector<uint64_t> init((size_t)n * 256), last((size_t)n * 128);
        for (int l = 0; l < n; l++) {
            std::vector<uint8_t> h;
            hex_to_vec(vs[base + l]->header_hex, h);
            make_init(h, vs[base + l]->nonce, m, init.data() + (size_t)l * 256);
        }
        double ms = 0;
        if (bg_hash(g, init.data(), last.data(), n, nullptr, &ms) != 0) { LOG("TEST GPU %d m=%u: %s", dev, m, bg_last_error(g)); fails++; break; }
        if (ms_out) *ms_out = ms;
        for (int l = 0; l < n; l++) {
            uint8_t out[32], expect[32];
            bnt::final_hash(last.data() + (size_t)l * 128, out);
            hex_to_bytes(vs[base + l]->hash_hex, expect, 32);
            bool ok = !memcmp(out, expect, 32);
            if (!ok) {
                fails++;
                LOG("TEST GPU %d m=%u nonce %llu: WRONG hash %s (expected %s)", dev, m, (unsigned long long)vs[base + l]->nonce, to_hex(out, 32).c_str(),
                    vs[base + l]->hash_hex);
            }
            checked++;
        }
        if (m == BNT_M_BLOCKS)
            LOG("TEST GPU %d m=%u (real Blocknet PoW): %d hash(es) in parallel in %.0f ms = %.3f H/s", dev, m, n, ms, ms > 0 ? n * 1000.0 / ms : 0.0);
    }
    bg_destroy(g);
    return fails;
}

static int run_tests(const Config& cfg)
{
    int fails = 0;
    // 1) CPU reference: BLAKE2b known answers, then every small vector (Rust argon2 crate)
    {
        uint8_t o[64];
        bnt::blake2b(o, 64, "abc", 3);
        bool ok = to_hex(o, 64) ==
                  "ba80a53f981c4d0d6a2797b69f12f6e94c212f14685ac4b74b12bb6fdbffa2d17d87c5392aab792dc252d5de4533cc9518d38aa8dbf1925ab92386edd4009923";
        int bad = ok ? 0 : 1, n = 0;
        for (size_t i = 0; i < NV; i++) {
            const BntVector& v = bnt_vectors[i];
            if (v.m > 65536) continue;
            std::vector<uint8_t> h;
            hex_to_vec(v.header_hex, h);
            uint8_t out[32], expect[32];
            hex_to_bytes(v.hash_hex, expect, 32);
            if (!bnt::pow_hash_cpu(h.data(), h.size(), v.nonce, v.m, out) || memcmp(out, expect, 32)) bad++;
            n++;
        }
        LOG("TEST CPU: BLAKE2b known answer + %d Argon2id vectors (m = 8 .. 65536 KiB, Rust argon2 crate): %s", n, bad ? "FAILED" : "OK");
        fails += bad;
    }
    if (fails) { LOG("TEST RESULT: FAILED"); return 1; }

    // 2) GPU: small memory sizes with odd launch chunks (prefetch and chunk boundaries), then
    //    the real 2 GiB PoW with as many lanes as there are vectors
    std::vector<int> devs = selected_devices(cfg);
    if (devs.empty()) { LOG("TEST: no CUDA device"); LOG("TEST RESULT: FAILED"); return 1; }
    int total_fail = 0;
    for (int dev : devs) {
        int fail = 0, checked = 0;
        const uint32_t ms_small[] = { 8, 64, 1024, 65536 };
        const int chunks[] = { 1, 5, 333, 0 };
        for (uint32_t m : ms_small)
            for (int ch : chunks) fail += gpu_vectors(dev, m, ch, cfg.reserve_mb, checked, nullptr);
        LOG("TEST GPU %d: %d small-memory hashes (4 chunk sizes): %s", dev, checked, fail ? "FAILED" : "OK");
        int c2 = 0;
        double ms = 0;
        fail += gpu_vectors(dev, BNT_M_BLOCKS, cfg.chunk, cfg.reserve_mb, c2, &ms);
        checked += c2;
        LOG("TEST GPU %d: %s (%d hashes checked)", dev, fail ? "FAILED" : "OK", checked);
        total_fail += fail;
    }
    LOG("TEST RESULT: %s", total_fail ? "FAILED" : "ALL OK");
    return total_fail ? 1 : 0;
}

static int run_bench(const Config& cfg)
{
    std::vector<int> devs = selected_devices(cfg);
    if (devs.empty()) { LOG("BENCH: no CUDA device"); return 1; }
    auto j = std::make_shared<PoolJob>();
    hex_to_vec(bnt_vectors[NV - 1].header_hex, j->header);
    memset(j->target, 0, 32);                  // impossible share target
    memset(j->net_target, 0, 32);
    j->job_id = "bench";
    j->seq = 1;
    j->nonce_start = 1;
    j->nonce_end = UINT64_MAX - 1;
    j->next = 1;
    { std::lock_guard<std::mutex> lk(g_mtx); g_job = j; g_paused = false; g_job_seq = 1; }
    for (size_t i = 0; i < devs.size(); i++) { g_stats.emplace_back(new GpuStat()); g_stats.back()->dev = devs[i]; }
    std::vector<std::thread> th;
    for (size_t i = 0; i < devs.size(); i++) th.emplace_back(gpu_worker, std::cref(cfg), (int)i, devs[i]);
    // warm-up: one run per GPU
    for (int i = 0; i < 1200 && g_running; i++) {
        bool all = true;
        for (auto& s : g_stats) if (s->hashes < 1) all = false;
        if (all) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    std::vector<uint64_t> h0;
    for (auto& s : g_stats) h0.push_back(s->hashes);
    auto t0 = Clock::now();
    for (int i = 0; i < cfg.bench_seconds && g_running; i++) std::this_thread::sleep_for(std::chrono::seconds(1));
    double secs = std::chrono::duration<double>(Clock::now() - t0).count();
    double total = 0;
    for (size_t i = 0; i < g_stats.size(); i++) {
        double hs = (double)(g_stats[i]->hashes - h0[i]) / secs;
        total += hs;
        int L = g_stats[i]->lanes;
        int rm = g_stats[i]->run_ms;
        LOG("BENCH GPU %d: %.3f H/s (%d lanes, %d ms per run = %.3f H/s from the run time) %s", g_stats[i]->dev, hs, L, rm,
            rm > 0 ? L * 1000.0 / rm : 0.0, g_stats[i]->name.c_str());
    }
    LOG("BENCH TOTAL: %.3f H/s over %zu GPU(s), %.0f s", total, g_stats.size(), secs);
    g_running = false;
    for (auto& t : th) t.join();
    return 0;
}

static int run_profile(const Config& cfg)
{
    std::vector<int> devs = selected_devices(cfg);
    if (devs.empty()) { LOG("PROFILE: no CUDA device"); return 1; }
    for (int dev : devs) {
        void* g = bg_create(dev, BNT_M_BLOCKS, cfg.lanes, cfg.reserve_mb, cfg.chunk);
        if (!g) { LOG("PROFILE GPU %d: %s", dev, bg_last_error(nullptr)); return 1; }
        int L = bg_lanes(g);
        for (int n = 1; n <= L; n++) {
            double a = 0, b = 0;
            if (bg_profile(g, n, &a, &b) != 0) { LOG("PROFILE GPU %d: %s", dev, bg_last_error(g)); bg_destroy(g); return 1; }
            double per = (a + b) / (BNT_M_BLOCKS - 2) * 1e6;
            LOG("PROFILE GPU %d, %d lane(s): Argon2i half %.0f ms, Argon2d half %.0f ms, %.0f ns per block, %.3f H/s", dev, n, a, b, per,
                n * 1000.0 / (a + b));
        }
        bg_destroy(g);
    }
    return 0;
}

static std::vector<std::string> expand_args(int argc, char** argv)
{
    std::vector<std::string> out;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--args-file" && i + 1 < argc) {
            FILE* f = fopen(argv[++i], "r");
            if (!f) { fprintf(stderr, "cannot open args file %s\n", argv[i]); continue; }
            char tok[1024];
            while (fscanf(f, "%1023s", tok) == 1) out.push_back(tok);
            fclose(f);
            continue;
        }
        out.push_back(a);
    }
    return out;
}

int main(int argc, char** argv)
{
    Config cfg;
    cfg.exe_dir = exe_directory(argv[0]);
    std::vector<std::string> args = expand_args(argc, argv);
    std::string conf = cfg.exe_dir + "/bntminer.conf";
    if (!file_exists(conf)) conf = cfg.exe_dir + "/../bntminer.conf";
    for (size_t i = 0; i < args.size(); i++)
        if ((args[i] == "--config" || args[i] == "-c") && i + 1 < args.size()) conf = args[++i];
    bool have_conf = load_config_file(cfg, conf);
    std::vector<std::string> cli_pools;
    for (size_t i = 0; i < args.size(); i++) {
        const std::string& a = args[i];
        if (a == "--config" || a == "-c") { i++; continue; }
        if (a == "--test") { cfg.test = true; continue; }
        if (a == "--profile") { cfg.profile = true; continue; }
        if (a == "--bench") {
            cfg.bench = true;
            if (i + 1 < args.size() && args[i + 1][0] != '-') cfg.bench_seconds = atoi(args[++i].c_str());
            continue;
        }
        if (a == "--help" || a == "-h") {
            printf("BntMiner %s - Blocknet (BNT) Argon2id GPU miner, no dev fee\n"
                   "  bntminer [--config file] [--args-file file] [-o url] [-u address] [--worker rig]\n"
                   "           [--gpus 0,1|all] [--lanes N] [--reserve-mb N] [--chunk N] [--cafile f] [--stats s] [--log file]\n"
                   "           [--test] [--bench [s]] [--profile]\n",
                   BNTMINER_VERSION);
            return 0;
        }
        if (i + 1 < args.size()) {
            if (a == "-o" || a == "--pool" || a == "--url") { cli_pools.push_back(args[++i]); continue; }
            if (apply_option(cfg, a, args[i + 1])) { i++; continue; }
        }
        fprintf(stderr, "unknown argument %s (see --help)\n", a.c_str());
        return 2;
    }
    if (!cli_pools.empty()) cfg.pools = cli_pools;
    if (cfg.cafile.empty()) {
        const char* cands[] = { "C:\\msys64\\mingw64\\etc\\ssl\\certs\\ca-bundle.crt", "/etc/ssl/certs/ca-certificates.crt" };
        std::string local = cfg.exe_dir + "/ca-bundle.crt";
        if (file_exists(local)) cfg.cafile = local;
        else for (auto c : cands) if (file_exists(c)) { cfg.cafile = c; break; }
    }
    if (!cfg.logfile.empty()) g_logf = fopen(cfg.logfile.c_str(), "a");
    if (cfg.stats < 10) cfg.stats = 10;
    net_global_init();

    signal(SIGINT, on_signal);
    signal(SIGTERM, on_signal);
    LOG("BntMiner %s (Blocknet BNT, Argon2id 2 GiB t=1 p=1) - no dev fee%s", BNTMINER_VERSION, have_conf ? "" : " - no config file");

    if (cfg.test) { deadline_watchdog(900, "TEST"); return run_tests(cfg); }
    if (cfg.profile) { deadline_watchdog(600, "PROFILE"); return run_profile(cfg); }
    if (cfg.bench) { deadline_watchdog(cfg.bench_seconds + 300, "BENCH"); return run_bench(cfg); }

    if (cfg.pools.empty() || cfg.user.empty()) { LOG("missing pool url or user (Blocknet address), see bntminer.conf"); return 2; }
    std::vector<int> devs = selected_devices(cfg);
    if (devs.empty()) { LOG("no CUDA device selected"); return 1; }

    for (size_t i = 0; i < devs.size(); i++) { g_stats.emplace_back(new GpuStat()); g_stats.back()->dev = devs[i]; }
    std::vector<std::thread> th;
    for (size_t i = 0; i < devs.size(); i++) th.emplace_back(gpu_worker, std::cref(cfg), (int)i, devs[i]);
    std::thread net(pool_worker, std::cref(cfg));

    std::vector<uint64_t> last(g_stats.size(), 0);
    auto t_last = Clock::now();
    while (g_running) {
        for (int i = 0; i < cfg.stats * 10 && g_running; i++) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            if (i % 10 != 9) continue;
            long long t = now_ms();
            for (size_t k = 0; k < g_stats.size(); k++) {
                if (!g_stats[k]->alive) continue;
                long long idle = t - g_stats[k]->beat_ms.load();
                if (idle > BM_STUCK_MS) {
                    LOG("GPU %d stuck for %lld s (run never returned, unstable clocks?): exiting with code 4 for an automatic restart",
                        g_stats[k]->dev, idle / 1000);
                    fflush(nullptr);
                    std::_Exit(4);
                }
            }
        }
        if (!g_running) break;
        double secs = std::chrono::duration<double>(Clock::now() - t_last).count();
        t_last = Clock::now();
        double total = 0;
        std::string per;
        for (size_t i = 0; i < g_stats.size(); i++) {
            uint64_t s = g_stats[i]->hashes;
            double r = (double)(s - last[i]) / secs;
            last[i] = s;
            total += r;
            char b[64];
            snprintf(b, sizeof b, "%sGPU%d %.3f", i ? " | " : "", g_stats[i]->dev, r);
            per += b;
        }
        bool paused;
        { std::lock_guard<std::mutex> lk(g_mtx); paused = g_paused; }
        LOG("%.3f H/s total (%s) | shares %llu/%llu | blocks %llu%s", total, per.c_str(), (unsigned long long)g_accepted.load(),
            (unsigned long long)(g_accepted.load() + g_rejected.load()), (unsigned long long)g_blocks.load(), paused ? " | waiting for work" : "");
    }
    if (g_gpu_failed) {
        LOG("GPU %d failed (CUDA error): exiting with code 3 for an automatic restart", g_gpu_failed.load() - 1);
        fflush(nullptr);
        std::_Exit(3);
    }
    LOG("stopping...");
    for (auto& t : th) t.join();
    net.join();
    return 0;
}
