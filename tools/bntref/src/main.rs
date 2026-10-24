// Reference vectors for Blocknet PoW: Argon2id v0x13, t=1, p=1, out 32,
// password = nonce (u64 LE), salt = header (without nonce). Same calls as blocknet core crypto-rs/src/pow.rs.
use argon2::{Algorithm, Argon2, Params, Version};
fn hex(b: &[u8]) -> String { b.iter().map(|x| format!("{:02x}", x)).collect() }
fn main() {
    let args: Vec<String> = std::env::args().collect();
    let m_kb: u32 = args.get(1).map(|s| s.parse().unwrap()).unwrap_or(2 * 1024 * 1024);
    let count: u64 = args.get(2).map(|s| s.parse().unwrap()).unwrap_or(4);
    let params = Params::new(m_kb, 1, 1, Some(32)).unwrap();
    let a = Argon2::new(Algorithm::Argon2id, Version::V0x13, params);
    // deterministic pseudo-random headers of 100 bytes
    let mut seed: u64 = 0x426c6f636b6e6574 ^ (m_kb as u64);
    for k in 0..count {
        let mut header = [0u8; 100];
        for b in header.iter_mut() { seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); *b = (seed >> 56) as u8; }
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let nonce: u64 = if k == 0 { 0 } else { seed };
        let mut out = [0u8; 32];
        let t = std::time::Instant::now();
        a.hash_password_into(&nonce.to_le_bytes(), &header, &mut out).unwrap();
        eprintln!("m={} k={} {:.2}s", m_kb, k, t.elapsed().as_secs_f64());
        println!("{} {} {} {}", m_kb, hex(&header), nonce, hex(&out));
    }
}
