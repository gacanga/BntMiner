# BntMiner — mineur GPU NVIDIA pour Blocknet (BNT), Argon2id 2 Gio

BntMiner mine [Blocknet](https://blocknetcrypto.com/) (BNT, la crypto confidentielle de `blocknetprivacy`) sur des
cartes NVIDIA. La preuve de travail est **Argon2id v1.3 avec 2 Gio de mémoire, 1 passe, 1 voie**, et le mineur
utilise le stratum de [bntpool.com](https://bntpool.com/). Le moteur CUDA a été écrit de zéro pour les cartes Pascal
(P104-100, GTX 1070/1080, sm_61).

- **Aucun frais de développeur.**
- Open source, Apache-2.0.
- Version Windows x64 compilée avec CUDA 12.9 (code sm_61 + PTX : les GPU plus récents l'exécutent via le JIT du
  pilote).
- Vérifié contre la crate Rust `argon2` appelée exactement comme le cœur de Blocknet, à plusieurs tailles de mémoire
  et aux vrais paramètres de 2 Gio, sur le CPU et sur chaque GPU (`test.bat`).

À ne pas confondre avec l'ancien projet Blocknet (BLOCK) ni avec Bancor (BNT).

English version: [README.md](README.md).

## Démarrage rapide (Windows)

1. Obtenez une adresse Blocknet avec le wallet officiel
   ([blocknet-gui](https://github.com/blocknetprivacy/blocknet-gui/releases/latest) ou la
   [version en ligne de commande](https://github.com/blocknetprivacy/blocknet/releases/latest)). Le nœud n'a pas besoin
   d'être synchronisé pour miner en pool.
2. Téléchargez le zip dans [Releases](https://github.com/gacanga/BntMiner/releases) et décompressez-le.
3. Ouvrez `bntminer.conf` et remplacez `YOUR_BLOCKNET_ADDRESS` par votre adresse (un nom court blocknet.id comme
   `@nom` fonctionne aussi) ; `rig1` est le nom du worker, modifiable.
4. Double-cliquez sur `start_bnt.bat`. Il relance le mineur 10 s après chaque arrêt et écrit `bntminer.log` ; fermez la
   fenêtre pour l'arrêter.

`test.bat` lance les tests de calcul sur chaque GPU, `bench.bat` une mesure des H/s pendant 120 s (arrêtez le minage
avant).

Prérequis : Windows 10/11 x64, un pilote NVIDIA compatible CUDA 12 (R525 ou plus récent ; R580 est la dernière branche
pour Pascal), **2 Gio de mémoire GPU libre par « lane »** (une carte de 8 Go en fait 3). Rien d'autre à installer.

## Pool

| Pool | URL | Frais |
| --- | --- | --- |
| bntpool.com | `stratum+tcp://bntpool.com:3333` | 2,5 %, PPLNS (fenêtre de 6 h), paiement minimum 0,1 BNT, 60 confirmations |

Protocole (lignes JSON, celui du mineur de référence [seine](https://github.com/zpalmtree/seine) ; vérifié sur la
pool réelle) :

```
-> {"id":1,"method":"login","params":{"address","worker","protocol_version":2,"capabilities":[...]}}
<- {"id":1,"status":"ok","result":{"protocol_version":2,"capabilities":[...],"required_capabilities":["submit_claimed_hash"]}}
<- {"method":"job","params":{"job_id","header_base","target","difficulty","network_target","height","nonce_start","nonce_end"}}
-> {"id":n,"method":"submit","params":{"job_id","nonce","claimed_hash","miner_version"}}
<- {"id":n,"status":"ok","result":{"accepted":true,"difficulty":d,"status":"provisional","verified":false}}
```

`header_base` est l'en-tête de bloc de 100 octets sans le nonce ; la pool donne à chaque job sa propre plage de
nonces. Une share est valable si `Argon2id(password = nonce en u64 little-endian, salt = header_base) <= target`
(octets big-endian). La pool vérifie les shares après coup (« provisional », puis « verified »).
Vos shares et votre solde : `https://bntpool.com/api/miner/<adresse>` et `.../balance?include_pending_estimate=true`.

## Configuration

Le mineur lit `bntminer.conf` à côté de l'exécutable (ou un dossier au-dessus), ou le fichier donné par `--config`.
Chaque ligne est de la forme `clé = valeur` ; `#` et `;` commencent un commentaire. Chaque clé est aussi une option de
ligne de commande (`--clé valeur`) ; la ligne de commande l'emporte sur le fichier, et `-o` en ligne de commande
remplace les pools du fichier.

| Option / clé | Défaut | Rôle |
| --- | --- | --- |
| `-o`, `--pool`, `--url` / `pool` | — | URL de la pool, répétable (ordre de secours). `stratum+tcp://` = TCP simple, `stratum+ssl://` ou `stratum+tls://` = TLS |
| `-u`, `--user`, `--wallet`, `--address` / `user` | — | Adresse Blocknet ou nom court blocknet.id |
| `--worker`, `--rig` / `worker` | `rig` | Nom du worker |
| `--gpus`, `--devices`, `-d` / `gpus` | `all` | `all` ou une liste comme `0,1,2` (ordre du bus PCI, comme nvidia-smi) |
| `--lanes` / `lanes` | `0` | Hashs calculés en parallèle par GPU (2 Gio chacun) ; `0` = autant que la mémoire libre le permet |
| `--reserve-mb` / `reserve-mb` | `160` | Mémoire GPU laissée libre (Mo) |
| `--chunk` / `chunk` | `32768` | Blocs de 1 Kio par lancement de kernel (le calcul peut s'arrêter entre deux lancements) |
| `--stats` / `stats` | `60` | Secondes entre deux lignes de résumé (minimum 10) |
| `--log` / `log` | — | Ajoute le journal à ce fichier |
| `--cafile` / `cafile` | `ca-bundle.crt` à côté de l'exe | Certificats pour les pools TLS |
| `--config`, `-c` | `bntminer.conf` | Fichier de configuration |
| `--args-file FICHIER` | — | Lit d'autres arguments dans un fichier (séparés par des espaces) |
| `--test` | | Tests de calcul (CPU, puis chaque GPU choisi), code de sortie 0 si tout passe |
| `--bench [s]` | 120 s | Mesure des H/s sur les GPU choisis |
| `--profile` | | Durée des moitiés Argon2i et Argon2d d'un hash, pour 1 à N lanes |
| `--help`, `-h` | | Aide |

Codes de sortie utilisés par la boucle de relance : 2 = argument incorrect ou pool/adresse manquante, 3 = erreur CUDA,
4 = un calcul GPU n'est pas revenu en 60 s (fréquences instables ?), 5 = délai dépassé pendant un test ou une mesure.

## Performances (mesurées)

NVIDIA P104-100 8 Go, bntpool :

| Configuration | H/s |
| --- | --- |
| 1 × P104-100, 1 lane | ≈ 0,48 |
| 1 × P104-100, 3 lanes | ≈ 1,5 (benchmark 1,50, pool 1,48–1,65) |
| 2 × P104-100 | ≈ 3,2 |

Une P104-100 ne consomme qu'environ 66 W sur cet algorithme. Avec une seule lane (une chaîne Argon2) par warp, la
vitesse dépend de la latence de la fonction de compression : elle augmente donc presque linéairement avec le nombre de
lanes, limité par la mémoire du GPU (2 Gio chacune).

## Fonctionnement

- PoW (cœur Blocknet, `crypto-rs/src/pow.rs`) : Argon2id v0x13, m = 2 097 152 Kio (2^21 blocs de 1 Kio), t = 1,
  p = 1, sortie de 32 octets. Avec une seule voie et une seule passe, les blocs 0 .. m/2-1 sont en adressage Argon2i
  (indices indépendants des données : table fixe de 4 Mo calculée une fois sur l'hôte) et les blocs m/2 .. m-1 en
  adressage Argon2d (J1 = 32 bits bas du bloc précédent). Index de référence du bloc i : area = i − 1,
  x = J1²>>32, ref = area − 1 − (area·x>>32).
- L'hôte calcule H0 et les deux premiers blocs (BLAKE2b, H'), le GPU remplit les 2^21 blocs, l'hôte calcule le résultat
  final H'(32, dernier bloc) et le compare à la cible.
- Moteur CUDA (`gpu/bnt_gpu.cu`) : un warp de 32 threads par lane. Chaque thread garde 4 des 128 mots du bloc courant
  dans ses registres (mots 2t, 2t+1, 64+2t, 65+2t, lus et écrits en deux accès coalescés de 16 octets). La permutation
  BlaMka travaille en mémoire partagée paddée (étape des lignes, puis des colonnes, chaque thread calculant un des 32
  quarts de tour, diagonales par shuffles de warp). Dans la moitié Argon2i, le bloc de référence de i+1 est chargé
  pendant le calcul du bloc i. Les lancements de 32 768 blocs évitent le chien de garde de l'affichage Windows et
  permettent d'arrêter le calcul plus tôt quand un nouveau bloc arrive.
- `src/argon_ref.h` : référence en en-tête seul (BLAKE2b, H', H0, G, table d'adresses, hash CPU complet), partagée
  par la DLL, le mineur et les tests.

## Compilation (Windows)

Prérequis attendus par les scripts :

- CUDA Toolkit **12.9** (`CUDA_PATH_V12_9`, par défaut `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9`) ;
  CUDA 13 ne sait plus compiler pour Pascal.
- Visual Studio 2022 (ou les Build Tools) avec les outils C++ x64 (trouvés par `vswhere`).
- MSYS2 dans `C:\msys64` avec les paquets MINGW64 `mingw-w64-x86_64-gcc` et `mingw-w64-x86_64-openssl`.

| Script | Rôle |
| --- | --- |
| `build_all.bat` | Compile `dev\bgpu.dll` (nvcc + MSVC, `windows\build_gpu_dll.bat`) et `dev\bntminer.exe` (g++ MSYS2, `windows/build_host.sh`), puis lance les tests. `build_all.bat nobuild` = tests seuls |
| `run_test.bat` | `dev\bntminer.exe --test`, journal `logs\test.log` ; arguments supplémentaires dans `run\test_args.txt` |
| `run_bench.bat` | Mesure de 120 s de `dev\`, journal `logs\bench.log` ; `run_bench.bat profile` = moitiés Argon2i / Argon2d (`logs\profile.log`) ; arguments dans `run\bench_args.txt` |
| `deploy.bat` | Copie `dev\` dans `bin\`, seulement si le dernier `logs\test.log` indique `TEST RESULT: ALL OK` |
| `bnt_pool.bat` | Mine avec `bin\bntminer.exe` et `bntminer.conf`, boucle de relance, journal `logs\pool.log` ; arguments dans `run\pool_args.txt` |
| `stop_bnt.bat` | Arrête la boucle de relance et `bntminer.exe` |

Options de compilation (variables d'environnement) : `BM_NVCC_EXTRA` (ou `run\nvcc_extra.txt`) ajoute des options à
nvcc, `BM_HOST_ARCH` remplace `-march=native`. Pour une version portable, faites
`set BM_HOST_ARCH=-march=x86-64 -mtune=generic` avant `build_all.bat`. `bgpu.dll` utilise le runtime C Microsoft
dynamique (`/MD`) : le zip fournit ses DLL. Le zip contient `bntminer.exe`, `bgpu.dll`, `cudart64_12.dll`, les DLL du
runtime VC++, `ca-bundle.crt` et les fichiers de [`dist/`](dist).

## Tests

`bntminer.exe --test` :

1. CPU : un résultat connu de BLAKE2b, puis 16 vecteurs Argon2id (m = 8, 64, 1 024 et 65 536 Kio, 4 nonces chacun) de
   `src/bnt_vectors.h` ;
2. chaque GPU choisi : les mêmes vecteurs avec 4 tailles de lancement (1, 5, 333 blocs et la valeur par défaut), ce qui
   couvre le préchargement et les frontières entre lancements, puis 3 vecteurs de la vraie PoW de 2 Gio calculés en
   parallèle (une lane chacun).

Les vecteurs viennent de `tools/bntref` (Rust, `argon2 = "0.5"`, même appel que le cœur Blocknet) :

```sh
cd tools/bntref && cargo build --release
for m in 8 64 1024 65536; do ./target/release/bntref $m 4; done > ../../tests/vectors.txt
./target/release/bntref 2097152 3 >> ../../tests/vectors.txt
cd ../.. && python3 tools/gen_vectors.py      # -> src/bnt_vectors.h
```

Contrôles sous Linux, sans GPU :

```sh
g++ -O2 -std=c++17 -Isrc tests/cloud_test.cpp -o cloud_test && ./cloud_test tests/vectors.txt full
g++ -O2 -std=gnu++17 -Isrc -Igpu -o bntminer_stub src/main.cpp src/tls_conn.cpp tests/bg_stub.cpp -lssl -lcrypto -pthread
./bntminer_stub --test
python3 tests/fake_pool.py &  ./bntminer_stub -o stratum+tcp://127.0.0.1:13333 -u TEST --lanes 3
```

`tests/cloud_test.cpp` vérifie le code de référence et une émulation CPU du découpage du kernel entre threads
(disposition, mémoire partagée paddée, shuffles) contre les vecteurs ; `tests/bg_stub.cpp` remplace `bgpu.dll` par le
CPU ; `tests/fake_pool.py` est un serveur minimal qui imite bntpool. `tools/nvrtc_check.py` compile le kernel pour
sm_61 avec NVRTC et affiche le rapport de ptxas (registres, débordements).

## Crédits et licence

- Moteur CUDA, code de référence, programme hôte et client stratum : gacanga. Le client stratum (`src/json.h`,
  `src/tls_conn.*`) vient de FiloMiner / NullMiner (Apache-2.0).
- Le protocole de pool suit [seine](https://github.com/zpalmtree/seine) (BSD-3-Clause) ; aucun code de seine n'est
  inclus.
- La crate Rust [argon2](https://github.com/RustCrypto/password-hashes) et le
  [cœur Blocknet](https://github.com/blocknetprivacy/core) ont servi de références pour les tests uniquement ; ils ne
  sont pas liés au mineur.

Licence Apache 2.0, voir [LICENSE](LICENSE) et [NOTICE](NOTICE). Aucune garantie : vous minez à vos risques, et le
minage consomme de l'électricité. Blocknet se décrit comme expérimental ; une mise à jour du réseau est prévue vers le
16/02/2027 avec une autre preuve de travail, après quoi ce mineur devra être mis à jour.
