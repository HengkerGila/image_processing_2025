# capsa_engine.py
# Engine logic Capsa 2-pemain (tanpa UI & tanpa kamera)

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum, auto
from collections import Counter
from itertools import combinations
import random
from typing import List, Tuple, Optional


# ===================== REPRESENTASI KARTU =====================

RANKS = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"]
SUITS = ["club", "diamond", "heart", "spade"]

# "2" -> 2, "A" -> 14
RANK_VALUE = {r: i + 2 for i, r in enumerate(RANKS)}

# Deck standar 52 kartu: "7_club", "A_spade", dst.
DECK = [f"{r}_{s}" for r in RANKS for s in SUITS]


# ===================== TIPE COMBO =====================

class ComboType(Enum):
    HIGH_CARD      = auto()
    PAIR           = auto()
    DOUBLE_PAIR    = auto()
    STRAIGHT_FLUSH = auto()
    ROYAL_FLUSH    = auto()


@dataclass
class Combo:
    """
    type : jenis kombinasi (HIGH_CARD, PAIR, ...)
    cards: list label kartu, contoh ["4_spade", "4_heart"]
    key  : tuple ranking untuk membandingkan combo sejenis
           (misal (6,) untuk pair 6, (9,) untuk high card 9, dst.)
    """
    type: ComboType
    cards: List[str]
    key: Tuple[int, ...]


@dataclass
class GameState:
    """
    State utama game untuk 2 pemain.
    """
    player_hand: List[str]
    bot_hand: List[str]
    current_player: str                # "player" atau "bot"
    required_type: Optional[ComboType] # jenis combo di ronde ini (None kalau ronde baru)
    best_combo: Optional[Combo]        # combo tertinggi di meja saat ini
    last_winner: str                   # "player" atau "bot" (pemenang ronde terakhir)
    passes_in_row: int                 # jumlah pass berturut-turut (dengan 2 pemain: 0/1)


# ===================== FUNGSI BANTU DASAR =====================

def parse_card(label: str) -> Tuple[int, str]:
    """
    '10_heart' -> (10, 'heart')  (10 di-mapping ke 10..14 sesuai RANK_VALUE)
    """
    rank_str, suit = label.split("_")
    return RANK_VALUE[rank_str], suit


def detect_combo(cards: List[str]) -> Optional[Combo]:
    """
    Mendeteksi kombinasi dari list kartu.
    Hanya mendukung:
      - High Card      (1 kartu)
      - Pair           (2 kartu)
      - Double Pair    (4 kartu)
      - Straight Flush (5 kartu)
      - Royal Flush    (5 kartu: 10..A satu suit)

    Return:
      Combo(...)  → jika kombinasi valid
      None        → jika tidak cocok / tidak valid
    """
    n = len(cards)
    if n == 0:
        return None

    ranks: List[int] = []
    suits: List[str] = []
    for c in cards:
        r, s = parse_card(c)
        ranks.append(r)
        suits.append(s)

    ranks_sorted = sorted(ranks)
    cnt = Counter(ranks)

    # 1. High card (1 kartu)
    if n == 1:
        return Combo(
            type=ComboType.HIGH_CARD,
            cards=cards[:],
            key=(ranks_sorted[-1],)     # nilai kartunya
        )

    # 2. Pair (2 kartu rank sama)
    if n == 2 and len(cnt) == 1:
        pair_rank = ranks_sorted[0]
        return Combo(
            type=ComboType.PAIR,
            cards=cards[:],
            key=(pair_rank,)
        )

    # 3. Double Pair (4 kartu = 2 pasang)
    if n == 4 and sorted(cnt.values()) == [2, 2]:
        pair_ranks = sorted([r for r, c in cnt.items() if c == 2], reverse=True)
        # Ranking: pair terbesar dulu, lalu yang kecil
        return Combo(
            type=ComboType.DOUBLE_PAIR,
            cards=cards[:],
            key=tuple(pair_ranks)
        )

    # 4. Straight Flush / Royal Flush (5 kartu)
    if n == 5:
        is_flush = (len(set(suits)) == 1)
        uniq = sorted(set(ranks))
        is_straight = (len(uniq) == 5 and uniq[-1] - uniq[0] == 4)

        if is_flush and is_straight:
            # Royal flush (10 J Q K A)
            if uniq == [10, 11, 12, 13, 14]:
                return Combo(
                    type=ComboType.ROYAL_FLUSH,
                    cards=cards[:],
                    key=(14,)   # As tinggi
                )
            # Straight flush biasa: ranking dengan high-card
            return Combo(
                type=ComboType.STRAIGHT_FLUSH,
                cards=cards[:],
                key=(max(uniq),)
            )

    # Tidak cocok kombinasi yang diijinkan
    return None


def compare_same_type(a: Combo, b: Combo) -> int:
    """
    Membandingkan dua combo dengan type yang sama.
    Return:
      +1  jika a > b
       0  jika a == b
      -1  jika a < b
    """
    assert a.type == b.type, "compare_same_type hanya untuk combo dengan type yang sama"

    if a.key > b.key:
        return 1
    elif a.key < b.key:
        return -1
    else:
        return 0


# ===================== ENGINE GAME 2 PEMAIN =====================

def deal_new_game() -> GameState:
    """
    Mengocok deck 52 kartu dan membagi:
      - 26 kartu untuk player
      - 26 kartu untuk bot
    Player selalu mulai dulu.
    """
    deck = DECK[:]
    random.shuffle(deck)

    player_hand = sorted(deck[:26])
    bot_hand    = sorted(deck[26:])

    return GameState(
        player_hand=player_hand,
        bot_hand=bot_hand,
        current_player="player",
        required_type=None,
        best_combo=None,
        last_winner="player",
        passes_in_row=0,
    )


def find_all_combos_of_type(hand: List[str], combo_type: ComboType) -> List[Combo]:
    """
    Generate semua combo dari sebuah hand dengan type tertentu.
    (Bruteforce kombinasi, cukup untuk 26 kartu 2 pemain.)
    """
    combos: List[Combo] = []

    if combo_type == ComboType.HIGH_CARD:
        for c in hand:
            combos.append(detect_combo([c]))  # pasti valid

    elif combo_type == ComboType.PAIR:
        for subset in combinations(hand, 2):
            combo = detect_combo(list(subset))
            if combo and combo.type == ComboType.PAIR:
                combos.append(combo)

    elif combo_type == ComboType.DOUBLE_PAIR:
        for subset in combinations(hand, 4):
            combo = detect_combo(list(subset))
            if combo and combo.type == ComboType.DOUBLE_PAIR:
                combos.append(combo)

    elif combo_type in (ComboType.STRAIGHT_FLUSH, ComboType.ROYAL_FLUSH):
        for subset in combinations(hand, 5):
            combo = detect_combo(list(subset))
            if combo and combo.type == combo_type:
                combos.append(combo)

    # Hilangkan None kalau ada
    combos = [c for c in combos if c is not None]
    return combos


# ------------------ TURN PLAYER ------------------

def try_play_player(state: GameState, chosen_cards: List[str]) -> Tuple[bool, str, Optional[Combo]]:
    """
    Player mencoba mengeluarkan 'chosen_cards'.
    - Cek apakah kartunya ada di tangan.
    - Cek kombinasinya valid.
    - Cek sesuai rules ronde (required_type & harus lebih tinggi dari best_combo).

    Return:
      (True,  "pesan OK", combo)  jika langkah valid
      (False, "alasan gagal", None) jika tidak sah
    """
    if not chosen_cards:
        return False, "Tidak ada kartu yang dipilih.", None

    # 1. cek semua kartu ada di tangan player
    for c in chosen_cards:
        if c not in state.player_hand:
            return False, "Kartu yang dipilih tidak ada di tangan pemain.", None

    combo = detect_combo(chosen_cards)
    if combo is None:
        return False, "Kombinasi tidak valid.", None

    # 2. kalau ronde baru, bebas pilih jenis
    if state.required_type is None:
        state.required_type = combo.type
        state.best_combo = combo
        state.last_winner = "player"
        state.passes_in_row = 0
    else:
        # harus type yang sama
        if combo.type != state.required_type:
            return False, "Jenis kombinasi harus sama dengan ronde ini.", None

        # harus lebih besar daripada best_combo
        if state.best_combo is not None:
            if compare_same_type(combo, state.best_combo) <= 0:
                return False, "Kombinasi masih lebih kecil atau sama.", None

        state.best_combo = combo
        state.last_winner = "player"
        state.passes_in_row = 0

    # 3. buang kartu dari tangan player
    for c in chosen_cards:
        state.player_hand.remove(c)

    # 4. ganti giliran ke bot
    state.current_player = "bot"
    return True, "Langkah pemain valid.", combo


def player_pass(state: GameState) -> str:
    """
    Player memilih PASS. Dengan 2 pemain, 1 pass sudah mengakhiri ronde.
    Pemenang ronde = last_winner, dan dia akan memulai ronde baru.
    """
    state.passes_in_row += 1

    # Karena hanya 2 pemain, 1 pass sudah cukup untuk mengakhiri ronde
    if state.passes_in_row >= 1:
        # ronde baru
        state.current_player = state.last_winner
        state.required_type = None
        state.best_combo = None
        state.passes_in_row = 0
        return "Player PASS. Ronde selesai, dimulai lagi oleh pemenang terakhir."

    # (secara teori tidak akan sampai sini di mode 2 pemain)
    state.current_player = "bot"
    return "Player PASS."


# ------------------ TURN BOT ------------------

def bot_pass(state: GameState) -> str:
    """
    Bot PASS. Dengan 2 pemain, 1 pass mengakhiri ronde.
    """
    state.passes_in_row += 1

    if state.passes_in_row >= 1:
        # ronde baru
        state.current_player = state.last_winner
        state.required_type = None
        state.best_combo = None
        state.passes_in_row = 0
        return "BOT PASS. Ronde selesai."
    else:
        state.current_player = "player"
        return "BOT PASS."


def bot_turn(state: GameState) -> Tuple[Optional[Combo], str]:
    """
    Logika sederhana untuk giliran bot:
      - Jika ronde baru: bot bebas memilih kombinasi (cari semua kemungkinan,
        lalu pilih yang paling lemah).
      - Jika ronde berjalan: bot hanya boleh type yang sama,
        dan harus mengalahkan best_combo dengan combo terkecil yang menang.
      - Kalau tidak punya kombinasi yang bisa menang → PASS.

    Return:
      (combo, "pesan")   jika bot main
      (None,  "BOT PASS ...") jika pass
    """
    hand = state.bot_hand

    # --- RONDE BARU: bot boleh pilih type apapun ---
    if state.required_type is None:
        all_types = [
            ComboType.HIGH_CARD,
            ComboType.PAIR,
            ComboType.DOUBLE_PAIR,
            ComboType.STRAIGHT_FLUSH,
            ComboType.ROYAL_FLUSH,
        ]
        candidate_combos: List[Combo] = []
        for t in all_types:
            candidate_combos.extend(find_all_combos_of_type(hand, t))

        if not candidate_combos:
            msg = bot_pass(state)
            return None, msg

        # Pilih combo "paling lemah" (type paling bawah + key terkecil)
        candidate_combos.sort(key=lambda c: (c.type.value, c.key))
        chosen = candidate_combos[0]

    # --- RONDE BERJALAN: harus type tertentu & mengalahkan best_combo ---
    else:
        candidate_combos = find_all_combos_of_type(hand, state.required_type)

        if state.best_combo is not None:
            candidate_combos = [
                c for c in candidate_combos
                if compare_same_type(c, state.best_combo) > 0
            ]

        if not candidate_combos:
            msg = bot_pass(state)
            return None, msg

        # Pilih combo terkecil yang masih menang
        candidate_combos.sort(key=lambda c: c.key)
        chosen = candidate_combos[0]

    # --- Mainkan chosen ---
    for c in chosen.cards:
        state.bot_hand.remove(c)

    state.best_combo = chosen
    state.required_type = chosen.type
    state.last_winner = "bot"
    state.passes_in_row = 0
    state.current_player = "player"

    return chosen, "BOT PLAY"
