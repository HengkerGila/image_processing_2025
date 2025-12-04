import cv2
import numpy as np
import os
from keras.models import load_model
import pygame
import random
from capsa_engine import (
    deal_new_game, GameState, try_play_player, player_pass,
    bot_turn, Combo, ComboType, detect_combo
)

def save_cards_to_txt(cards, combo, filename="cards_state.txt"):
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"COMBO {combo}\n")
        if cards:
            f.write("CARDS " + " ".join(cards) + "\n")
        else:
            f.write("CARDS\n")


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def nothing(x):
    pass


def is_valid_card_ratio(width, height, err=0.75):
    target_ratio = 2 / 3
    ratio = width / height
    return (target_ratio * (1 - err) <= ratio <= target_ratio * (1 + err))


def parse_card(label):
    rank_str, suit = label.split("_")
    rank_map = {
        "2": 2, "3": 3, "4": 4, "5": 5,
        "6": 6, "7": 7, "8": 8, "9": 9,
        "10": 10, "J": 11, "Q": 12, "K": 13, "A": 14
    }
    return rank_map[rank_str], suit


lookup_table = {
    "Pair": lambda ranks, suits: (
        len(ranks) == 2 and len(set(ranks)) == 1
    ),
    "Double Pair": lambda ranks, suits: (
        len(ranks) == 4 and sorted([ranks.count(x) for x in set(ranks)]) == [2, 2]
    ),
    "Royal Flush": lambda ranks, suits: (
        len(ranks) == 5 and len(set(suits)) == 1 and
        sorted(ranks) == [10, 11, 12, 13, 14]
    ),
    "Straight Flush": lambda ranks, suits: (
        len(ranks) == 5 and len(set(suits)) == 1 and
        len(set(ranks)) == 5 and max(ranks) - min(ranks) == 4
    ),
}


def detect_hand_LUT(cards):
    if len(cards) < 2:
        return "Tidak Valid"

    ranks, suits = [], []
    for c in cards:
        r, s = parse_card(c)
        ranks.append(r)
        suits.append(s)

    for combo_name, rule in lookup_table.items():
        if rule(ranks, suits):
            return combo_name

    return "Tidak Valid"


# ---- scoring untuk melawan komputer ----

HAND_SCORE = {
    "High Card": 1,
    "Pair": 2,
    "Double Pair": 3,
    "Straight Flush": 4,
    "Royal Flush": 5,
    "Tidak Valid": 0,
}

def evaluate_hand(cards):
    """
    cards: list label, contoh ['10_heart', 'J_heart', ...]
    """
    combo = detect_hand_LUT(cards)
    # kalau tidak cocok tapi ada kartu, anggap High Card
    if combo == "Tidak Valid" and len(cards) >= 1:
        combo = "High Card"

    ranks = sorted([parse_card(c)[0] for c in cards], reverse=True)
    score = HAND_SCORE.get(combo, 0)
    return score, ranks, combo


def compare_hands(player_cards, bot_cards):
    ps, pr, pc = evaluate_hand(player_cards)
    bs, br, bc = evaluate_hand(bot_cards)

    if ps > bs:
        return "PLAYER WINS", pc, bc
    elif bs > ps:
        return "BOT WINS", pc, bc
    else:
        # tie → cek kicker (ranks)
        if pr > br:
            return "PLAYER WINS", pc, bc
        elif br > pr:
            return "BOT WINS", pc, bc
        else:
            return "DRAW", pc, bc


# deck untuk bot & player (52 kartu)
RANKS = ["2","3","4","5","6","7","8","9","10","J","Q","K","A"]
SUITS = ["club","diamond","heart","spade"]
DECK = [f"{r}_{s}" for r in RANKS for s in SUITS]

class EngineState:
    def __init__(self):
        self.player_hand = []   # 26 kartu pemain
        self.bot_hand    = []   # 26 kartu bot

def init_new_game():
    deck = DECK[:]
    random.shuffle(deck)
    half = len(deck)//2
    state = EngineState()
    state.player_hand = deck[:half]
    state.bot_hand    = deck[half:]
    return state

# --- GLOBAL ENGINE STATE ---
engine_state = init_new_game()

# ===================== ENGINE STATE UNTUK 2 PLAYER =====================

class GameState:
    def __init__(self, player_hand, bot_hand):
        self.player_hand = player_hand  # 26 kartu di tangan pemain
        self.bot_hand = bot_hand        # 26 kartu di tangan bot


def init_game_state():
    """Shuffle deck dan bagi 26-26 ke player & bot."""
    deck_copy = DECK.copy()
    random.shuffle(deck_copy)

    player_hand = deck_copy[:26]
    bot_hand = deck_copy[26:52]   # sisa 26 kartu

    return GameState(player_hand, bot_hand)


# bikin 1 instance global yang dipakai di seluruh game
state = init_game_state()


# CNN Inference

cap = cv2.VideoCapture(0)

DirektoriDataSet = "dataset"
LabelKelas = tuple(
    sorted(
        d for d in os.listdir(DirektoriDataSet)
        if os.path.isdir(os.path.join(DirektoriDataSet, d))
    )
)
print("LabelKelas:", LabelKelas)

ModelCNN = load_model("Hasil.h5")
print("Model CNN loaded.")
IMG_SIZE = 128

kernel = np.ones((5, 5), np.uint8)

cv2.namedWindow("Tracking")
cv2.createTrackbar("LH", "Tracking", 60, 255, nothing)
cv2.createTrackbar("LS", "Tracking", 30, 255, nothing)
cv2.createTrackbar("LV", "Tracking", 40, 255, nothing)
cv2.createTrackbar("UH", "Tracking", 100, 255, nothing)
cv2.createTrackbar("US", "Tracking", 255, 255, nothing)
cv2.createTrackbar("UV", "Tracking", 255, 255, nothing)


def KlasifikasiCitraTunggal(warped_card_img, LabelKelas, ModelCNN, threshold=0.5):
    img = cv2.resize(warped_card_img, (IMG_SIZE, IMG_SIZE))
    img = np.asarray(img) / 255.0
    img = img.astype("float32")
    X = np.expand_dims(img, axis=0)
    hs = ModelCNN.predict(X, verbose=0)[0]

    if hs.max() > threshold:
        idx = int(np.argmax(hs))
        kelas = LabelKelas[idx]
    else:
        idx = -1
        kelas = "UNKNOWN"

    confidence = float(hs.max())
    return kelas, confidence


# UI

pygame.init()
WIDTH, HEIGHT = 1920, 1080
WIN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("2 PLAYER CAPSA")

FONT = pygame.font.SysFont("consolas", 22)
SMALL = pygame.font.SysFont("consolas", 18)
BIG = pygame.font.SysFont("consolas", 48)

BG_COLOR = (250, 250, 250)
CARD_COLOR = (10, 10, 10)
TEXT_COLOR = (10, 10, 10)
ACCENT = (10, 10, 10)

clock = pygame.time.Clock()

# ukuran kartu
CARD_W = 128
CARD_H = 128

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CARD_ASSET_DIR = os.path.join(BASE_DIR, "assets", "cards")

_card_texture_cache = {}


def get_card_texture(label, size=None):
    if size is None:
        size = (CARD_W, CARD_H)

    key = (label, size)
    if key in _card_texture_cache:
        return _card_texture_cache[key]

    filename = f"{label}.png"   # contoh: 5_spade.png
    path = os.path.join(CARD_ASSET_DIR, filename)

    if not os.path.exists(path):
        _card_texture_cache[key] = None
        return None

    try:
        img = pygame.image.load(path).convert_alpha()
        img = pygame.transform.smoothscale(img, size)
        _card_texture_cache[key] = img
        return img
    except Exception as e:
        print(f"Gagal memuat texture {path}: {e}")
        _card_texture_cache[key] = None
        return None


class CardSprite:
    def __init__(self, label, x, y, w=CARD_W, h=CARD_H):
        self.label = label
        self.rect = pygame.Rect(x, y, w, h)
        self.image = get_card_texture(label, (w, h))

    def draw(self, surface):
        if self.image is not None:
            surface.blit(self.image, self.rect.topleft)
        else:
            pygame.draw.rect(surface, CARD_COLOR, self.rect, border_radius=10)
            pygame.draw.rect(surface, (180, 180, 180), self.rect, 2, border_radius=10)

            if "_" in self.label:
                rank_str, suit = self.label.split("_")
            else:
                rank_str, suit = self.label, "?"

            rank_text = FONT.render(rank_str, True, TEXT_COLOR)
            surface.blit(rank_text, (self.rect.x + 6, self.rect.y + 6))

            suit_short = suit[0].upper() if suit else "?"
            suit_text = FONT.render(suit_short, True, TEXT_COLOR)
            surface.blit(suit_text, (self.rect.x + 6, self.rect.y + 40))


def create_card_sprites(cards, start_x, start_y, per_row=10):
    sprites = []
    gap_x = CARD_W + 10
    x = start_x
    y = start_y

    for i, c in enumerate(cards):
        spr = CardSprite(c, x, y)
        sprites.append(spr)
        x += gap_x
        if (i + 1) % per_row == 0:
            x = start_x
            y += CARD_H + 20

    return sprites

# Urutan buat sort kartu di tangan (biar rapi)
SUIT_ORDER = {
    "club": 0,
    "diamond": 1,
    "heart": 2,
    "spade": 3,
}

def card_sort_key(label: str):
    """
    label: 'J_spade', '10_heart', dll
    output: tuple (suit_order, rank) untuk sorted()
    """
    rank, suit = parse_card(label)   # parse_card sudah kamu punya di atas
    return (SUIT_ORDER.get(suit, 99), rank)


def create_hand_sprites(cards, start_x, start_y, per_row=13, scale=0.6):
    """
    Render 26 kartu di tangan (lebih kecil dari kartu di meja).
    cards: list label di tangan player (state.player_hand)
    """
    w = int(CARD_W * scale)
    h = int(CARD_H * scale)
    gap_x = w + 6

    sprites = []
    x, y = start_x, start_y

    # urutkan supaya rapi
    cards_sorted = sorted(cards, key=card_sort_key)

    for i, c in enumerate(cards_sorted):
        spr = CardSprite(c, x, y, w, h)
        sprites.append(spr)
        x += gap_x
        if (i + 1) % per_row == 0:
            x = start_x
            y += h + 10

    return sprites


def cv_frame_to_surface(frame_bgr, max_width, max_height):
    """
    Convert frame OpenCV (BGR) jadi pygame.Surface dan scale agar muat box.
    """
    if frame_bgr is None:
        return None

    h, w = frame_bgr.shape[:2]

    # hitung scale supaya fit ke max_width x max_height
    scale = min(max_width / w, max_height / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(frame_bgr, (new_w, new_h))
    frame_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    surface = pygame.image.frombuffer(frame_rgb.tobytes(), (new_w, new_h), "RGB")
    return surface

def draw_main_menu():
    WIN.fill(BG_COLOR)

    title = BIG.render("2 PLAYER CAPSA", True, TEXT_COLOR)
    title_rect = title.get_rect(center=(WIDTH // 2, HEIGHT // 3))
    WIN.blit(title, title_rect)

    # tombol start
    start_text = FONT.render("START", True, BG_COLOR)
    quit_text = FONT.render("QUIT", True, BG_COLOR)

    button_w, button_h = 200, 60
    gap = 20
    total_h = button_h * 2 + gap

    start_rect = pygame.Rect(0, 0, button_w, button_h)
    quit_rect = pygame.Rect(0, 0, button_w, button_h)

    start_rect.center = (WIDTH // 2, HEIGHT // 2)
    quit_rect.center = (WIDTH // 2, HEIGHT // 2 + button_h + gap)

    # simpan rect di global biar gampang cek klik
    return_rects = {"start": start_rect, "quit": quit_rect}

    pygame.draw.rect(WIN, ACCENT, start_rect, border_radius=10)
    pygame.draw.rect(WIN, ACCENT, quit_rect, border_radius=10)

    WIN.blit(start_text, start_text.get_rect(center=start_rect.center))
    WIN.blit(quit_text, quit_text.get_rect(center=quit_rect.center))

    info = SMALL.render("ENTER/SPACE = START, ESC = QUIT", True, (120, 120, 120))
    WIN.blit(info, (WIDTH // 2 - info.get_width() // 2, HEIGHT * 0.75))

    pygame.display.flip()
    return return_rects

def draw_game_ui(
    player_cards_now,      # kartu yang sedang terdeteksi kamera
    player_combo_text,     # combo dari kamera / engine
    bot_combo_text,        # combo bot untuk ronde ini
    camera_frame,
    winner_text,
    status_text,
    state,                 # engine_state (punya player_hand, bot_hand)
    bot_cards_on_table,    # kartu bot yang lagi di meja (bukan di tangan)
    show_hand,             # bool toggle daftar kartu tangan
    last_player_play,
    last_bot_play,
    last_player_combo,
    last_bot_combo,
    round_history,
):
    WIN.fill(BG_COLOR)
    mid_x = WIDTH // 2

    # Garis pemisah tengah
    pygame.draw.line(WIN, (180, 180, 180), (mid_x, 0), (mid_x, HEIGHT), 2)

    # ===== PLAYER (KIRI) =====
    player_title = BIG.render("PLAYER", True, TEXT_COLOR)
    WIN.blit(player_title, (40, 20))

    combo_txt = FONT.render(f"Combo (kamera): {player_combo_text}", True, ACCENT)
    WIN.blit(combo_txt, (40, 60))

    # info sisa kartu di tangan
    hand_info = SMALL.render(f"Sisa kartu di tangan: {len(state.player_hand)}", True, TEXT_COLOR)
    WIN.blit(hand_info, (40, 90))

    # ---- kamera ----
    cam_surface = cv_frame_to_surface(
        camera_frame,
        max_width=mid_x - 80,
        max_height=HEIGHT - 260,
    )
    cards_y_start = 160

    if cam_surface is not None:
        cam_rect = cam_surface.get_rect()
        cam_rect.topleft = (40, 120)
        WIN.blit(cam_surface, cam_rect.topleft)
        pygame.draw.rect(
            WIN, (180, 180, 180),
            (cam_rect.x - 2, cam_rect.y - 2,
             cam_rect.width + 4, cam_rect.height + 4), 2
        )
        cards_y_start = cam_rect.bottom + 20

    # ---- kartu yang sedang terdeteksi di meja ----
    if player_cards_now:
        label = SMALL.render("Kartu terdeteksi:", True, TEXT_COLOR)
        WIN.blit(label, (40, cards_y_start - 30))
        sprites = create_card_sprites(player_cards_now, 40, cards_y_start)
        for spr in sprites:
            spr.draw(WIN)
    else:
        no_card = FONT.render("Tidak ada kartu terdeteksi.", True, TEXT_COLOR)
        WIN.blit(no_card, (40, cards_y_start))

    # ---- kartu di tangan pemain (26 kartu) ----
    hand_bar_y = HEIGHT - 150
    toggle_text = "[H] Sembunyikan kartu tangan" if show_hand else "[H] Tampilkan kartu tangan"
    hand_label = SMALL.render(
        f"Kartu di tangan pemain ({len(state.player_hand)}): {toggle_text}",
        True, TEXT_COLOR
    )
    WIN.blit(hand_label, (40, hand_bar_y))

    if show_hand and state.player_hand:
        hand_sprites = create_hand_sprites(
            state.player_hand,
            start_x=40,
            start_y=hand_bar_y + 25,
            per_row=13,
            scale=0.6,
        )
        for spr in hand_sprites:
            spr.draw(WIN)

    # ===== BOT (KANAN) =====
    bot_title = BIG.render("BOT COMPUTER", True, TEXT_COLOR)
    WIN.blit(bot_title, (mid_x + 40, 20))

    bot_hand_info = SMALL.render(f"Sisa kartu bot: {len(state.bot_hand)}", True, TEXT_COLOR)
    WIN.blit(bot_hand_info, (mid_x + 40, 60))

    bot_combo_txt = FONT.render(f"Combo bot: {bot_combo_text}", True, ACCENT)
    WIN.blit(bot_combo_txt, (mid_x + 40, 90))

    # kartu bot yang lagi di meja
    y_bot_cards = 140
    if bot_cards_on_table:
        label = SMALL.render("Kartu bot di meja:", True, TEXT_COLOR)
        WIN.blit(label, (mid_x + 40, y_bot_cards))
        bot_sprites = create_card_sprites(bot_cards_on_table, mid_x + 40, y_bot_cards + 25)
        for spr in bot_sprites:
            spr.draw(WIN)
        y_bot_cards += 25 + CARD_H * 0.6 + 20
    else:
        no_bot = SMALL.render("Belum ada kartu bot.", True, TEXT_COLOR)
        WIN.blit(no_bot, (mid_x + 40, y_bot_cards))

    # winner teks besar
    if winner_text:
        winner_surf = BIG.render(winner_text, True, (200, 0, 0))
        WIN.blit(winner_surf, (mid_x + 40, y_bot_cards + 10))
        y_bot_cards += 80

    # status (instruksi)
    info = SMALL.render(status_text, True, (120, 120, 120))
    WIN.blit(info, (mid_x + 40, y_bot_cards + 10))

    # ==== history ronde terakhir ====
    history_y = y_bot_cards + 40
    hist_title = SMALL.render("History ronde:", True, TEXT_COLOR)
    WIN.blit(hist_title, (mid_x + 40, history_y))
    history_y += 20

    if round_history:
        # Tampilkan maksimal 4 ronde terakhir (dari yang terbaru)
        for rnd in reversed(round_history[-4:]):
            res, p_combo, p_cards, b_combo, b_cards = rnd
            line1 = SMALL.render(
                f"{res} | P: {p_combo} ({', '.join(p_cards)})",
                True, TEXT_COLOR
            )
            WIN.blit(line1, (mid_x + 40, history_y))
            history_y += 18
            line2 = SMALL.render(
                f"      B: {b_combo} ({', '.join(b_cards)})",
                True, TEXT_COLOR
            )
            WIN.blit(line2, (mid_x + 40, history_y))
            history_y += 22
    else:
        no_hist = SMALL.render("Belum ada history ronde.", True, (120, 120, 120))
        WIN.blit(no_hist, (mid_x + 40, history_y))

    pygame.display.flip()





def update_pygame_ui(cards, combo, camera_frame):
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return False

    clock.tick(60)
    WIN.fill(BG_COLOR)

    # --- text info di kiri atas ---
    title = BIG.render("CAPSA GAME", True, TEXT_COLOR)
    WIN.blit(title, (40, 30))

    combo_txt = FONT.render(f"Combo: {combo}", True, ACCENT)
    WIN.blit(combo_txt, (40, 90))

    info_text = "ESC untuk keluar."
    info = SMALL.render(info_text, True, (120, 120, 120))
    WIN.blit(info, (40, 130))

    # --- kartu di kiri ---
    if not cards:
        no_card = FONT.render("Tidak ada kartu terdeteksi.", True, TEXT_COLOR)
        WIN.blit(no_card, (40, 200))
    else:
        label = SMALL.render("Kartu terdeteksi:", True, TEXT_COLOR)
        WIN.blit(label, (40, 170))
        sprites = create_card_sprites(cards, 40, 200)
        for spr in sprites:
            spr.draw(WIN)

    # --- kamera di kanan ---
    cam_surface = cv_frame_to_surface(camera_frame, max_width=700, max_height=500)
    if cam_surface is not None:
        cam_rect = cam_surface.get_rect()
        cam_rect.topright = (WIDTH - 40, 200)   # pojok kanan atas dengan margin 40
        WIN.blit(cam_surface, cam_rect.topleft)

        # border
        pygame.draw.rect(WIN, (180, 180, 180),
                         (cam_rect.x - 2, cam_rect.y - 2,
                          cam_rect.width + 4, cam_rect.height + 4), 2)

    pygame.display.flip()
    return True

# ===================== GAME STATE =====================

game_state = "menu"       # "menu" atau "play"
play_phase = "live"       # "live" atau "result"

bot_combo = "Belum dimainkan"
bot_cards = []
winner_text = ""
status_text = "SPACE: main vs bot | ESC: kembali ke menu"

locked_player_cards = []
locked_player_combo = "Tidak Valid"

player_display_combo = "Tidak Valid"
recognized_unique = []
combo = "Tidak Valid"

engine_state: GameState = deal_new_game()
bot_last_combo: Combo | None = None
player_last_combo: Combo | None = None
status_text = "SPACE: main vs bot | BACKSPACE: PASS | ESC: menu"

show_hand = True  # toggle list kartu tangan (H untuk show/hide)

# history ronde
round_history = []            # simpan beberapa ronde terakhir
last_player_play = []
last_bot_play = []
last_player_combo = "Tidak Valid"
last_bot_combo = "Belum dimainkan"


# main loop
running = True

while running:
    if game_state == "menu":
        button_rects = draw_main_menu()

        menu_loop = True
        while menu_loop:
            clock.tick(60)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    menu_loop = False

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                        menu_loop = False
                    elif event.key in (pygame.K_RETURN, pygame.K_SPACE):
                        # mulai game baru
                        engine_state = deal_new_game()
                        bot_last_combo = None
                        winner_text = ""
                        status_text = (
                            "SPACE = mainkan kartu dari kamera "
                            "| BACKSPACE = PASS | ESC = kembali ke menu"
                        )
                        game_state = "play"
                        menu_loop = False

                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    mx, my = event.pos
                    if button_rects["start"].collidepoint(mx, my):
                        engine_state = deal_new_game()
                        bot_last_combo = None
                        winner_text = ""
                        status_text = (
                            "SPACE = mainkan kartu dari kamera "
                            "| BACKSPACE = PASS | ESC = kembali ke menu"
                        )
                        game_state = "play"
                        menu_loop = False
                    elif button_rects["quit"].collidepoint(mx, my):
                        running = False
                        menu_loop = False

        if not running:
            break

        # lanjut ke iterasi while berikutnya (game_state sudah 'play')
        continue



    ret, img = cap.read()
    if not ret:
        break

    else:
        
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        l_h = cv2.getTrackbarPos("LH", "Tracking")
        l_s = cv2.getTrackbarPos("LS", "Tracking")
        l_v = cv2.getTrackbarPos("LV", "Tracking")
        u_h = cv2.getTrackbarPos("UH", "Tracking")
        u_s = cv2.getTrackbarPos("US", "Tracking")
        u_v = cv2.getTrackbarPos("UV", "Tracking")

        lower_bound = np.array([l_h, l_s, l_v])
        upper_bound = np.array([u_h, u_s, u_v])

        mask_green = cv2.inRange(hsv, lower_bound, upper_bound)
        mask_cards = cv2.bitwise_not(mask_green)

        mask_opened = cv2.morphologyEx(mask_cards, cv2.MORPH_OPEN, kernel)
        mask_opclo = cv2.morphologyEx(mask_opened, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask_opclo, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        result_frame = img.copy()

        warped_cards = []
        detected_count = 0
        recognized_cards = []

        for contour in contours:
            if cv2.contourArea(contour) < 3000:
                continue

            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(approx)
                if not is_valid_card_ratio(w, h):
                    continue

                src_pts = order_points(approx.reshape(4, 2))
                width, height = 200, 300
                dst_pts = np.array(
                    [[0, 0], [width - 1, 0],
                    [width - 1, height - 1], [0, height - 1]],
                    dtype="float32"
                )
                matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
                warped = cv2.warpPerspective(img, matrix, (width, height))
                if warped.shape[0] < warped.shape[1]:
                    warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)

                best_card, confidence = KlasifikasiCitraTunggal(
                    warped, LabelKelas, ModelCNN, threshold=0.5
                )

                if best_card != "UNKNOWN":
                    recognized_cards.append(best_card)

                cv2.putText(
                    result_frame,
                    f"{best_card} ({confidence:.2f})",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1
                )

                warped_cards.append(warped)
                cv2.drawContours(result_frame, [approx], -1, (0, 255, 0), 2)
                detected_count += 1

        # hilangkan duplikat, tentukan combo dari LUT
        recognized_unique = list(dict.fromkeys(recognized_cards))
        player_combo_from_cam = (
            detect_combo(recognized_unique).type.name.replace("_", " ").title()
            if recognized_unique and detect_combo(recognized_unique) is not None
            else "Tidak Valid"
        )

        # ---- EVENT PYGAME DI MODE PLAY ----
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    game_state = "menu"

                elif event.key == pygame.K_h:
                    show_hand = not show_hand

                elif event.key == pygame.K_SPACE:
                    # hanya main kalau ada kartu terdeteksi + kartu itu memang ada di tangan
                    if not recognized_unique:
                        winner_text = ""
                        bot_combo = "Belum dimainkan"
                        bot_cards = []
                        status_text = "Tidak ada kartu untuk bermain. Taruh kartu, lalu tekan SPACE."
                        player_display_combo = player_combo_text  # atau combo lama
                    else:
                        # kartu yang dipakai ronde ini
                        cards_for_round = recognized_unique[:5]
                        # cek kartunya benar-benar ada di tangan
                        if not all(c in state.player_hand for c in cards_for_round):
                            status_text = "Kartu yang dipilih tidak ada di tangan pemain."
                        else:
                            n = len(cards_for_round)
                            # hapus dari tangan player
                            for c in cards_for_round:
                                state.player_hand.remove(c)

                            # bot pilih kartu
                            available = [c for c in state.bot_hand if c not in cards_for_round]
                            n = min(n, len(available))
                            bot_hand = random.sample(available, n)
                            for c in bot_hand:
                                state.bot_hand.remove(c)

                            bot_cards = bot_hand  # tampil di meja

                            result, player_used_combo, bot_used_combo = compare_hands(
                                cards_for_round, bot_hand
                            )
                            winner_text = result
                            bot_combo = bot_used_combo
                            player_display_combo = player_used_combo
                            status_text = "SPACE: ronde berikutnya | ESC: kembali ke menu"

                            # simpan history ronde
                            last_player_play = cards_for_round
                            last_bot_play = bot_hand
                            last_player_combo = player_used_combo
                            last_bot_combo = bot_used_combo
                            round_history.append(
                                (result, player_used_combo, cards_for_round, bot_used_combo, bot_hand)
                            )
                            if len(round_history) > 10:
                                round_history.pop(0)

        # cek apakah ada yg habis kartu → menang game
        if not engine_state.player_hand:
            winner_text = "PLAYER MENANG GAME! (kartu habis)"
        elif not engine_state.bot_hand:
            winner_text = "BOT MENANG GAME! (kartu habis)"

            # recognized_unique = list(dict.fromkeys(recognized_cards))
        player_cards_now = recognized_unique          # kartu deteksi kamera
        player_combo_text = combo                    # atau player_display_combo kalau kamu bedakan

        # panggil UI
        draw_game_ui(
            recognized_unique,
            player_display_combo,
            bot_combo,
            result_frame,
            winner_text,
            status_text,
            state,           # <--- ini
            bot_cards,
            show_hand,
            last_player_play,
            last_bot_play,
            last_player_combo,
            last_bot_combo,
            round_history,
        )









        # draw_game_ui(recognized_unique, combo, bot_combo, result_frame)

        # tombol ESC dari window OpenCV masih aktif kalau kamu mau:
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

cap.release()
cv2.destroyAllWindows()
pygame.quit()
