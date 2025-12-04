import pygame
import os

# TXT Api

class GameAPIState:
    def __init__(self, combo="Tidak Ada Data", cards=None):
        self.combo = combo
        self.cards = cards or []


def load_cards_state(filename="cards_state.txt"):
    """
    Membaca file txt dengan format:
    COMBO <nama_combo>
    CARDS <c1> <c2> ...
    """
    state = GameAPIState()

    if not os.path.exists(filename):
        state.combo = "File belum ada"
        return state

    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            tag = parts[0]

            if tag == "COMBO":
                state.combo = " ".join(parts[1:]) if len(parts) > 1 else "Tidak Valid"
            elif tag == "CARDS":
                state.cards = parts[1:] if len(parts) > 1 else []

    return state


# UI
pygame.init()
WIDTH, HEIGHT = 1920, 1080
WIN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("CAPSA GAME")

FONT = pygame.font.SysFont("consolas", 22)
SMALL = pygame.font.SysFont("consolas", 18)
BIG = pygame.font.SysFont("consolas", 28)

BG_COLOR = (250, 250, 250)
CARD_COLOR = (10, 10, 10)
TEXT_COLOR = (10, 10, 10)
ACCENT = (10, 10, 10)

clock = pygame.time.Clock()

# Texture Loader
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CARD_ASSET_DIR = os.path.join(BASE_DIR, "assets", "cards")

# cache supaya tidak load file berulang-ulang
_card_texture_cache = {}


def get_card_texture(label, size):
    """
    Mencoba load gambar kartu dari assets/cards/<label>.png
    dan resize ke 'size' (w, h).
    Kalau gagal, mengembalikan None.
    """
    key = (label, size)

    if key in _card_texture_cache:
        return _card_texture_cache[key]

    filename = f"{label}.png"
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


# ============ CARD SPRITE ============

class CardSprite:
    def __init__(self, label, x, y, w=80, h=110):
        self.label = label
        self.rect = pygame.Rect(x, y, w, h)
        # coba load texture berdasarkan label
        self.image = get_card_texture(label, (w, h))

    def draw(self, surface):
        if self.image is not None:
            # kalau ada gambar, pakai gambar
            surface.blit(self.image, self.rect.topleft)
        else:
            # fallback: kotak dan tulisan
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
    gap_x = 90
    x = start_x
    y = start_y

    for i, c in enumerate(cards):
        spr = CardSprite(c, x, y)
        sprites.append(spr)
        x += gap_x
        if (i + 1) % per_row == 0:
            x = start_x
            y += 130

    return sprites


# ============ MAIN LOOP ============

def main():
    state = load_cards_state("cards_state.txt")
    sprites = create_card_sprites(state.cards, 40, 200)

    info_text = "Tekan [R] untuk mengupdate state"

    running = True
    while running:
        clock.tick(60)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    state = load_cards_state("cards_state.txt")
                    sprites = create_card_sprites(state.cards, 40, 200)

        # ==== RENDER ====
        WIN.fill(BG_COLOR)

        title = BIG.render("CAPSA GAME", True, TEXT_COLOR)
        WIN.blit(title, (40, 30))

        combo_txt = FONT.render(f"Combo: {state.combo}", True, ACCENT)
        WIN.blit(combo_txt, (40, 90))

        info = SMALL.render(info_text, True, (180, 180, 180))
        WIN.blit(info, (40, 130))

        if not state.cards:
            no_card = FONT.render("Tidak ada kartu terdeteksi.", True, TEXT_COLOR)
            WIN.blit(no_card, (40, 200))
        else:
            label = SMALL.render("Kartu terdeteksi:", True, TEXT_COLOR)
            WIN.blit(label, (40, 170))

            for spr in sprites:
                spr.draw(WIN)

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
