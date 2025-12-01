import pygame
import os

# ============================
# 1. API: baca cards_state.txt
# ============================

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


# ============================
# 2. Pygame UI
# ============================

pygame.init()
WIDTH, HEIGHT = 1000, 700
WIN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Capsa Detector (Pygame)")

FONT = pygame.font.SysFont("consolas", 22)
SMALL = pygame.font.SysFont("consolas", 18)
BIG = pygame.font.SysFont("consolas", 28)

BG_COLOR = (30, 30, 40)
CARD_COLOR = (50, 50, 80)
TEXT_COLOR = (230, 230, 230)
ACCENT = (0, 200, 120)

clock = pygame.time.Clock()


class CardSprite:
    def __init__(self, label, x, y, w=80, h=110):
        self.label = label
        self.rect = pygame.Rect(x, y, w, h)

    def draw(self, surface):
        pygame.draw.rect(surface, CARD_COLOR, self.rect, border_radius=10)
        pygame.draw.rect(surface, (180, 180, 180), self.rect, 2, border_radius=10)

        # label: "7_club" -> "7" dan "C"
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


def main():
    state = load_cards_state("cards_state.txt")
    sprites = create_card_sprites(state.cards, 40, 200)

    info_text = "Tekan [R] untuk reload cards_state.txt | Jalankan CardGrid.py di jendela lain."

    running = True
    while running:
        clock.tick(60)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    # reload file dari CardGrid.py
                    state = load_cards_state("cards_state.txt")
                    sprites = create_card_sprites(state.cards, 40, 200)

        # ==== RENDER ====
        WIN.fill(BG_COLOR)

        title = BIG.render("Capsa Card Detector (Pygame + CardGrid.py)", True, TEXT_COLOR)
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
