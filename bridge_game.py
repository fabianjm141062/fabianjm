import random
import streamlit as st

# Deklarasi suit dan ranks
SUITS = ['Hearts', 'Diamonds', 'Clubs', 'Spades']
RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']

# Buat dek kartu Bridge (52 kartu)
def create_deck():
    deck = [f"{rank} of {suit}" for suit in SUITS for rank in RANKS]
    random.shuffle(deck)
    return deck

# Fungsi untuk mengocok dan membagikan kartu ke 4 pemain
def deal_cards(deck):
    player1 = deck[:13]
    player2 = deck[13:26]
    player3 = deck[26:39]
    player4 = deck[39:]
    return player1, player2, player3, player4

# Fungsi AI sederhana untuk memilih kartu terbaik
def ai_play(cards, played_suit=None):
    if played_suit:
        suit_cards = [card for card in cards if played_suit in card]
        if suit_cards:
            return suit_cards[0]  # Pilih kartu dengan suit yang sesuai
    return cards[0]  # Pilih kartu acak jika tidak ada suit yang sama

# Fungsi untuk menampilkan tangan pemain
def show_hands(player_hands):
    for i, hand in enumerate(player_hands, 1):
        st.write(f"Player {i}'s Hand: {', '.join(hand)}")

# Fungsi utama untuk Streamlit
def main():
    st.title("Bridge Game with AI oleh Fabian J Manoppo")

    # Inisialisasi game Bridge
    deck = create_deck()
    player1, player2, player3, player4 = deal_cards(deck)
    player_hands = [player1, player2, player3, player4]

    # Tampilkan tangan pemain (untuk player 1, lainnya disembunyikan)
    show_hands([player1])

    # Input dari pemain manusia
    user_move = st.selectbox("Pilih kartu Anda:", player1)

    if st.button("Mainkan kartu"):
        st.write(f"Anda memainkan: {user_move}")
        player1.remove(user_move)

        # AI memainkan kartu
        played_suit = user_move.split()[-1]  # Suit yang dimainkan
        ai1_card = ai_play(player2, played_suit)
        ai2_card = ai_play(player3, played_suit)
        ai3_card = ai_play(player4, played_suit)

        st.write(f"AI 1 memainkan: {ai1_card}")
        st.write(f"AI 2 memainkan: {ai2_card}")
        st.write(f"AI 3 memainkan: {ai3_card}")

        # Remove AI cards from hands
        player2.remove(ai1_card)
        player3.remove(ai2_card)
        player4.remove(ai3_card)

if __name__ == "__main__":
    main()
