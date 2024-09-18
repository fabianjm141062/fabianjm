import streamlit as st
import chess
import chess.svg
from stockfish import Stockfish

# Path ke Stockfish engine di komputer Anda
STOCKFISH_PATH = "/path/to/stockfish"

# Inisialisasi Stockfish
stockfish = Stockfish(STOCKFISH_PATH)
stockfish.set_skill_level(20)  # Level tertinggi

# Fungsi untuk menginisialisasi papan catur
def initialize_game():
    return chess.Board()

# Fungsi untuk menampilkan papan catur dalam format SVG
def render_board(board):
    return chess.svg.board(board=board)

# Fungsi untuk menjalankan langkah dari Stockfish
def ai_move(board):
    stockfish.set_fen_position(board.fen())
    ai_move = stockfish.get_best_move()
    board.push(chess.Move.from_uci(ai_move))

# Aplikasi Streamlit
def main():
    st.title("AI Chess Game with Grandmaster Level (Stockfish)")

    # Inisialisasi game atau ambil state dari game
    if 'board' not in st.session_state:
        st.session_state.board = initialize_game()
    
    # Render papan catur
    board_svg = render_board(st.session_state.board)
    st.image(board_svg, use_column_width=True)

    # Input dari pengguna
    user_move = st.text_input("Masukkan langkah Anda (format UCI, contoh: e2e4):")

    if st.button("Jalankan Langkah"):
        try:
            move = chess.Move.from_uci(user_move)
            if move in st.session_state.board.legal_moves:
                st.session_state.board.push(move)  # Jalankan langkah pemain
                ai_move(st.session_state.board)    # AI melakukan langkah
            else:
                st.error("Langkah ilegal, coba lagi!")
        except:
            st.error("Format langkah tidak valid!")

    # Tombol untuk mereset game
    if st.button("Reset Game"):
        st.session_state.board = initialize_game()

if __name__ == '__main__':
    main()
