import streamlit as st
import chess
import chess.svg
import requests

# Lichess API token (buat token dari akun Lichess)
API_TOKEN = "YOUR_LICHESS_API_TOKEN"

# Fungsi untuk mendapatkan langkah terbaik dari Stockfish via API Lichess
def get_best_move(fen):
    url = "https://lichess.org/api/cloud-eval"
    params = {'fen': fen, 'multiPv': 1}
    headers = {'Authorization': f'Bearer {API_TOKEN}'}
    response = requests.get(url, headers=headers, params=params)

    if response.status_code == 200:
        return response.json()['pvs'][0]['moves']
    else:
        return "Error fetching best move"

# Fungsi untuk menginisialisasi papan catur
def initialize_game():
    return chess.Board()

# Fungsi untuk merender papan catur dalam format SVG
def render_board(board):
    return chess.svg.board(board=board)

# Aplikasi Streamlit
def main():
    st.title("AI Chess Game (Stockfish via Lichess API) oleh Fabian J Manoppo")

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

                # Mendapatkan langkah terbaik dari Stockfish melalui API Lichess
                fen = st.session_state.board.fen()
                best_move = get_best_move(fen)
                if "Error" not in best_move:
                    st.session_state.board.push(chess.Move.from_uci(best_move.split()[0]))
                else:
                    st.error(best_move)
            else:
                st.error("Langkah ilegal, coba lagi!")
        except:
            st.error("Format langkah tidak valid!")

    # Tombol untuk mereset game
    if st.button("Reset Game"):
        st.session_state.board = initialize_game()

if __name__ == '__main__':
    main()
