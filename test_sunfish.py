import numpy as np
import torch
import torch.nn as nn


from chess_llm_db.chess_tokenizer import ChessTokenizer
from sunfish import Searcher, Position, initial, render, Move
from transformers import GPT2LMHeadModel, AutoModel


def main():

    vocab_path = "./data/vocab.txt"
    tokenizer = ChessTokenizer(vocab_path)
    game_prefix = [tokenizer.bos_token_id]
    model = GPT2LMHeadModel.from_pretrained('shtoshni/gpt2-chess-uci')
    softmax = nn.Softmax()
    hist = [Position(initial, 0, (True, True), (True, True), 0, 0, "")]
    searcher = Searcher()
    moves_str = []
    moves = []
    troll_lambda = 0.1
    
    for _ in range(10):
        move_str = None
        move = None
        c = 0
        greedy_game_prefix = list(game_prefix)
        prefix_tens = torch.tensor([greedy_game_prefix])
        with torch.no_grad():
            logits = model(prefix_tens)[0]
            prob_next = softmax(logits.flatten()).detach().numpy()
            #prob_next[np.where(prob_next < 0.0025)] = 1 # ignore actions that have < .25% prob of being played
            prob_next = 1.0 - prob_next

        get_troll_score = lambda gamma, idx: prob_next[idx]*gamma*troll_lambda
        for depth, gamma, score, move in searcher.search(hist):
            # The only way we can be sure to have the real move in tp_move,
            # is if we have just failed high.
            move_str = render(move.i) + render(move.j) + move.prom.lower()
            idx = tokenizer.vocab[move_str[:2]]
            #troll_score_ = get_troll_score(gamma, idx)
            if score >= gamma:
                i, j = move.i, move.j
                if len(hist) % 2 == 0:
                    i, j = 119 - i, 119 - j
                print("info depth", depth, "score cp", score, "pv", move_str)
                c += 1 
            if c >= 15:
                break

        hist.append(hist[-1].move(Move(move.i, move.j, move.prom)))
        moves.append(move)
        moves_str.append(move_str)
        game_prefix.extend(tokenizer.encode(move_str, add_special_tokens=False, get_move_end_positions=False))


   
if __name__ == "__main__":
    main()
