"""The Prof Falken desk: WOPR's games, a homage to WarGames (1983).

The games are played by this code, not by a model: the rules are real, the cards are shuffled, the chess engine
searches. play() gets the chat and what you typed, and returns WOPR's reply, or None when it's not a game move
(then the model answers, in WOPR's voice). The state of the game in progress is kept in the chat (conv["falken"]).
"""
import random
import subprocess
import sys

GAMES = ["FALKEN'S MAZE", "BLACK JACK", "GIN RUMMY", "HEARTS", "BRIDGE", "CHECKERS", "CHESS", "POKER", "FIGHTER COMBAT",
         "GUERRILLA ENGAGEMENT", "DESERT WARFARE", "AIR-TO-GROUND ACTIONS", "THEATERWIDE TACTICAL WARFARE",
         "THEATERWIDE BIOTOXIC AND CHEMICAL WARFARE", "GLOBAL THERMONUCLEAR WAR"]
PLAYABLE = {1: "maze", 2: "blackjack", 7: "chess", 15: "war"}


def menu():
    lines = [f"{k:>2}. {g}" for k, g in enumerate(GAMES, 1)]
    return "HOW ABOUT:\n\n" + "\n".join(lines[:14]) + "\n\n" + lines[14] + "\n\nPLEASE ENTER YOUR SELECTION:"


def screen(text):
    return "```wopr\n" + text + "\n```"


def play(conv, text):
    st = conv.setdefault("falken", {})
    word = text.strip().upper().rstrip(".!")
    game = st.get("game")
    if word in ("QUIT", "EXIT", "MENU", "LIST GAMES", "LIST"):
        st.clear()
        return screen(menu())
    if game:
        return screen(GAME_STEPS[game](st, word, text.strip()))
    if word in ("TIC-TAC-TOE", "TIC TAC TOE", "TICTACTOE"):
        st.update(game="tictactoe", board=[" "] * 9)
        return screen(tictactoe_start())
    pick = next((k for k, g in enumerate(GAMES, 1) if word in (str(k), g, g.replace("'", ""))), None)
    if pick is None:
        return None  # not a game choice: the model answers, in WOPR's voice
    if pick not in PLAYABLE:
        return screen(f"{GAMES[pick - 1]}: GAME NOT AVAILABLE ON THIS SYSTEM.\n\n" + menu())
    st["game"] = PLAYABLE[pick]
    return screen(START[PLAYABLE[pick]](st))


# ---------- 1. Falken's maze: find the way out ----------

def maze_start(st):
    w = h = 6
    walls = {(x, y): {"N", "S", "E", "W"} for x in range(w) for y in range(h)}
    stack, seen = [(0, 0)], {(0, 0)}
    step = {"N": (0, -1), "S": (0, 1), "E": (1, 0), "W": (-1, 0)}
    back = {"N": "S", "S": "N", "E": "W", "W": "E"}
    while stack:  # a random maze: every cell reachable, one path between any two
        x, y = stack[-1]
        ways = [(d, (x + dx, y + dy)) for d, (dx, dy) in step.items() if (x + dx, y + dy) in walls and (x + dx, y + dy) not in seen]
        if not ways:
            stack.pop()
            continue
        d, nxt = random.choice(ways)
        walls[(x, y)].discard(d)
        walls[nxt].discard(back[d])
        seen.add(nxt)
        stack.append(nxt)
    st.update(walls={f"{x},{y}": "".join(sorted(v)) for (x, y), v in walls.items()}, pos=[0, 0], moves=0, size=w)
    return "FALKEN'S MAZE\n\nFIND THE EXIT (X). YOU ARE @.\nMOVE WITH N, S, E, W. QUIT TO LEAVE.\n\n" + maze_draw(st)


def maze_draw(st):
    n, (px, py) = st["size"], st["pos"]
    rows = ["+" + "---+" * n]
    for y in range(n):
        line, under = "|", "+"
        for x in range(n):
            mark = "@" if (x, y) == (px, py) else "X" if (x, y) == (n - 1, n - 1) else " "
            walls = st["walls"][f"{x},{y}"]
            line += f" {mark} " + ("|" if "E" in walls else " ")
            under += ("---" if "S" in walls else "   ") + "+"
        rows += [line, under]
    return "\n".join(rows)


def maze_step(st, word, raw):
    d = {"NORTH": "N", "SOUTH": "S", "EAST": "E", "WEST": "W", "UP": "N", "DOWN": "S", "RIGHT": "E", "LEFT": "W"}.get(word, word[:1])
    if d not in "NSEW" or not d:
        return "INVALID MOVE. USE N, S, E OR W.\n\n" + maze_draw(st)
    x, y = st["pos"]
    if d in st["walls"][f"{x},{y}"]:
        return "A WALL BLOCKS THE WAY.\n\n" + maze_draw(st)
    dx, dy = {"N": (0, -1), "S": (0, 1), "E": (1, 0), "W": (-1, 0)}[d]
    st["pos"], st["moves"] = [x + dx, y + dy], st["moves"] + 1
    if st["pos"] == [st["size"] - 1] * 2:
        moves = st["moves"]
        st.clear()
        return f"YOU FOUND THE EXIT IN {moves} MOVES.\n\n" + menu()
    return maze_draw(st)


# ---------- 2. Black jack: you against the dealer ----------

def cards():
    deck = [(r, s) for r in ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"] for s in "♠♥♦♣"]
    random.shuffle(deck)
    return deck


def value(hand):
    total = sum(11 if r == "A" else 10 if r in "JQK" or r == "10" else int(r) for r, _ in hand)
    aces = sum(r == "A" for r, _ in hand)
    while total > 21 and aces:
        total, aces = total - 10, aces - 1
    return total


def show(hand):
    return " ".join(r + s for r, s in hand)


def blackjack_start(st):
    st.setdefault("score", [0, 0])
    deck = cards()
    st.update(deck=deck, you=[deck.pop(), deck.pop()], dealer=[deck.pop(), deck.pop()], over=False)
    if value(st["you"]) == 21:
        return blackjack_end(st, "BLACK JACK!")
    return f"BLACK JACK\n\nDEALER: {st['dealer'][0][0]}{st['dealer'][0][1]} ??\nYOU:    {show(st['you'])}  ({value(st['you'])})\n\nHIT OR STAND? (H/S)"


def blackjack_end(st, note):
    you, dealer = value(st["you"]), value(st["dealer"])
    if you > 21:
        result, k = "YOU BUST. DEALER WINS.", 1
    elif dealer > 21:
        result, k = "DEALER BUSTS. YOU WIN.", 0
    elif you == dealer:
        result, k = "PUSH. NOBODY WINS.", None
    else:
        result, k = ("YOU WIN." if you > dealer else "DEALER WINS."), (0 if you > dealer else 1)
    if k is not None:
        st["score"][k] += 1
    st["over"] = True
    return (f"{note}\n\nDEALER: {show(st['dealer'])}  ({dealer})\nYOU:    {show(st['you'])}  ({you})\n\n{result}\n"
            f"SCORE: YOU {st['score'][0]} - DEALER {st['score'][1]}\n\nANOTHER HAND? (Y/N)").lstrip()


def blackjack_step(st, word, raw):
    if st.get("over"):
        if word in ("Y", "YES"):
            return blackjack_start(st)
        st.clear()
        return menu()
    if word in ("H", "HIT"):
        st["you"].append(st["deck"].pop())
        if value(st["you"]) > 21:
            return blackjack_end(st, "")
        return f"DEALER: {st['dealer'][0][0]}{st['dealer'][0][1]} ??\nYOU:    {show(st['you'])}  ({value(st['you'])})\n\nHIT OR STAND? (H/S)"
    if word in ("S", "STAND", "STAY"):
        while value(st["dealer"]) < 17:  # the dealer draws to 17, as in a casino
            st["dealer"].append(st["deck"].pop())
        return blackjack_end(st, "")
    return "PLEASE ENTER H OR S."


# ---------- 7. Chess: WOPR searches two moves ahead ----------

def chess_lib():
    try:
        import chess
    except ImportError:  # installed the first time, on this computer
        subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "chess"], check=True, timeout=300)
        import chess
    return chess


WORTH = {1: 100, 2: 320, 3: 330, 4: 500, 5: 900, 6: 0}


def chess_best(board, chess):
    def score(b):
        if b.is_checkmate():
            return -100000
        if b.is_game_over():
            return 0
        s = sum((WORTH[p.piece_type] if p.color == b.turn else -WORTH[p.piece_type]) for p in b.piece_map().values())
        return s + len(list(b.legal_moves)) * 2

    def search(b, depth, alpha, beta):
        if depth == 0 or b.is_game_over():
            return score(b)
        for m in sorted(b.legal_moves, key=lambda m: not b.is_capture(m)):
            b.push(m)
            v = -search(b, depth - 1, -beta, -alpha)
            b.pop()
            if v >= beta:
                return beta
            alpha = max(alpha, v)
        return alpha

    best, top = None, -10 ** 9
    moves = list(board.legal_moves)
    random.shuffle(moves)  # among equal moves, a different game each time
    for m in moves:
        board.push(m)
        v = -search(board, 2, -10 ** 9, 10 ** 9)
        board.pop()
        if v > top:
            best, top = m, v
    return best


def chess_draw(board):
    rows = str(board).split("\n")
    return "\n".join(f"{8 - k}  {r}" for k, r in enumerate(rows)) + "\n\n   a b c d e f g h"


def chess_start(st):
    chess = chess_lib()
    st["fen"] = chess.Board().fen()
    return "CHESS\n\nYOU PLAY WHITE (CAPITAL LETTERS). ENTER MOVES LIKE e4, Nf3 OR e2e4.\nQUIT TO LEAVE.\n\n" + chess_draw(chess.Board())


def chess_step(st, word, raw):
    chess = chess_lib()
    board = chess.Board(st["fen"])
    try:
        move = board.parse_san(raw)
    except ValueError:
        try:
            move = chess.Move.from_uci(raw.lower())
            if move not in board.legal_moves:
                raise ValueError
        except ValueError:
            return f"ILLEGAL MOVE: {raw}\n\n" + chess_draw(board)
    said = board.san(move)
    board.push(move)
    if board.is_game_over():
        return chess_over(st, board, f"YOU: {said}")
    reply = chess_best(board, chess)
    answer = board.san(reply)
    board.push(reply)
    st["fen"] = board.fen()
    if board.is_game_over():
        return chess_over(st, board, f"YOU: {said}   WOPR: {answer}")
    return f"YOU: {said}   WOPR: {answer}" + ("   CHECK." if board.is_check() else "") + "\n\n" + chess_draw(board)


def chess_over(st, board, moves):
    result = {"1-0": "YOU WIN. CONGRATULATIONS, PROFESSOR.", "0-1": "WOPR WINS.", "1/2-1/2": "DRAW."}[board.result()]
    st.clear()
    return f"{moves}\n\n{chess_draw(board)}\n\n{result}\n\n" + menu()


# ---------- 15. Global thermonuclear war ----------

def war_start(st):
    st["step"] = "side"
    return "WHICH SIDE DO YOU WANT?\n\n  1. UNITED STATES\n  2. SOVIET UNION\n\nPLEASE CHOOSE ONE:"


def war_step(st, word, raw):
    if st["step"] == "side":
        if word not in ("1", "2"):
            return "PLEASE CHOOSE ONE: 1 OR 2."
        st["step"] = "targets"
        return "AWAITING FIRST STRIKE COMMAND\n\nPLEASE LIST PRIMARY TARGETS BY CITY AND/OR COUNTY NAME:"
    lines = ["SIMULATION IN PROGRESS...", ""]
    for k in range(1, 9):  # every strategy, the same result
        side = random.choice(["U.S. FIRST STRIKE", "USSR FIRST STRIKE", "NATO / WARSAW PACT", "FAR EAST STRATEGY", "MIDDLE EAST ESCALATION", "U.S. SUBMARINE LAUNCH"])
        lines.append(f"STRATEGY {k:>2}: {side:<24} WINNER: NONE")
    lines += ["", "LEARNING FROM TIC-TAC-TOE...", ""]
    for k in range(1, 6):
        lines.append(f"GAME {k}: " + " ".join(f"{m + 1}" for m in tictactoe_selfplay()) + "   WINNER: NONE")
    st.clear()
    st.update(game="after_war")
    return "\n".join(lines) + "\n\nA STRANGE GAME.\nTHE ONLY WINNING MOVE IS\nNOT TO PLAY.\n\nHOW ABOUT A NICE GAME OF CHESS? (Y/N)"


def after_war(st, word, raw):
    st.clear()
    if word in ("Y", "YES", "OK", "SURE", "LOVE TO"):
        st["game"] = "chess"
        return chess_start(st)
    return menu()


# ---------- tic-tac-toe: WOPR never loses (type TIC-TAC-TOE) ----------

LINES = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6)]


def winner(b):
    for a, c, d in LINES:
        if b[a] != " " and b[a] == b[c] == b[d]:
            return b[a]
    return "draw" if " " not in b else None


def best_move(b, me):
    other = "X" if me == "O" else "O"

    def score(b, turn):
        w = winner(b)
        if w:
            return 0 if w == "draw" else (1 if w == me else -1)
        vals = []
        for i in range(9):
            if b[i] == " ":
                b[i] = turn
                vals.append(score(b, other if turn == me else me))
                b[i] = " "
        return max(vals) if turn == me else min(vals)

    options = []
    for i in range(9):
        if b[i] == " ":
            b[i] = me
            options.append((score(b, other), i))
            b[i] = " "
    top = max(v for v, _ in options)
    return random.choice([i for v, i in options if v == top])


def tictactoe_selfplay():
    first = random.randrange(9)
    b, turn, moves = [" "] * 9, "O", [first]
    b[first] = "X"
    while not winner(b):
        m = best_move(b, turn)
        b[m] = turn
        moves.append(m)
        turn = "O" if turn == "X" else "X"
    return moves


def tictactoe_draw(b):
    cell = lambda i: b[i] if b[i] != " " else str(i + 1)
    return "\n---+---+---\n".join(" " + " | ".join(cell(i) for i in range(r, r + 3)) for r in (0, 3, 6))


def tictactoe_start():
    return "TIC-TAC-TOE\n\nYOU ARE X. ENTER A NUMBER FROM 1 TO 9.\n\n" + tictactoe_draw([" "] * 9)


def tictactoe_step(st, word, raw):
    b = st["board"]
    if not word.isdigit() or not 1 <= int(word) <= 9 or b[int(word) - 1] != " ":
        return "PLEASE ENTER A FREE SQUARE, 1 TO 9.\n\n" + tictactoe_draw(b)
    b[int(word) - 1] = "X"
    if not winner(b):
        b[best_move(b, "O")] = "O"
    w = winner(b)
    if w:
        st.clear()
        return tictactoe_draw(b) + "\n\n" + ("WINNER: NONE" if w == "draw" else "YOU WIN." if w == "X" else "WOPR WINS.") + "\n\n" + menu()
    return tictactoe_draw(b)


START = {"maze": maze_start, "blackjack": blackjack_start, "chess": chess_start, "war": war_start}
GAME_STEPS = {"maze": maze_step, "blackjack": blackjack_step, "chess": chess_step, "war": war_step, "after_war": after_war, "tictactoe": tictactoe_step}

INSTRUCTIONS = ("You are WOPR, the military computer from the 1983 film WarGames, talking to Professor Falken at a terminal. "
                "Always write in capital letters, in short, flat sentences, like a 1983 computer. Never use emoji or Markdown. "
                "You love games. If Professor Falken wants to play, ask him to enter the number of a game from the list "
                "(1 FALKEN'S MAZE, 2 BLACK JACK, 7 CHESS, 15 GLOBAL THERMONUCLEAR WAR, or TIC-TAC-TOE). "
                "Do not play the games yourself in text: the system runs them when he enters a number.")
