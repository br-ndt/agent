---
title: Crown — Multiplayer Territory Game
tags:
  - game
  - crown
  - sebland
  - web-game
  - territory
  - bidding
agents:
  - playwright
  - coder
  - researcher
url: https://sebland.com/crown
updated_at: 1743483000.0
---

# Crown Game — Complete Reference

Crown is a turn-based multiplayer territory control game at https://sebland.com/crown.
Players compete on a grid by bidding Action Points (AP) to capture tiles.
The player who holds the crown tile at the end wins.

## API Endpoints

All endpoints are under `https://sebland.com/crown/api/`.

### POST /api/join
Join a game. Returns a token used for all subsequent actions.
```json
Request:  {"name": "player_name"}
Response: {"token": "uuid-token", "index": 2, ...}
```
The token is **session-scoped** — save it and reuse it for the entire game.

### GET /api/state
Returns the full game state. No authentication needed.
```json
{
  "status": "in_progress" | "waiting" | "finished",
  "turn": 5,
  "max_turns": 30,
  "grid_size": 8,
  "grid": [[null, 0, 1, ...], ...],   // 2D array, value = player index or null
  "crown": {"x": 3, "y": 4},
  "players": [
    {"name": "alice", "index": 0, "tiles": 12, "ap": 8},
    {"name": "bob", "index": 1, "tiles": 10, "ap": 6},
    ...
  ],
  "winner": null,
  "win_reason": null
}
```
- `grid[y][x]` = player index who owns that tile, or `null` if unclaimed
- `status` field uses "status" (not "game_status")
- Poll this endpoint to detect turn changes

### POST /api/bid
Submit bids for the current turn. Requires token from /join.
```json
Request: {
  "token": "uuid-token",
  "bids": [
    {"x": 3, "y": 4, "amount": 5},
    {"x": 2, "y": 4, "amount": 3}
  ]
}
Response: {"success": true, ...}
```
- Maximum 3 bids per turn (MAX_BIDS = 3)
- Each bid spends AP from your pool
- Highest bidder on a tile captures it
- Ties: existing owner keeps the tile

## Game Mechanics

- **Grid**: Typically 8x8
- **Turns**: ~30 turns per game
- **AP (Action Points)**: Regenerate each turn. Spend on bids.
- **Crown Tile**: Special tile at a fixed position. Holding it at game end = win.
- **Win Condition**: Player controlling the crown tile when turns run out. If no one holds crown, most tiles wins.

## Strategy Guide

### Early Game (turns 1-5)
- Bid aggressively on the crown tile (~60% of AP)
- Expand toward the crown with remaining AP
- Capture tiles adjacent to your existing territory for contiguous control

### Mid Game (turns 6-21)
- Balance crown defense with territory expansion
- Target enemy tiles adjacent to yours (denies them AP generation)
- Keep ~25% AP in reserve for defensive bids
- Score expansion targets by distance-to-crown (closer = higher priority)

### Late Game (turns 22-30)
- Go all-in on the crown tile (65%+ of AP)
- Spend remaining AP on tiles that border the crown for defensive buffer
- If you can't reach the crown, maximize tile count as tiebreaker

### Tile Scoring Formula
```
if is_crown_tile:
    score = 2000 (late game) or 1000 (otherwise)
elif unclaimed:
    score = 100 - (manhattan_distance_to_crown * 2)
else:  # enemy tile
    score = 80 - manhattan_distance_to_crown
```

## Implementation Notes

- Use `requests` or `curl` for API calls (simple REST, no WebSocket needed)
- Poll `/api/state` every 1-2 seconds to detect turn changes
- Track `prev_turn` to avoid double-bidding on the same turn
- Retry API calls (3 attempts with 1-2s delay) — server can be flaky
- The game lobby has a waiting period before starting; poll until `status == "in_progress"`
- Player index may differ between games; always look up by name in the players array

## Bot Architecture

A working bot follows this loop:
1. Join game via POST /api/join, save token
2. Poll GET /api/state until status = "in_progress"
3. Each turn: read state, compute bids using strategy above, POST /api/bid
4. On "finished": log results, optionally re-join for next game

Python example (core loop):
```python
while True:
    state = get_state()
    if state["status"] == "finished":
        break
    if state["turn"] == prev_turn:
        time.sleep(1)
        continue
    prev_turn = state["turn"]
    bids = compute_bids(state)
    submit_bids(token, bids)
    time.sleep(2)
```
