# Hybrid Chess Bot - Neural + Structural Architecture

## Design Philosophy: A Symbiotic System

This chess engine implements a **hybrid architecture** that combines two complementary components:

### 1. The "Predictive Core" (Structural Logic)
The deterministic search algorithm that explores the game tree:
- **Minimax with Alpha-Beta Pruning**: Efficiently explores future game states
- **Iterative Deepening**: Starts shallow, increases depth within time budget
- **Move Ordering**: MVV-LVA (Most Valuable Victim - Least Valuable Attacker) for captures
- **Transposition Tables**: Memoization to avoid re-evaluating identical positions
- **Quiescence Search**: Tactical stability at leaf nodes to avoid horizon effect

### 2. The "Intuitive Evaluator" (Neuro-Aesthetic)
The neural network that perceives board positions:
- **Compact CNN Architecture**: Optimized for fast inference (~64 filters)
- **Residual Connections**: Deep feature extraction with gradient flow
- **Board Encoding**: 14-channel representation (12 pieces + castling/en passant + turn)
- **Trained on Real Games**: Learns from the Simple Chess Database

### 3. The Opening Book Integration
Pre-computed optimal opening moves:
- **Opening Database**: Uses `openings.csv` for first 15 moves
- **Win Rate Selection**: Chooses moves with highest historical win rates
- **Energy Efficiency**: Avoids wasting computation on known theory

## Architecture Details

### Neural Network Architecture
```
Input: (14, 8, 8) tensor
  - Channels 0-11: Piece positions (one-hot encoded)
  - Channel 12: Castling rights and en passant squares
  - Channel 13: Turn indicator

Layers:
  - Conv2d(14 → 64, 3x3) + BatchNorm + ReLU
  - 3x Residual Blocks (64 → 64)
  - Value Head: Conv2d(64 → 32, 1x1) → FC(2048 → 128) → FC(128 → 1)
  
Output: Single scalar [-1, +1] (evaluation from white's perspective)
```

### Search Algorithm
```python
def find_best_move(board, time_limit):
    # 1. Check opening book first (moves 1-15)
    if in_opening_book(board):
        return opening_book_move
    
    # 2. Iterative deepening with time management
    for depth in range(1, max_depth):
        if time_limit_reached():
            break
        
        # 3. Alpha-beta search with move ordering
        for move in ordered_legal_moves:
            score = alpha_beta(board, depth, alpha, beta)
            
            # 4. Neural evaluation at leaf nodes
            if depth == 0:
                return neural_evaluator.evaluate(board)
```

## Key Optimizations

### Time Management
- **280 second budget**: Leaves buffer in 300s limit
- **Adaptive time allocation**: More time for complex middle-game positions
- **Iterative deepening**: Always has a best move even if time runs out

### Search Efficiency
- **Alpha-Beta Pruning**: Cuts ~50% of tree on average
- **Move Ordering**: 
  - Captures (MVV-LVA heuristic)
  - Checks (priority +5000)
  - Promotions (priority +8000)
  - History heuristic (successful cutoff moves)
- **Transposition Table**: Avoids re-searching identical positions
- **Quiescence Search**: Only tactical moves (captures, checks) at leaves

### Memory Management
- **Compact model**: Only 64 filters (~500K parameters)
- **Batch normalization**: Faster convergence, better generalization
- **Cache clearing**: Transposition table limited to 100K entries

## Training Protocol

The neural evaluator is trained on positions extracted from real games:

1. **Data Source**: `shared_resources/games.csv` (20,000+ games)
2. **Position Extraction**: Excludes first 10 moves (opening theory)
3. **Outcome Labeling**: 
   - White win = +1.0
   - Black win = -1.0
   - Draw = 0.0
4. **Position Weighting**: Later positions weighted more heavily
5. **Loss Function**: Mean Squared Error (MSE)
6. **Optimizer**: Adam with learning rate scheduling

### Training Command
```bash
python -m demo_bots.hybrid_bot train
```

## Usage

### Play as White
```bash
python -m demo_bots.hybrid_bot play w
```

### Play as Black
```bash
python -m demo_bots.hybrid_bot play b
```

### Train the Neural Network
```bash
python -m demo_bots.hybrid_bot train
```

## Performance Characteristics

### Strength
- **Opening**: Strong (uses opening book with win rates)
- **Middle Game**: Strong tactical vision (quiescence search)
- **Endgame**: Competent (material + positional evaluation)

### Speed
- **Typical depth**: 5-7 ply within time limit
- **Nodes/second**: ~5000-10000 (depending on hardware)
- **Response time**: 3-15 seconds per move (adaptive)

### Fallback Behavior
If neural network is unavailable (PyTorch not installed):
- Falls back to material evaluation
- Uses piece-square tables for positional awareness
- Still competitive with opening book + alpha-beta search

## File Structure

```
hybrid_bot/
├── __init__.py           # Module initialization
├── __main__.py           # Entry point (CLI handler)
├── interface.py          # I/O interfaces
├── play.py              # Main game loop
├── train.py             # Neural network training
├── engine.py            # Search algorithm + opening book
├── neural_evaluator.py  # Neural network architecture
├── model_weights.pth    # Trained model (created by train.py)
└── README.md           # This file
```

## Dependencies

- `chess >= 1.11.2` - Board representation and move validation
- `torch >= 2.0.0` - Neural network framework
- `numpy >= 1.24.0` - Numerical operations
- `pandas >= 2.0.0` - CSV parsing for game database

## Design Principles

1. **Efficiency over Depth**: Compact model for fast inference
2. **Hybrid Evaluation**: Neural perception + structural search
3. **Knowledge Integration**: Opening book reduces computation waste
4. **Time-Aware**: Adaptive time allocation per move
5. **Graceful Degradation**: Falls back to material eval if needed

## Future Enhancements

Potential improvements for even stronger play:
- [ ] MCTS (Monte Carlo Tree Search) for tactical positions
- [ ] Endgame tablebase integration
- [ ] Deeper networks with GPU acceleration
- [ ] Self-play reinforcement learning
- [ ] Pondering (thinking during opponent's time)

---

**"Do not build a 'calculator.' Build an 'evaluator.' Design a system that perceives the 'spatial potential' of the 'board topology' and acts with ruthless, logical elegance."**
