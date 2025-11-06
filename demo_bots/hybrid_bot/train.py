 

import os
import sys
import chess
import chess.pgn
import io
import requests
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from .neural_evaluator import NeuralEvaluator, BoardEncoder, CompactChessNet
import argparse

# Import zstandard for .zst decompression
try:
    import zstandard as zstd
except ImportError:
    print("Installing zstandard for PGN.zst support...")
    os.system(f"{sys.executable} -m pip install zstandard")
    import zstandard as zstd


class ChessGameDataset(Dataset):
    
    def __init__(self, pgn_zst_url=None, pgn_zst_urls=None, pgn_zst_path=None, max_games=5000, min_elo=1600, bot_games_only=False):
        self.positions = []
        self.outcomes = []
        self.encoder = BoardEncoder()
        
        print(f"Loading games from PGN.zst...")
        
        # Support a list of URLs (multiple months)
        if pgn_zst_urls and isinstance(pgn_zst_urls, (list, tuple)):
            self._load_from_multiple_urls(pgn_zst_urls, max_games, min_elo, bot_games_only)
            print(f"Loaded {len(self.positions)} positions (multi-URL)")
            return
        
        # Download if single URL provided
        if pgn_zst_url and not pgn_zst_path:
            pgn_zst_path = self._download_database(pgn_zst_url)
        
        if pgn_zst_path and os.path.exists(pgn_zst_path):
            self._load_games_from_pgn_zst(pgn_zst_path, max_games, min_elo, bot_games_only)
        else:
            print(f"ERROR: No PGN.zst file found!")
            
        print(f"Loaded {len(self.positions)} positions from {max_games} games")
    
    def _download_database(self, url):
        filename = os.path.basename(url)
        filepath = os.path.join(os.path.dirname(__file__), filename)
        
        if os.path.exists(filepath):
            print(f"Database already downloaded: {filepath}")
            return filepath
        
        print(f"Downloading {url}...")
        print(f"This may take a few minutes...")
        
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(filepath, 'wb') as f:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"\rDownloading: {percent:.1f}%", end='', flush=True)
        
        print(f"\n✅ Downloaded to {filepath}")
        return filepath

    def _load_from_multiple_urls(self, urls, max_games, min_elo, bot_games_only):
        unlimited = (max_games is None) or (isinstance(max_games, int) and max_games <= 0)
        games_loaded_total = 0
        for i, url in enumerate(urls, 1):
            try:
                path = self._download_database(url)
                if not path or not os.path.exists(path):
                    print(f"Skipping URL (download failed): {url}")
                    continue
                remaining_games = None
                if not unlimited:
                    remaining_games = max(0, max_games - games_loaded_total)
                    if remaining_games <= 0:
                        break
                loaded_now = self._load_games_from_pgn_zst(path, remaining_games, min_elo, bot_games_only)
                games_loaded_total += loaded_now
                print(f"Month {i}: +{loaded_now} games, total_games={games_loaded_total}, accumulated_positions={len(self.positions)}")
            except Exception as e:
                print(f"Error processing {url}: {e}")
                continue
    
    def _load_games_from_pgn_zst(self, filepath, max_games, min_elo, bot_games_only):
        unlimited = (max_games is None) or (isinstance(max_games, int) and max_games <= 0)
        games_loaded = 0
        games_processed = 0
        games_skipped_parse = 0
        print(f"Decompressing and parsing PGN.zst...")
        print(f"Filters: Bot-only={bot_games_only}, Min Elo={min_elo}")
        
        def iter_pgn_blocks(text_stream):
            """Yield complete PGN game blocks (headers + movetext) as strings without requiring seek()."""
            buf = []
            in_game = False
            in_headers = False
            for line in text_stream:
                if line.startswith('['):
                    # Start of headers or continuation
                    if in_game and not in_headers:
                        # New game starting -> yield the previous game
                        yield ''.join(buf).strip() + "\n"
                        buf = []
                    in_headers = True
                    in_game = True
                    buf.append(line)
                else:
                    if in_headers and line.strip() == '':
                        # End of header section
                        buf.append(line)
                        in_headers = False
                    else:
                        buf.append(line)
            if buf:
                yield ''.join(buf).strip() + "\n"

        def parse_headers_from_block(block_text):
            headers = {}
            for l in block_text.splitlines():
                l = l.strip()
                if not l.startswith('['):
                    break
                try:
                    key, rest = l[1:-1].split(' ', 1)
                    value = rest.strip().strip('"')
                    headers[key] = value
                except Exception:
                    continue
            return headers

        def is_standard_start(headers):
            fen = headers.get("FEN", "") or ""
            setup = headers.get("SetUp", headers.get("Setup", "0")) or "0"
            if setup == "1" and fen:
                start_rank = fen.split()[0]
                return start_rank == "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"
            return True

        def is_variant(headers):
            variant = (headers.get("Variant", headers.get("VariantName", "standard")) or "standard").lower()
            event = (headers.get("Event", "") or "").lower()
            if any(k in variant for k in ("chess960", "fischer", "960", "crazyhouse", "atomic", "antichess", "horde")):
                return True
            if any(k in event for k in ("chess960", "fischer", "960")):
                return True
            return False

        def maybe_mark_chess960(block_text, headers):
            """If the FEN indicates a non-standard start but no variant tag, add a Variant "Chess960" header to enable castling parsing."""
            if is_variant(headers):
                return block_text
            if not is_standard_start(headers):
                lines = block_text.splitlines()
                # Insert Variant line before the blank line terminating headers
                for i, l in enumerate(lines):
                    if l.strip() == '':
                        lines.insert(i, '[Variant "Chess960"]')
                        return "\n".join(lines) + "\n"
                # If no explicit header terminator, append after last header
                last_header_idx = 0
                for i, l in enumerate(lines):
                    if l.startswith('['):
                        last_header_idx = i
                lines.insert(last_header_idx + 1, '[Variant "Chess960"]')
                return "\n".join(lines) + "\n"
            return block_text

        try:
            with open(filepath, 'rb') as compressed_file:
                dctx = zstd.ZstdDecompressor()
                with dctx.stream_reader(compressed_file) as reader:
                    text_stream = io.TextIOWrapper(reader, encoding='utf-8')
                    
                    for block in iter_pgn_blocks(text_stream):
                        if not (unlimited or (games_loaded < max_games)):
                            break
                        # Early header filtering
                        headers = parse_headers_from_block(block)
                        if not headers:
                            continue
                        games_processed += 1
                        if games_processed % 1000 == 0:
                            print(f"Processed {games_processed} games, loaded {games_loaded} quality games...")

                        try:
                            white_elo = int(headers.get("WhiteElo", "0"))
                            black_elo = int(headers.get("BlackElo", "0"))
                            if white_elo < min_elo or black_elo < min_elo:
                                continue
                            if bot_games_only:
                                white_title = headers.get("WhiteTitle", "")
                                black_title = headers.get("BlackTitle", "")
                                if white_title != "BOT" or black_title != "BOT":
                                    continue
                            # Skip explicit variants
                            if is_variant(headers):
                                continue
                            # If non-standard FEN start, mark as Chess960 to make castling SAN legal
                            if not is_standard_start(headers):
                                block = maybe_mark_chess960(block, headers)
                        except Exception:
                            continue

                        try:
                            game = chess.pgn.read_game(io.StringIO(block))
                            if game is None:
                                continue
                        except Exception:
                            games_skipped_parse += 1
                            continue

                        result = game.headers.get("Result", "*")
                        if result == "1-0":
                            game_outcome = 1.0
                        elif result == "0-1":
                            game_outcome = -1.0
                        elif result == "1/2-1/2":
                            game_outcome = 0.0
                        else:
                            continue

                        board = game.board()
                        move_count = 0
                        try:
                            for node in game.mainline():
                                move_count += 1
                                position_value = None
                                if node.comment:
                                    import re
                                    eval_match = re.search(r'\[%eval ([#\-\d.]+)\]', node.comment)
                                    if eval_match:
                                        eval_str = eval_match.group(1)
                                        if eval_str.startswith('#'):
                                            mate_moves = int(eval_str[1:])
                                            position_value = 10.0 if mate_moves > 0 else -10.0
                                        else:
                                            centipawns = float(eval_str)
                                            position_value = np.tanh(centipawns / 400.0)
                                if position_value is None:
                                    position_value = game_outcome if board.turn == chess.WHITE else -game_outcome
                                    weight = min(move_count / 40.0, 1.0)
                                    position_value = position_value * (0.3 + 0.7 * weight)
                                else:
                                    if board.turn == chess.BLACK:
                                        position_value = -position_value
                                try:
                                    self.positions.append(self.encoder.encode_board(board))
                                    self.outcomes.append(position_value)
                                except Exception:
                                    pass
                                if node.move:
                                    board.push(node.move)
                        except ValueError:
                            games_skipped_parse += 1
                            continue
                        games_loaded += 1
                    
        except Exception as e:
            print(f"\nError loading PGN.zst: {e}")
            import traceback
            traceback.print_exc()
        if games_skipped_parse:
            print(f"Skipped {games_skipped_parse} games due to parse errors (malformed SAN)")
        return games_loaded
    
    def __len__(self):
        return len(self.positions)
    
    def __getitem__(self, idx):
        position = torch.from_numpy(self.positions[idx]).float()
        outcome = torch.tensor([self.outcomes[idx]], dtype=torch.float32)
        return position, outcome


def train(epochs=15, batch_size=256, learning_rate=0.001, max_games=0, min_elo=1600, source="standard"):
    print("=" * 80)
    print("HYBRID CHESS BOT - TRAINING THE NEURAL EVALUATOR")
    print("=" * 80)
    
    # Check if PyTorch and dependencies are available
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        print("\nWARNING: PyTorch not installed. Skipping neural network training.")
        print("The bot will use material evaluation as fallback.")
        print("To train the neural network, install: pip install torch numpy zstandard")
        return
    
    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM Available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Using March 2015 Lichess database
    database_urls = ["https://database.lichess.org/standard/lichess_db_standard_rated_2015-03.pgn.zst"]
    print(f"\n📥 Using Lichess database (March 2015)...")
    for u in database_urls:
        print(f" - {u}")
    print(f"Quality filter: Players rated {min_elo}+ only")
    unlimited = (max_games is None) or (isinstance(max_games, int) and max_games <= 0)
    target_label = 'UNLIMITED' if unlimited else str(max_games)
    print(f"Target games: {target_label}")
    
    # Load dataset
    try:
        dataset = ChessGameDataset(
            pgn_zst_urls=database_urls,
            max_games=max_games,
            min_elo=min_elo
        )
        if unlimited and len(dataset.positions) > 0 and len(dataset.outcomes) > 0 and len(dataset.positions) == len(dataset.outcomes):
            print(f"Loaded {len(dataset)} total positions across {len(database_urls)} archives (unlimited mode)")
        
        if len(dataset) == 0:
            print("No positions loaded. Training aborted.")
            return
        
        # Split into train/validation
        train_size = int(0.9 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        
        print(f"\nTraining set: {train_size} positions")
        print(f"Validation set: {val_size} positions")
        print(f"Batch size: {batch_size}")
        
    except Exception as e:
        print(f"\nError loading dataset: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Initialize model
    model = CompactChessNet(num_filters=64).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)
    
    # Training loop
    print("\n" + "=" * 80)
    print("TRAINING")
    print("=" * 80)
    
    import time
    start_time = time.time()
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for batch_idx, (positions, outcomes) in enumerate(train_loader):
            positions = positions.to(device)
            outcomes = outcomes.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            predictions = model(positions)
            if isinstance(predictions, (tuple, list)):
                predictions = predictions[0]
            loss = criterion(predictions, outcomes)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            if batch_idx % 50 == 0:
                elapsed = time.time() - start_time
                print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(train_loader)}, "
                      f"Loss: {loss.item():.4f}, Time: {elapsed/60:.1f}min")
        
        train_loss /= len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for positions, outcomes in val_loader:
                positions = positions.to(device)
                outcomes = outcomes.to(device)
                predictions = model(positions)
                if isinstance(predictions, (tuple, list)):
                    predictions = predictions[0]
                loss = criterion(predictions, outcomes)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        elapsed = time.time() - start_time
        print(f"\nEpoch {epoch+1}/{epochs}")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}")
        print(f"  LR:         {optimizer.param_groups[0]['lr']:.6f}")
        print(f"  Time:       {elapsed/60:.1f} minutes")
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_path = os.path.join(os.path.dirname(__file__), "model_weights.pth")
            torch.save(model.state_dict(), model_path)
            print(f"  ✓ Saved best model to {model_path}")
        
        print()
    
    total_time = time.time() - start_time
    print("=" * 80)
    print("TRAINING COMPLETE")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Total time: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
    print(f"Estimated time on RTX 3090: ~{(total_time/60) * 0.4:.0f} minutes")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Hybrid Chess Bot neural evaluator")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--max-games", type=int, default=0, help="0 or negative means unlimited")
    parser.add_argument("--min-elo", type=int, default=2000)
    parser.add_argument("--source", type=str, default="standard", choices=["standard", "broadcast"], help="Data source: standard rated games or broadcast snapshots")
    args = parser.parse_args()

    mg = None if args.max_games <= 0 else args.max_games

    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_games=mg,
        min_elo=args.min_elo,
        source=args.source,
    )