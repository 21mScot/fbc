def btc_mined_between(height_start, height_end, include_fees=False):
    subsidy = 0
    fees = 0
    for h in range(height_start, height_end + 1):
        era = h // 210_000
        block_subsidy_sats = 50_0000_0000 >> era  # exact subsidy in satoshis
        subsidy += block_subsidy_sats
        if include_fees:
            fees += 10_000_000  # fake 0.1 BTC fee per block
    total_sats = subsidy + fees
    return total_sats / 100_000_000.0

# ================== EDIT THESE ANYTIME ==================
height_start = 884914   # Nov 1, 2025
height_end   = 885922   # Nov 8, 2025
include_fees = False
# ========================================================

btc = btc_mined_between(height_start, height_end, include_fees)
blocks = height_end - height_start + 1

# === FIXED: Correct current block reward ===
era = height_start // 210_000
current_reward_sats = 50_0000_0000 >> era
current_reward = current_reward_sats / 100_000_000

# Terminal output
print("=" * 55)
print(f"Bitcoin mined from block {height_start:,} → {height_end:,}")
print(f"Total blocks mined : {blocks:,}")
print(f"Total BTC          : {btc:,.8f} BTC")
print(f"Current block reward: {current_reward:.8f} BTC")  # Now shows 3.12500000
print("=" * 55)

# HTML output
html_output = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Bitcoin Mined Calculator</title>
    <style>
        body {{ font-family: Arial, sans-serif; padding: 40px; background: #fff8f0; }}
        h1 {{ color: #f7931a; }}
        .big {{ font-size: 36px; font-weight: bold; color: #00008b; }}
    </style>
</head>
<body>
    <h1>Bitcoin Mined Calculator</h1>
    <p>From block <strong>{height_start:,}</strong> to <strong>{height_end:,}</strong></p>
    <p>Total blocks: <strong>{blocks:,}</strong></p>
    <p class="big">{btc:,.8f} BTC</p>
    <p>Fees included: <strong>{include_fees}</strong></p>
    <p>Current block reward: <strong>{current_reward:.8f} BTC</strong></p>
    <hr>
    <small>Generated: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</small>
</body>
</html>
"""
with open('output.html', 'w') as f:
    f.write(html_output)

print("\nHTML saved → double-click 'output.html' to open in browser!")