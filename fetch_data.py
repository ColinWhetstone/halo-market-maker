
# Imports needed for connecting to Kraken and printing clean output.
import ccxt


def main() -> None:
    # Create a public Kraken exchange client (no API keys required for public market data).
    exchange = ccxt.kraken(
        {
            "enableRateLimit": True,
        }
    )

    # Define the market symbol and request the live order book snapshot.
    symbol = "ALGO/USD"
    order_book = exchange.fetch_order_book(symbol)

    # Extract and print the top 5 asks and bids with their price and volume.
    asks = order_book.get("asks", [])[:5]
    bids = order_book.get("bids", [])[:5]

    print(f"Kraken order book: {symbol}")

    print("\nTop 5 asks (price, volume):")
    for row in asks:
        price, volume = row[0], row[1]
        print(f"{price:>12.8f}  {volume:>12.8f}")

    print("\nTop 5 bids (price, volume):")
    for row in bids:
        price, volume = row[0], row[1]
        print(f"{price:>12.8f}  {volume:>12.8f}")


if __name__ == "__main__":
    # Run the script entrypoint.
    main()

