No external API integration: this task adds no vendor capability — it fixes a CLI-side
zero-success guard, makes an existing Zarr conversion an explicit opt-in, and makes an
existing local roster query order-deterministic. The Tiingo/Alpaca capability surface is
unchanged; no request shape, endpoint or vendor method is added, removed or altered.
