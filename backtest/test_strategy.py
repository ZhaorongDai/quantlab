import datetime
from typing import Any

import torch
import xarray as xr
from nautilus_trader.config import PositiveFloat, StrategyConfig
from nautilus_trader.core.data import Data
from nautilus_trader.core.datetime import unix_nanos_to_dt
from nautilus_trader.core.message import Event
from nautilus_trader.model.book import OrderBook
from nautilus_trader.model.data import (
    Bar,
    BarType,
    OrderBookDeltas,
    QuoteTick,
    TradeTick,
)
from nautilus_trader.model.enums import OrderSide, TimeInForce
from nautilus_trader.model.identifiers import ExecAlgorithmId, InstrumentId
from nautilus_trader.model.instruments import Instrument
from nautilus_trader.model.orders import MarketOrder
from nautilus_trader.trading.strategy import Strategy

from config import alpha101_config, alpha158_config
from dl_model.rnn_classification import ModelRCrypto
from factor import alpha101, alpha158


class TestConfig(StrategyConfig, frozen=True):
    """
    Configuration for ``Test`` instances.

    Parameters
    ----------
    instrument_id : InstrumentId
        The instrument ID for the strategy.
    bar_type : BarType
        The bar type for the strategy.
    position_size_pct : float
        The position size as percentage of available balance (0.0-1.0).
    model_path : str
        The path to the model.
    prediction_horizon : int, default 1
        The prediction horizon in minutes (n minutes ahead).
    confidence_threshold : float, default 0.6
        The minimum confidence threshold for trading decisions.
    max_position_pct : float, default 0.2
        The maximum total position size as percentage of balance.
    twap_horizon_secs : PositiveFloat, default 30.0
        The TWAP horizon (seconds) over which the algorithm will execute.
    twap_interval_secs : PositiveFloat, default 3.0
        The TWAP interval (seconds) between orders.
    close_positions_on_stop : bool, default True
        If all open positions should be closed on strategy stop.

    """

    instrument_id: InstrumentId
    bar_type: BarType
    model_path: str
    position_size_pct: float = 0.1  # 10% of available balance per trade
    prediction_horizon: int = 1  # Predict n minutes ahead
    confidence_threshold: float = 0.51  # Minimum confidence for trading
    max_position_pct: float = 0.2  # Max 20% total position
    twap_horizon_secs: PositiveFloat = 30.0
    order_quantity_precision = None
    twap_interval_secs: PositiveFloat = 3.0
    close_positions_on_stop: bool = True


class Test(Strategy):
    """
    A simple moving average cross example strategy.

    When the fast EMA crosses the slow EMA then enter a position at the market
    in that direction.

    Cancels all orders and closes all positions on stop.

    Parameters
    ----------
    config : EMACrossConfig
        The configuration for the instance.

    Raises
    ------
    ValueError
        If `config.fast_ema_period` is not less than `config.slow_ema_period`.
    ValueError
        If `config.twap_interval_secs` is not less than or equal to `config.twap_horizon_secs`.

    """

    def __init__(self, config: TestConfig) -> None:
        super().__init__(config)

        self.alpha101 = alpha101.Alpha101SpotKline(
            alpha101_config(mode="batch", symbols=["BTCUSDT"])
        )
        self.alpha158 = alpha158.Alpha158SpotKline(
            alpha158_config(mode="batch", symbols=["BTCUSDT"])
        )

        self.device = torch.device(
            "cpu"
        )
        num_features = self.alpha158.num_factors + self.alpha101.num_factors
        """
        hidden_sizes=[256, 128, 64],
        dropout_rates=[0.1, 0.1, 0.1],
        hidden_sizes_linear=[32],
        dropout_rates_linear=[0.1],
        """
        self.model = ModelRCrypto(
            input_size=num_features,
            num_labels=3,
            hidden_sizes=[256, 128, 64],
            dropout_rates=[0.1, 0.1, 0.1],
            hidden_sizes_linear=[32],
            dropout_rates_linear=[0.1],
            model_type="gru",
        ).to(self.device)
        self.model.load_state_dict(torch.load(self.config.model_path))

        self.instrument: Instrument = None  # Initialized in on_start

        # Prediction tracking
        self.predictions_history = []  # Store predictions with timestamps
        self.last_prediction_time = None
        self.current_prediction = None
        self.current_confidence = 0.0

        # Position management
        self.target_position = (
            0.0  # Target position (-1 for short, 0 for flat, 1 for long)
        )
        self.last_trade_time = None

        # Order management
        self.twap_exec_algorithm_id = ExecAlgorithmId("TWAP")
        self.twap_exec_algorithm_params: dict[str, Any] = {
            "horizon_secs": config.twap_horizon_secs,
            "interval_secs": config.twap_interval_secs,
        }

    def on_start(self) -> None:
        """
        Actions to be performed on strategy start.
        """
        self.instrument = self.cache.instrument(self.config.instrument_id)
        if self.instrument is None:
            self.log.error(
                f"Could not find instrument for {self.config.instrument_id}"
            )
            self.stop()
            return

        # Get historical data
        # self.request_bars(
        #     BarType.from_str("BTCUSDT.BINANCE-1-MINUTE-LAST-EXTERNAL"),
        #     start=datetime.datetime(2023, 1, 1, tzinfo=UTC),
        # )
        # Subscribe to live data
        self.subscribe_bars(self.config.bar_type)

    def on_instrument(self, instrument: Instrument) -> None:
        """
        Actions to be performed when the strategy is running and receives an instrument.

        Parameters
        ----------
        instrument : Instrument
            The instrument received.

        """
        # For debugging (must add a subscription)
        # self.log.info(repr(instrument), LogColor.CYAN)

    def on_order_book_deltas(self, deltas: OrderBookDeltas) -> None:
        """
        Actions to be performed when the strategy is running and receives order book
        deltas.

        Parameters
        ----------
        deltas : OrderBookDeltas
            The order book deltas received.

        """
        # For debugging (must add a subscription)
        # self.log.info(repr(deltas), LogColor.CYAN)

    def on_order_book(self, order_book: OrderBook) -> None:
        """
        Actions to be performed when the strategy is running and receives an order book.

        Parameters
        ----------
        order_book : OrderBook
            The order book received.

        """
        # For debugging (must add a subscription)
        # self.log.info(repr(order_book), LogColor.CYAN)

    def on_quote_tick(self, tick: QuoteTick) -> None:
        """
        Actions to be performed when the strategy is running and receives a quote tick.

        Parameters
        ----------
        tick : QuoteTick
            The tick received.

        """
        # For debugging (must add a subscription)
        # self.log.info(repr(tick), LogColor.CYAN)

    def on_trade_tick(self, tick: TradeTick) -> None:
        """
        Actions to be performed when the strategy is running and receives a trade tick.

        Parameters
        ----------
        tick : TradeTick
            The tick received.

        """
        # For debugging (must add a subscription)
        # self.log.info(repr(tick), LogColor.CYAN)
        #

    def on_historical_data(self, data):
        pass
        # if isinstance(data, Bar):
        # self.log.info(
        #     f"Waiting for factor warm up {unix_nanos_to_dt(data.ts_event)}"
        # )
        # bar = data
        # open = bar.open.as_double()
        # high = bar.high.as_double()
        # low = bar.low.as_double()
        # close = bar.close.as_double()
        # volume = bar.volume.as_double()
        # amount = volume * close
        # timestamp = bar.ts_event
        # symbol = [self.instrument.id]
        # input = {
        #     "open": np.array([open], dtype=np.float32),
        #     "high": np.array([high], dtype=np.float32),
        #     "low": np.array([low], dtype=np.float32),
        #     "close": np.array([close], dtype=np.float32),
        #     "volume": np.array([volume], dtype=np.float32),
        #     "amount": np.array([amount], dtype=np.float32),
        # }
        # self.alpha101.cal_stream(input, timestamp, symbol)
        # self.alpha158.cal_stream(input, timestamp, symbol)

    def on_bar(self, bar: Bar) -> None:
        """
        Actions to be performed when the strategy is running and receives a bar.

        Parameters
        ----------
        bar : Bar
            The bar received.

        """

        if bar.is_single_price():
            # Implies no market information for this bar
            return

        current_time = unix_nanos_to_dt(bar.ts_event)

        # Generate prediction
        prediction_result = self._generate_prediction(bar)
        if prediction_result is None:
            return

        predicted_class, confidence = prediction_result

        # Store prediction with timestamp
        self.predictions_history.append(
            {
                "timestamp": current_time,
                "prediction": predicted_class,
                "confidence": confidence,
                "target_time": current_time
                + datetime.timedelta(minutes=self.config.prediction_horizon),
            }
        )

        # Check if it's time to act on previous predictions
        self._check_and_execute_predictions(current_time)

        # Update current prediction for logging
        self.current_prediction = predicted_class
        self.current_confidence = confidence

        self.log.info(
            f"Prediction: {predicted_class}, Confidence: {confidence:.3f}, "
            f"Current Position: {self._get_current_position_pct():.3f}"
        )

    def _generate_prediction(self, bar: Bar) -> tuple[int, float] | None:
        """Generate model prediction from bar data."""

        current_time = unix_nanos_to_dt(bar.ts_event)
        # open_price = bar.open.as_double()
        # high = bar.high.as_double()
        # low = bar.low.as_double()
        # close = bar.close.as_double()
        # volume = bar.volume.as_double()
        # amount = volume * close
        # timestamp = bar.ts_event
        # symbol = [self.instrument.id]

        # input_data = {
        #     "open": np.array([open_price], dtype=np.float32),
        #     "high": np.array([high], dtype=np.float32),
        #     "low": np.array([low], dtype=np.float32),
        #     "close": np.array([close], dtype=np.float32),
        #     "volume": np.array([volume], dtype=np.float32),
        #     "amount": np.array([amount], dtype=np.float32),
        # }

        # alpha101 = self.alpha101.cal_stream(
        #     input_data, timestamp, symbol
        # ).get_features()
        # alpha158 = self.alpha158.cal_stream(
        #     input_data, timestamp, symbol
        # ).get_features()
        alpha101 = (
            self.alpha101.read()
            .get_features()
            .sel(timestamp=current_time.to_datetime64())
        )
        alpha158 = (
            self.alpha158.read()
            .get_features()
            .sel(timestamp=current_time.to_datetime64())
        )
        features = xr.combine_by_coords([alpha101, alpha158])

        data = (
            torch.from_numpy(
                features.to_dataarray()
                .fillna(0)
                .transpose("symbol", "variable")
                .sortby(["symbol", "variable"])
                .values
            )
            .unsqueeze(0)
            .to(self.device)
        )

        # Get model prediction (binary classification)
        primary_pred, _ = self.model(data)

        # Apply softmax to get probabilities
        pred_probs = torch.softmax(primary_pred, dim=-1)
        pred_class = torch.argmax(pred_probs, dim=-1).detach().cpu().item()
        confidence = torch.max(pred_probs, dim=-1)[0].detach().cpu().item()

        return pred_class, confidence

    def _check_and_execute_predictions(
        self, current_time: datetime.datetime
    ) -> None:
        """Check if any predictions should be executed and execute trades."""
        if not self.predictions_history:
            return

        # Find predictions that should be executed now
        ready_predictions = [
            pred
            for pred in self.predictions_history
            if pred["target_time"] <= current_time
            and pred["confidence"] >= self.config.confidence_threshold
        ]

        if not ready_predictions:
            return

        # Get the most recent ready prediction
        latest_prediction = max(ready_predictions, key=lambda x: x["timestamp"])

        # Remove executed predictions
        self.predictions_history = [
            pred
            for pred in self.predictions_history
            if pred["target_time"] > current_time
        ]

        # Execute trade based on prediction
        self._execute_trade_decision(latest_prediction)

    def _execute_trade_decision(self, prediction: dict) -> None:
        """Execute trade based on prediction."""
        predicted_class = prediction["prediction"]
        confidence = prediction["confidence"]

        # Determine target position based on prediction
        # 0 = down (short), 1 = up (long)
        if predicted_class == 1:  # Predicted up
            target_position = confidence  # Position size based on confidence
        else:  # Predicted down
            target_position = -confidence  # Short position based on confidence

        self._adjust_position_to_target(target_position)

    def _adjust_position_to_target(self, target_position: float) -> None:
        """Adjust current position to target position."""
        current_pos_pct = self._get_current_position_pct()

        # Check if we need to trade
        position_diff = target_position - current_pos_pct

        if abs(position_diff) < 0.01:  # Less than 1% difference
            return

        # Check maximum position limits
        max_pos = self.config.max_position_pct
        target_position = max(-max_pos, min(max_pos, target_position))

        if target_position > current_pos_pct:
            # Need to buy (go long or reduce short)
            self._execute_buy(target_position - current_pos_pct)
        elif target_position < current_pos_pct:
            # Need to sell (go short or reduce long)
            self._execute_sell(current_pos_pct - target_position)

    def _get_current_position_pct(self) -> float:
        """Get current position as percentage of balance."""
        if self.portfolio.is_flat(self.config.instrument_id):
            return 0.0

        # Get positions for this instrument from cache
        positions = self.cache.positions_open(
            venue=None, instrument_id=self.config.instrument_id
        )

        if not positions:
            return 0.0

        # Get account balance using portfolio API
        try:
            # Get account using portfolio.account() method
            account = self.portfolio.account(self.config.instrument_id.venue)
            if account is None:
                self.log.warning(
                    "No account found for venue, using default balance"
                )
                balance = 10000.0  # Default balance
            else:
                # Try to get balance from account
                balances = account.balances()
                if balances:
                    # Sum all currency balances (balances is dict[Currency, AccountBalance])
                    total_balance = 0.0
                    for currency, account_balance in balances.items():
                        # AccountBalance has total, locked, free attributes (all Money objects)
                        total_balance += account_balance.total.as_double()
                    balance = total_balance if total_balance > 0 else 10000.0
                else:
                    balance = 10000.0  # Default balance if no balances

        except Exception as e:
            self.log.warning(
                f"Error getting portfolio balance: {e}, using default"
            )
            balance = 10000.0  # Default fallback balance
        if balance <= 0:
            return 0.0

        # Calculate total position value
        total_position_value = 0.0
        position_side = None

        for position in positions:
            # Get current price for notional value calculation
            try:
                last_bar = self.cache.bar(self.config.bar_type)
                if last_bar is not None:
                    current_price = last_bar.close
                    notional_value = position.notional_value(current_price)
                    position_value = abs(notional_value.as_double())
                else:
                    # Fallback: use quantity * average open price
                    position_value = abs(
                        position.quantity.as_double()
                        * position.avg_px_open.as_double()
                    )
            except Exception:
                # Fallback: use quantity * average open price
                position_value = abs(
                    position.quantity.as_double()
                    * position.avg_px_open.as_double()
                )

            total_position_value += position_value

            # Determine if it's long or short (use the first position's side)
            if position_side is None:
                position_side = position.side

        if total_position_value == 0:
            return 0.0

        position_pct = total_position_value / balance

        # Return negative for short positions
        if position_side and position_side.name == "SHORT":
            return -position_pct
        else:
            return position_pct

    def _execute_buy(self, position_pct: float) -> None:
        """
        Execute buy order with percentage-based position sizing.

        Parameters
        ----------
        position_pct : float
            The percentage of balance to allocate to this position.
        """
        if position_pct <= 0:
            return

        # Get account balance using portfolio API
        try:
            # Get account using portfolio.account() method
            account = self.portfolio.account(self.config.instrument_id.venue)
            if account is None:
                self.log.warning(
                    "No account found for venue, using default balance"
                )
                balance = 10000.0  # Default balance
            else:
                # Try to get balance from account
                balances = account.balances()
                if balances:
                    # Sum all currency balances (balances is dict[Currency, AccountBalance])
                    total_balance = 0.0
                    for currency, account_balance in balances.items():
                        # AccountBalance has total, locked, free attributes (all Money objects)
                        total_balance += account_balance.total.as_double()
                    balance = total_balance if total_balance > 0 else 10000.0
                else:
                    balance = 10000.0  # Default balance if no balances

        except Exception as e:
            self.log.warning(
                f"Error getting portfolio balance: {e}, using default"
            )
            balance = 10000.0  # Default fallback balance
        if balance <= 0:
            self.log.warning("No available balance for buying")
            return

        # Calculate position size
        position_value = balance * min(
            position_pct, self.config.position_size_pct
        )

        try:
            # Get current market price from latest bar or quote
            instrument = self.cache.instrument(self.config.instrument_id)
            if instrument is None:
                self.log.warning("Instrument not found for buy order")
                return

            # Try to get last price from cache
            last_bar = self.cache.bar(self.config.bar_type)
            if last_bar is not None:
                current_price = last_bar.close
            else:
                self.log.warning("No price data available for buy order")
                return

            quantity = position_value / current_price.as_double()
            order_qty = instrument.make_qty(quantity)

            order: MarketOrder = self.order_factory.market(
                instrument_id=self.config.instrument_id,
                order_side=OrderSide.BUY,
                quantity=order_qty,
                time_in_force=TimeInForce.GTC,
            )

            self.submit_order(order)
            self.log.info(
                f"Submitted BUY order: {order_qty} at ~{current_price}"
            )

        except Exception as e:
            self.log.error(f"Error executing buy order: {e}")

    def _execute_sell(self, position_pct: float) -> None:
        """
        Execute sell order with percentage-based position sizing.

        Parameters
        ----------
        position_pct : float
            The percentage of balance to allocate to this short position.
        """
        if position_pct <= 0:
            return

        # Get account balance using portfolio API
        try:
            # Get account using portfolio.account() method
            account = self.portfolio.account(self.config.instrument_id.venue)
            if account is None:
                self.log.warning(
                    "No account found for venue, using default balance"
                )
                balance = 10000.0  # Default balance
            else:
                # Try to get balance from account
                balances = account.balances()
                if balances:
                    # Sum all currency balances (balances is dict[Currency, AccountBalance])
                    total_balance = 0.0
                    for currency, account_balance in balances.items():
                        # AccountBalance has total, locked, free attributes (all Money objects)
                        total_balance += account_balance.total.as_double()
                    balance = total_balance if total_balance > 0 else 10000.0
                else:
                    balance = 10000.0  # Default balance if no balances

        except Exception as e:
            self.log.warning(
                f"Error getting portfolio balance: {e}, using default"
            )
            balance = 10000.0  # Default fallback balance
        if balance <= 0:
            self.log.warning("No available balance for selling")
            return

        # Calculate position size
        position_value = balance * min(
            position_pct, self.config.position_size_pct
        )

        try:
            # Get current market price from latest bar or quote
            instrument = self.cache.instrument(self.config.instrument_id)
            if instrument is None:
                self.log.warning("Instrument not found for sell order")
                return

            # Try to get last price from cache
            last_bar = self.cache.bar(self.config.bar_type)
            if last_bar is not None:
                current_price = last_bar.close
            else:
                self.log.warning("No price data available for sell order")
                return

            quantity = position_value / current_price.as_double()
            order_qty = instrument.make_qty(quantity)

            order: MarketOrder = self.order_factory.market(
                instrument_id=self.config.instrument_id,
                order_side=OrderSide.SELL,
                quantity=order_qty,
                time_in_force=TimeInForce.GTC,
            )

            self.submit_order(order)
            self.log.info(
                f"Submitted SELL order: {order_qty} at ~{current_price}"
            )

        except Exception as e:
            self.log.error(f"Error executing sell order: {e}")

    def buy(self) -> None:
        """
        Legacy buy method for backward compatibility.
        """
        self._execute_buy(self.config.position_size_pct)

    def sell(self) -> None:
        """
        Legacy sell method for backward compatibility.
        """
        self._execute_sell(self.config.position_size_pct)

    def on_data(self, data: Data) -> None:
        """
        Actions to be performed when the strategy is running and receives data.

        Parameters
        ----------
        data : Data
            The data received.

        """

    def on_event(self, event: Event) -> None:
        """
        Actions to be performed when the strategy is running and receives an event.

        Parameters
        ----------
        event : Event
            The event received.

        """

    def on_stop(self) -> None:
        """
        Actions to be performed when the strategy is stopped.
        """
        self.cancel_all_orders(self.config.instrument_id)
        if self.config.close_positions_on_stop:
            self.close_all_positions(self.config.instrument_id)

        # Unsubscribe from data
        self.unsubscribe_bars(self.config.bar_type)

    def on_reset(self) -> None:
        """
        Actions to be performed when the strategy is reset.
        """

    def on_save(self) -> dict[str, bytes]:
        """
        Actions to be performed when the strategy is saved.

        Create and return a state dictionary of values to be saved.

        Returns
        -------
        dict[str, bytes]
            The strategy state dictionary.

        """
        return {}

    def on_load(self, state: dict[str, bytes]) -> None:
        """
        Actions to be performed when the strategy is loaded.

        Saved state values will be contained in the give state dictionary.

        Parameters
        ----------
        state : dict[str, bytes]
            The strategy state dictionary.

        """

    def on_dispose(self) -> None:
        """
        Actions to be performed when the strategy is disposed.

        Cleanup any resources used by the strategy here.

        """
