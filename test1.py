import logging
import typing
from mlthon.api.istrategy import IStrategy
from mlthon.api.istrat_env import IStratEnv, MDSub
from mlthon import TickUnit
from mlthon.basics.defs import Exchange, Side, OrderType, TIF, ExecInstructions, GtwRejectCode, CancelCode, \
    GtwStatus, FeedHandlerStatus
from mlthon.basics.price import Price
from mlthon.basics.qty import Qty
from mlthon.basics.tick_format import TickFormat
from mlthon.order.order import Order
from mlthon.order.order_mgr import OrderMgr
from mlthon.order.books.level_book import LevelBook

from datetime import timedelta
from datetime import datetime

import numpy as np
import math
import statistics

import pandas as pd
from tabulate import tabulate


class Gradient3(IStrategy):

    def __init__(self, cfg: dict):

        self._order_mgr_ = OrderMgr("bln")
        self.log_ = logging.getLogger(__name__)
        self.log_.info("OHLC constructor called!")
        # self.env_ = env  # type: IStratEnv
        self.log_.info("Finished populating env")
        self.log_.info("--------------------------------------------")
        # self.log_.info(env)
        # self.env_ = None

        self.num_4hcandles = 15

        self.bigger_candles = ["5Min", "1H", "4H", "1D"]

        self.intermediate_df = pd.DataFrame()

        self.just_candles = {"5Min": pd.DataFrame(), "1H": pd.DataFrame(), "4H": pd.DataFrame(), "1D": pd.DataFrame()}
        self.signals_dataframe = {"5Min": pd.DataFrame(), "1H": pd.DataFrame(), "4H": pd.DataFrame(),
                                  "1D": pd.DataFrame()}

        self.candle_df = pd.DataFrame()
        self.trade_prices = []
        self.trade_times_prices_1s = []
        self.trade_times_prices_all = []

        self.open = []
        self.close = []
        self.high = []
        self.low = []

        self.ohlc = np.zeros((0, 4))

        self.features = []
        self.ema_medium_lag = []
        self.ema_long_lag = []
        self.short_tr = []
        self.medium_tr = []
        self.long_tr = []
        self.emaMedium_prev = None
        self.emaLong_prev = None
        self.atrShort_prev = None
        self.atrMedium_prev = None
        self.atrLong_prev = None
        self.isSignalOn = False
        self.tradeType = None
        self.isSignalCheck = False
        self.isWorkingOrder = False
        self.isTradeEntry = None
        self.hit_price = 0
        self.pos = float(0)
        self.fill_pos = 0
        self.net_notional = float(0)
        self.pnl = float(0)

        self.lvl_book_ = LevelBook()

        # self.daily_candles = self.candle_builder('1D')
        # self.fourhour_candles = self.candle_builder('240Min')

        # General self config
        max_pos = 100000
        hit_qty = 0.1
        if hit_qty <= 0:
            raise Exception("Hit Qty is not positive!")
        if max_pos <= 0:
            raise Exception("Max pos is not positive!")
        self.hit_qty = hit_qty
        self.max_pos = max_pos

        self.trade_times_prices = []

        lag = 10
        exitLossTicks = 100
        exitProfitTicks = 100
        ticks_away = 0.5
        active_order_ticks_through_self = 50

        min_order_size = 0.001

        atrShort_threshold = 5

        if lag <= 0:
            raise Exception("Signal.Lag is non-positive!")
        if exitLossTicks <= 0:
            raise Exception("Signal.ExitLossTicks is non-positive!")
        if exitProfitTicks <= 0:
            raise Exception("Signal.ExitProfitTicks is non-positive!")
        if ticks_away < 0:
            raise Exception("Signal.TicksAway is negative!")
        if active_order_ticks_through_self < 0:
            raise Exception("Active order ticks through self is negative!")

        if min_order_size <= 0:
            raise Exception("Signal.MinOrderSize is non-positive!")

        self.lag = lag
        self.exitLossTicks = exitLossTicks
        self.exitProfitTicks = exitProfitTicks
        self.ticks_away = ticks_away
        self.active_order_ticks_through_self = active_order_ticks_through_self
        self.min_order_size = min_order_size

        self.is_active_entry = True
        self.is_active_exit = True
        self.allow_momentum = True
        self.allow_reversion = True

        self.allow_sell_short = True

        self.medium_lag = 2 * self.lag
        self.long_lag = 3 * self.lag
        self.emaMediumAlpha = 2.0 / (self.medium_lag + 1)
        self.emaLongAlpha = 2.0 / (self.long_lag + 1)

        self.orders = []
        self.is_ready_for_orders = False
        self.current_status = None
        self.log_.info("Finished init of strategy")
        self._is_feed_handler_ready_ = False

    def on_framework_connected(self, env: IStratEnv):
        self.log_.info("Now connected to framework")
        # print(self.env)
        self.env_ = env
        self.env_.setup_timer(30000, self.on_recurring_thirty_secs)
        self.env_.login("grad_strat_1807")
        # self.env_.add_cmd_hook("get_BBO", self.get_BBO_command)
        self._exch_ = Exchange.Bybit
        self._products_ = ["BTC-USD"]
        self._symbol_ = self._products_[0]
        # self.env_.subscribe_to_market_data(self._exch_, self._products_, [MDSub.BBO])
        self.env_.subscribe_to_market_data(self._exch_, self._products_, [MDSub.PublicTrades])
        self.env_.request_execution_connection(self._exch_, self._products_)
        self.env_.publish_telegram("The api is now ready to be started :)")

    def on_framework_disconnected(self):
        self.log_.info("Framework is disconnected!!")

    def on_start(self, params: str):

        self.log_.info("on_start() called with params '" + params + "'")
        self.env_.setup_timer(3000, self.on_timer_elapsed)
        self.env_.setup_timer(1000, self.on_candle_builder)
        self.log_.info("Adding timer for check")
        self.env_.setup_timer(3000, self.on_check)

        self.env_.add_cmd_hook("python_new_order", self.sendManualOrderHook)
        self.env_.add_cmd_hook("python_cancel_all_orders", self.sendCancelAllOrdersHook)
        self.env_.add_cmd_hook("get_open_orders", self.getOpenOrdersHook)
        self.env_.add_cmd_hook("get_position", self.getPositionHook)
        self.env_.add_cmd_hook("get_ohlc", self.getOhlcHook)
        self.env_.add_cmd_hook("get_features", self.getFeaturesHook)
        self.env_.add_cmd_hook("get_current_status", self.getCurrentStatusHook)
        self.env_.add_cmd_hook("get_pnl", self.getPnLHook)
        self.env_.add_cmd_hook("debug", self.debugHook)

    def on_check(self):
        self.log_.info("Running check!")

    def on_stop(self, params: str):
        self.log_.info("on_stop() called with params '" + params + "'")
        self.env_.send_stop_ack()
        # add flatten out on stop

    # -----------------------------------------------------------------------------------
    # ----------------------------- Timer and Hook Callback -----------------------------
    # -----------------------------------------------------------------------------------
    def on_recurring_thirty_secs(self):
        if self._is_feed_handler_ready_:
            self.log_.info("MidPrice Ema: " + str(self._mid_ema_))
            pass

        return True  # This timer is recurring because the callback returned true.

    def on_timer_elapsed(self):
        # self.log_.info("recurring timer callback!")
        self.get_open_orders()
        # self.log_.info([str(order) for order in self.orders])
        num_orders = self._order_mgr_.get_num_open_orders()
        return True

    def on_candle_builder(self):
        interval_map_mins = {"4H": 240, "1H": 60, "30Min": 30, "15Min": 15, "5Min": 5, "1D": 24 * 60}
        num_candles_cache = {"5Min": 20, "1H": 20, "4H": 20, "1D": 30}
        agg_constants = {"open": "first", "high": "max", "low": "min", "close": "last"}

        # Take all trades, find last trade within minute, ship those to 1m dataframe

        if len(self.trade_times_prices) > 0:
            last_time = datetime.fromtimestamp(self.trade_times_prices[-1][-1] / 1000).second
            if last_time == 0:
                index_last_trade = -1
                self.log_.info("true")
                minute_trades = self.trade_times_prices[:index_last_trade]

                # clear cache of trades
                del self.trade_times_prices[:index_last_trade]

                # Create a single 1minute dataframe row from list of trades and append it to 1m dataframe
                raw_df = pd.DataFrame(minute_trades, columns=["Price", "DateTime"])

                # self.log_.info(raw_df.head(5))
                raw_df = raw_df.set_index(pd.to_datetime(raw_df['DateTime'], unit='ms'), drop=True)
                raw_df = raw_df.drop(columns=['DateTime'])
                df = raw_df.resample('1Min').ohlc()

                df.columns = [col[1] for col in df.columns]

                if self.intermediate_df.empty:
                    self.intermediate_df = df
                    self.log_.info("Not concatting for")
                else:
                    self.log_.info(f"Shape of pre df is {df.shape}")
                    if df.shape[0] != 0:
                        self.intermediate_df = pd.concat([self.intermediate_df, df])

                # self.log_.info(self.intermediate_df)
                # self.log_.info(f"Shape of intermediate_df after appending is {self.intermediate_df.shape}")

                for interval, dataframe in self.just_candles.items():

                    # Keep 1m dataframe only as long as we need it
                    if self.intermediate_df.shape[0] > num_candles_cache[interval] * interval_map_mins[interval]:
                        excess_candles = self.intermediate_df.shape[0] - self.num_candles_cache_4H * interval_map_mins[
                            interval]
                        self.intermediate_df = self.intermediate_df[excess_candles:]

                    # resample new minute dataframe to create new 'interval' dataframe
                    self.just_candles[interval] = self.intermediate_df.resample(interval).agg(agg_constants)

                    # self.calculate_signals(interval)
                # self.log_.info('----------------')
                # self.log_.info(self.just_candles["4H"])
                # self.log_.info('----------------')
                # self.log_.info(self.just_candles["1D"])

                self.five_min = self.gradients(self.just_candles['5Min']['close'], 9).iloc[-1]
                self.hourly_grad = self.gradients(self.just_candles['1H']['close'], 9).iloc[-1]
                self.fourhour_grad = self.gradients(self.just_candles['4H']['close'], 9).iloc[-1]
                self.daily_grad = self.gradients(self.just_candles['4H']['close'], 11).iloc[-1]

                self.log_.info(f"five_grad = {self.five_min}")
                self.log_.info(f"hour_grad = {self.hourly_grad}")
                self.log_.info(f"fourhour_grad = {self.fourhour_grad}")
                self.log_.info(f"daily_grad = {self.daily_grad}")

        return True

    def minute_elapsed(self):

        first_time = datetime.fromtimestamp(self.trade_times_prices[0][-1] / 1000).minute
        last_time = datetime.fromtimestamp(self.trade_times_prices[-1][-1] / 1000).minute

        if first_time != last_time:
            return True
        else:
            return False

    def find_index_last_trade(self):
        seconds = [datetime.fromtimestamp(x[-1] / 1000).second for x in self.trade_times_prices]
        # find index of last 1 second

    def update_recent_boundary(self):
        self.most_recent_boundary += timedelta(hours=4)

    def add_candle(self):
        ticks = pd.DataFrame(self.trade_times)
        new_row = ticks.resample('4H').ohlc()
        self.candle_df = self.candle_df.append(new_row)
        if self.candle_df.shape[0] > self.num_4hcandles:
            num_larger_than_required = self.candle_df.shape[0] - self.num_4hcandles
            self.candle_df = self.candle_df[num_larger_than_required:]

    def get_BBO_command(self, params: str):
        if self._invalid_market_:
            self.env_.publish_telegram("Market is invalid!!")
        else:
            self.env_.publish_telegram("BBO: " + self._best_bid_.to_str() + "@" + self._best_ask_.to_str())

    def candle_builder(self, window: str, lookback: int):
        df = pd.DataFrame(self.trade_prices)
        self.log_.info("-------------- Trade prices -------------")
        self.log_.info(self.trade_prices)
        if window == '1D':
            setattr(self, 'daily_ohlc')
        elif window == '240Min':
            setattr(self, 'fourhourly_ohlc')

    def volatility(self):
        self.daily_ohlc['average'] = (self.daily_ohlc['open'] + self.daily_ohlc['close'] + self.daily_ohlc['high'] +
                                      self.daily_ohlc['low']) / 4
        self.daily_ohlc = self.daily_ohlc.dropna(axis=0, how='any')
        self.daily_ohlc['log'] = np.log(self.daily_ohlc['av'] / self.daily_ohlc['av'].shift(1))
        self.daily_ohlc['vol'] = self.daily_ohlc['log'].rolling(30).std() * 100

    def cusum(self):
        pass

    def gradients(self, series, ema_period):
        ma = pd.Series.ewm(series, span=ema_period).mean()

        ma = ma * 10000 / series

        ma_dydx = pd.Series(ma).diff()
        ma_smooth = pd.Series.ewm(ma_dydx, span=ema_period).mean()

        ma2 = pd.Series(ma_smooth).diff()
        ma2_smooth = pd.Series.ewm(ma2, span=ema_period).mean()

        ma3 = pd.Series(ma2_smooth).diff()
        ma3_s = pd.Series.ewm(ma3, span=ema_period).mean()
        return ma3_s

    def _print_all_orders(self, prefix: str):
        all_orders = self._order_mgr_.get_open_orders()
        log_line = prefix
        for order in all_orders:
            log_line += "\n\t" + str(order)
        # self.log_.info(log_line)

    def get_open_orders(self):
        self.orders = self._order_mgr_.get_open_orders()
        return self.orders

    def debugHook(self, params: str):
        self.env_.publish_telegram("isSignalOn: " + str(self.isSignalOn) + ", isSignalCheck: " + str(
            self.isSignalCheck) + ", tradeType: " + str(self.tradeType) + ", isWorkingOrder: " + str(
            self.isWorkingOrder) + ", isTradeEntry: " + str(self.isTradeEntry))

    def getOpenOrdersHook(self, params: str):
        self.get_open_orders()

        num_orders = self._order_mgr_.get_num_open_orders()
        if num_orders > 0:
            self.env_.publish_telegram("Open orders: " + str([str(order) for order in self.orders]))

        else:
            self.env_.publish_telegram("No open orders")

    def getPositionHook(self, params: str):
        self.env_.publish_telegram("Position: " + str(self.pos))

    def getCurrentStatusHook(self, params: str):
        self.env_.publish_telegram("Current Status: " + str(self.current_status))

    def getPnLHook(self, params: str):
        self.env_.publish_telegram("PnL: " + str(self.pnl))

    def getOhlcHook(self, params: str):
        self.env_.publish_telegram(
            "```" + tabulate(pd.DataFrame(self.ohlc).tail(), tablefmt='fancy_grid', numalign="left") + "```")

    def getFeaturesHook(self, params: str):
        self.env_.publish_telegram("```" + tabulate(pd.DataFrame(self.features).transpose(), tablefmt='fancy_grid',
                                                    numalign="left") + "```")

    def sendManualOrderHook(self, params: str):
        price = 11052.14
        qty = 0.01

        self.sendOrder(qty=qty, price=price, side=Side.Sell, exec_instr=ExecInstructions.Unset,
                       reason="manual order")
        self._print_all_orders("After sending manual order:")

    def sendOrder(self, side: Side, price: Price, qty: Qty, reason: str, exchange=Exchange.Blockchain, symbol="BTC-USD",
                  exec_instr=ExecInstructions.Unset, ord_type=OrderType.Limit, tif=TIF.GTC):
        num_orders = self._order_mgr_.get_num_open_orders()
        if num_orders > 0:
            return
        # roudning qty down to nearest tick down, with a tick being 1 lot
        qty = TickFormat.qty_down_from_float(qty, tick_size=0.001, tick_unit=TickUnit.Lot)

        if side == Side.Buy:
            # rounding price to the nearest tick up, with a tick being 0.01
            price = TickFormat.price_up_from_float(price, tick_size=0.01, tick_unit=TickUnit.Cent)
        else:
            # rounding price to the nearest tick up, with a tick being 0.01
            price = TickFormat.price_down_from_float(price, tick_size=0.01, tick_unit=TickUnit.Cent)

        order = self._order_mgr_.prepare_new_order(symbol=symbol, side=side, price=price, qty=qty,
                                                   exec_instr=exec_instr, order_type=OrderType.Limit, tif=TIF.GTC,
                                                   nullable_attachment=reason)
        if order:
            self.new_order_clid = order.get_client_id()
            self.env_.send_new_order(client_id=self.new_order_clid, side=side, price=price, qty=qty,
                                     exec_instr=exec_instr, exchange=exchange, symbol=symbol, ord_type=ord_type,
                                     time_in_force=tif)

            self.env_.publish_telegram("Sent a new order from python with order_id: " + str(self.new_order_clid) +
                                       " reason: " + reason)
        else:
            self.env_.publish_telegram("Not able to send a new_order:" + str(self.new_order_clid))
            self.env_.publish_telegram("There is a pending cancel-all!")
        self.sent_qty = qty
        self._print_all_orders("After sending new order (client_id: " + self.new_order_clid + "):")

    def sendCancelAllOrdersHook(self, reason='cancel_all'):
        self.get_open_orders()
        for order in self.orders:
            order_id = order.get_client_id()
            ready_to_send_cancel = self._order_mgr_.prepare_cancel_order(client_id=order_id)
            if ready_to_send_cancel:
                self.env_.send_cancel_order(Exchange.Blockchain, "BTC-USD", order_id)
                self.env_.publish_telegram("Sent a cancel: " + str(order_id) + " reason:" + reason)
            else:
                self.env_.publish_telegram("Not able to send a cancel:" + str(order_id))

    def sendCancelIndividualOrder(self, order_id):
        ready_to_send_cancel = self._order_mgr_.prepare_cancel_order(client_id=order_id)
        if ready_to_send_cancel:
            self.env_.send_cancel_order(Exchange.Blockchain, "BTC-USD", order_id)
            self.env_.publish_telegram("Sent a cancel: " + str(order_id))
            self._print_all_orders("After sending cancel order (client_id: " + order_id + "):")
        else:
            self.env_.publish_telegram("Not able to send a cancel:" + str(order_id))

    # -----------------------------------------------------------------------------------
    # ------------------------------ Order Entry Callbacks ------------------------------
    # -----------------------------------------------------------------------------------
    def on_new_order_ack(self, exchange: Exchange, client_id: str, exchange_id: str, product: str, side: Side,
                         price: Price, qty: Qty, leaves_qty: Qty, order_type: OrderType, tif: TIF,
                         exec_instr: ExecInstructions):
        order = self._order_mgr_.get_order_with_client_id(client_id)
        if order:
            self.log_.info("NewOrderAck: order_id: " + client_id + ", acked_qty: " + str(order.get_pending_qty()))
            self._order_mgr_.apply_new_order_ack(client_id=client_id, exchange_id=exchange_id, price=price,
                                                 leaves_qty=order.get_pending_qty())
            self._print_all_orders("After new order ack (client_id: " + client_id + "):")

    def on_modify_order_ack(self, exchange: Exchange, client_id: str, new_price: Price, new_qty: Qty, leaves_qty: Qty):
        self._order_mgr_.apply_modify_order_ack(client_id=client_id, leaves_qty=leaves_qty)

    def on_cancel_order_ack(self, exchange: Exchange, client_id: str):
        self._order_mgr_.apply_cancel_order_ack(client_id=client_id)

    def on_cancel_all_ack(self, exchange: Exchange, symbol: str):
        self.log_.debug("on_cancel_all_ack() callback")

    def on_new_order_reject(self, exchange: Exchange, client_id: str, reject_code: GtwRejectCode, reject_reason: str):
        self._order_mgr_.apply_new_order_reject(client_id=client_id)
        # self.isWorkingOrder = True

    def on_modify_order_reject(self, exchange: Exchange, client_id: str, reject_code: GtwRejectCode,
                               reject_reason: str):
        self._order_mgr_.apply_modify_order_reject(client_id=client_id)

    def on_cancel_order_reject(self, exchange: Exchange, client_id: str, reject_code: GtwRejectCode,
                               reject_reason: str):
        self._order_mgr_.apply_cancel_order_reject(client_id=client_id)

    def on_cancel_all_reject(self, exchange: Exchange, symbol: str, reject_code: GtwRejectCode, reject_reason: str):
        self.log_.debug("on_cancel_all_reject() callback")

    def on_order_execution(self, exchange: Exchange, client_id: str, side: Side, price: Price, fill_qty: Qty,
                           leaves_qty: Qty, exec_ts: int, recv_ts: int):
        is_fully_executed = self._order_mgr_.apply_order_execution(client_id=client_id, fill_qty=fill_qty,
                                                                   nullable_leaves_qty=None)

        if is_fully_executed is not None:
            self.get_open_orders()
            fill_qty = fill_qty.to_float()
            fill_pos = fill_qty if side == Side.Buy else -fill_qty
            self.pos += fill_pos
            self.net_notional += -price.to_float * fill_pos
            if self.pos == 0:
                self.pnl = self.net_notional
            # self.log_.info([str(order) for order in self.orders])
            if is_fully_executed:
                self.isSignalCheck = False
                ## flip the isTradeEntry to start new entry or start exit if already entered.
                self.isTradeEntry = not self.isTradeEntry
                self.isSignalOn = False
                self.isWorkingOrder = False
            string_to_output = 'Filled: ' + side.name + ' ' + str(
                fill_qty) + ' @ ' + price.to_str() + ' vs Last Price: ' + str(self.last_price)
            self.log_.info(string_to_output)
            self.env_.publish_telegram(string_to_output)
            self._print_all_orders("After order execution (client_id: " + client_id + "):")

    def on_order_cancelled(self, exchange: Exchange, client_id: str, unsolicited: bool, engine_ts: int,
                           recv_ts: int, cancel_code: CancelCode, cancel_reason: str):
        self._order_mgr_.apply_order_cancelled(client_id=client_id)

    # -----------------------------------------------------------------------------------
    # ------------------------------ Account Info Callbacks -----------------------------
    # -----------------------------------------------------------------------------------
    def on_orders_info(self, exchange: Exchange, orders: typing.List[Order]):
        # self.log_.info("on_orders_info() callback")
        pass

    def on_wallet_info(self, exchange: Exchange, coin: str, total_balance: Qty, available_balance: Qty):
        # self.log_.info("on_wallet_info() callback")
        pass

    def on_position_info(self, exchange: Exchange, symbol: str, position: Qty):
        # self.log_.info("on_position_info() callback")
        pass

    def on_funding_info(self, exchange: Exchange, symbol: str, funding_rate: float, next_funding_ts: int):
        # self.log_.info("on_funding_info() callback")
        pass

    def on_account_info(self, exchange: Exchange, user_id: str):
        # self.log_.info("on_account_info() callback")
        pass

    def on_request_reject(self, exchange: Exchange, reject_code: GtwRejectCode, detail: str, rejected_rqst_type: str):
        # self.log_.info("on_request_reject() callback")
        pass

    # -----------------------------------------------------------------------------------
    # ------------------------------- Market Data Callbacks -----------------------------
    # -----------------------------------------------------------------------------------

    def on_public_trade(self, exchange: Exchange, symbol: str, side: Side, price: Price, qty: Qty, exec_ts: int,
                        recv_ts: int):
        if exchange == Exchange.Bybit or exchange == 'Bybit' or exchange == Bybit:
            price_float = price.to_float()
            self.trade_times_prices.append((price_float, recv_ts))
            self.last_price = price_float

    def on_add_price_level(self, exchange: Exchange, symbol: str, side: Side, price: Price, qty: Qty, recv_ts: int):
        pass  # self.log_.info("on_add_price_level() callback")

    def on_modify_price_level(self, exchange: Exchange, symbol: str, side: Side, price: Price, new_qty: Qty,
                              recv_ts: int):
        pass  # self.log_.info("on_modify_price_level() callback")

    def on_delete_price_level(self, exchange: Exchange, symbol: str, side: Side, price: Price, recv_ts: int):
        pass  # self.log_.info("on_delete_price_level() callback")

    def on_add_level(self, exchange: Exchange, symbol: str, side: Side, level_id: int, price: Price, qty: Qty,
                     recv_ts: int):
        # self.log_.info("on_add_level() callback")
        self.lvl_book_.add_lvl(level_id, side, price, qty)
        self.printBBO()

    def on_modify_level(self, exchange: Exchange, symbol: str, side: Side, level_id: int, new_qty: Qty, recv_ts: int):
        # self.log_.info("on_modify_level() callback")
        self.lvl_book_.modify_lvl_qty(level_id, new_qty)
        self.printBBO()

    def on_delete_level(self, exchange: Exchange, symbol: str, side: Side, level_id: int, recv_ts: int):
        pass  # self.log_.info("on_delete_level() callback")
        self.lvl_book_.delete_lvl(level_id)
        self.printBBO()

    def on_best_bid_level_update(self, exchange: Exchange, symbol: str, price: Price, qty: Qty, recv_ts: int):
        self.best_bid = price.to_float()

    #    self.log_.info("Best bid updated")

    def on_best_ask_level_update(self, exchange: Exchange, symbol: str, price: Price, qty: Qty, recv_ts: int):
        self.best_ask = price.to_float()

    def printBBO(self):
        (bid_updated, best_bid) = self.lvl_book_.get_best_bid_lvl()
        (ask_updated, best_ask) = self.lvl_book_.get_best_ask_lvl()
        if bid_updated or ask_updated:
            if best_bid:
                self.best_bid = best_bid.get_price().to_float()
            if best_ask:
                self.best_ask = best_ask.get_price().to_float()
            # self.log_.info("BBO: " + str(best_bid) + " | " + str(best_ask))

    # -----------------------------------------------------------------------------------
    # --------------------------- Gtw and Feed Status Callbacks -------------------------
    # -----------------------------------------------------------------------------------
    def on_gateway_status(self, exchange: Exchange, status: GtwStatus, detail: str):
        self.log_.info("on_gateway_status() callback")
        self.is_ready_for_orders = (status == GtwStatus.ReadyToTrade)

    def on_feed_handler_status(self, exchange: Exchange, status: FeedHandlerStatus, detail: str):
        self.log_.info("on_feed_handler_status() callback")
        if status == FeedHandlerStatus.SnapshotStart:
            self.lvl_book_.clear()

    def on_rate_limit_info(self, exchange: Exchange, rate_limit: int, rate_remaining: int, reset_ts: int):
        self.log_.info("on_rate_limit_info() callback")

    def on_unsupported_op(self, exchange: Exchange, unsupported_msg_type: str):
        self.log_.info("on_unsupported_op() callback")

    # -----------------------------------------------------------------------------------
    # --------------------------- Custom Functions -------------------------
    # -----------------------------------------------------------------------------------
    ### computes EMA
    def ema(self):
        if not math.isnan(self.ohlc[-1, 3]):
            if self.emaMedium_prev is None or self.emaLong_prev is None:
                return self.ohlc[-1, 3], self.ohlc[-1, 3]
            else:
                curr_emaMedium = self.emaMedium_prev + self.emaMediumAlpha * (self.ohlc[-1, 3] - self.emaMedium_prev)
                curr_emaLong = self.emaLong_prev + self.emaLongAlpha * (self.ohlc[-1, 3] - self.emaLong_prev)
                return round(curr_emaMedium, 10), round(curr_emaLong, 10)

    ## computes true range
    def true_range(self):
        if len(self.ohlc) > 1:
            # return max(high-low, abs(high-prev_close), abs(low-prev_close))
            return max((self.ohlc[-1, 1] - self.ohlc[-1, 2]), abs(self.ohlc[-1, 1] - self.ohlc[-2, 3]),
                       abs(self.ohlc[-1, 2] - self.ohlc[-2, 3]))
        else:
            return 0.0

    ## compute atr_short using true range
    def atr_short(self, lag):
        curr_short_atr = float('nan')
        if len(self.ohlc) < lag:
            self.short_tr.append(self.true_range())
        elif len(self.ohlc) == lag:
            curr_short_atr = statistics.mean(self.short_tr)
        else:
            curr_short_atr = (self.atrShort_prev * (lag - 1) + self.true_range()) / lag
        return round(curr_short_atr, 10)

    ## compute atr_medium using true range
    def atr_medium(self, lag):
        curr_medium_atr = float('nan')
        if len(self.ohlc) < lag:
            self.medium_tr.append(self.true_range())
        elif len(self.ohlc) == lag:
            curr_medium_atr = statistics.mean(self.medium_tr)
        else:
            curr_medium_atr = (self.atrMedium_prev * (lag - 1) + self.true_range()) / lag
        return round(curr_medium_atr, 10)

    ## compute atr_long using true range
    def atr_long(self, lag):
        ## lag changes for short, medium and high
        curr_long_atr = float('nan')
        if len(self.ohlc) < lag:
            self.long_tr.append(self.true_range())
        elif len(self.ohlc) == lag:
            curr_long_atr = statistics.mean(self.long_tr)
        else:
            curr_long_atr = (self.atrLong_prev * (lag - 1) + self.true_range()) / lag
        return round(curr_long_atr, 10)
