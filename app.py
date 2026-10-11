import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf

st.set_page_config(
    page_title="Alpari Analyzer Pro",
    page_icon="📊",
    layout="wide"
)

st.title("Alpari Analyzer Pro")
st.caption("تحلیل بازار | پرایس اکشن | مدیریت ریسک")

MARKETS = {
  "طلا (XAUUSD)": "XAUUSD=X",
    "یورو / دلار": "EURUSD=X",
    "پوند / دلار": "GBPUSD=X",
    "دلار / ین": "JPY=X",
    "بیت‌کوین": "BTC-USD",
    "اتریوم": "ETH-USD",
    "نقره": "SI=F",
    "نفت": "CL=F",
}

TIMEFRAMES = {
    "۵ دقیقه": ("5m", "5d"),
    "۱۵ دقیقه": ("15m", "5d"),
    "۳۰ دقیقه": ("30m", "5d"),
    "۱ ساعت": ("60m", "1mo"),
    "۱ روز": ("1d", "1y"),
    "۱ هفته": ("1wk", "5y"),
}

def get_data(symbol, interval, period):
    df = yf.download(
        symbol,
        interval=interval,
        period=period,
        progress=False,
        auto_adjust=True
    )

    if df is None or df.empty:
        return None

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    required = ["Open", "High", "Low", "Close"]
    if not all(c in df.columns for c in required):
        return None

    df = df[required].dropna()

    if len(df) < 50:
        return None

    return df


def analyze(df):
    close = df["Close"]
    high = df["High"]
    low = df["Low"]

    ema20 = close.ewm(span=20, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()

    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / 14, min_periods=14, adjust=False
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / 14, min_periods=14, adjust=False
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)

    atr = tr.ewm(
        alpha=1 / 14, min_periods=14, adjust=False
    ).mean()

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal_line = macd.ewm(span=9, adjust=False).mean()

    price = float(close.iloc[-1])
    e20 = float(ema20.iloc[-1])
    e50 = float(ema50.iloc[-1])
    rsi_now = float(rsi.iloc[-1])
    atr_now = float(atr.iloc[-1])

    if not np.isfinite([price, e20, e50, rsi_now, atr_now]).all():
        return None

    score = 0

    if e20 > e50:
        score += 1
    else:
        score -= 1

    if price > e20:
        score += 1
    else:
        score -= 1

    if macd.iloc[-1] > signal_line.iloc[-1]:
        score += 1
    else:
        score -= 1

    if 50 < rsi_now < 70:
        score += 1
    elif 30 < rsi_now < 50:
        score -= 1

    if score >= 3:
        decision = "خرید احتمالی"
        stop = price - 1.5 * atr_now
        targets = [
            price + atr_now,
            price + 2 * atr_now,
            price + 3 * atr_now
        ]
    elif score <= -3:
        decision = "فروش احتمالی"
        stop = price + 1.5 * atr_now
        targets = [
            price - atr_now,
            price - 2 * atr_now,
            price - 3 * atr_now
        ]
    else:
        decision = "انتظار"
        stop = None
        targets = []

    return {
        "price": price,
        "decision": decision,
        "score": score,
        "rsi": rsi_now,
        "ema20": e20,
        "ema50": e50,
        "atr": atr_now,
        "stop": stop,
        "targets": targets,
        "support": float(low.tail(50).min()),
        "resistance": float(high.tail(50).max())
    }


market = st.selectbox("انتخاب بازار", list(MARKETS.keys()))
timeframe = st.selectbox("تایم‌فریم", list(TIMEFRAMES.keys()))

interval, period = TIMEFRAMES[timeframe]

if st.button("تحلیل بازار", type="primary"):
    with st.spinner("در حال دریافت داده و تحلیل..."):
        try:
            df = get_data(MARKETS[market], interval, period)

            if df is None:
                st.error(
                    "داده کافی دریافت نشد. نماد یا تایم‌فریم دیگری انتخاب کن."
                )
            else:
                result = analyze(df)

                if result is None:
                    st.error("محاسبات معتبر نیستند؛ داده بیشتری لازم است.")
                else:
                    st.subheader(f"نتیجه تحلیل {market}")

                    col1, col2, col3 = st.columns(3)
                    col1.metric("قیمت آخرین کندل", f"{result['price']:.6g}")
                    col2.metric("تصمیم", result["decision"])
                    col3.metric("امتیاز روند", f"{result['score']} از ۴")

                    col1, col2, col3 = st.columns(3)
                    col1.metric("RSI", f"{result['rsi']:.2f}")
                    col2.metric("EMA 20", f"{result['ema20']:.6g}")
                    col3.metric("EMA 50", f"{result['ema50']:.6g}")

                    st.subheader("سطوح معاملاتی")

                    col1, col2 = st.columns(2)
                    col1.metric(
                        "حمایت تقریبی",
                        f"{result['support']:.6g}"
                    )
                    col2.metric(
                        "مقاومت تقریبی",
                        f"{result['resistance']:.6g}"
                    )

                    if result["stop"] is not None:
                        st.metric(
                            "حد ضرر پیشنهادی",
                            f"{result['stop']:.6g}"
                        )

                        for i, target in enumerate(
                            result["targets"], start=1
                        ):
                            st.write(
                                f"**هدف سود {i}:** {target:.6g}"
                            )

                    st.info(
                        "این خروجی سیگنال آزمایشی است، نه تضمین سود. "
                        "داده‌ها ممکن است با قیمت کارگزار متفاوت باشند. "
                        "پیش از معامله واقعی، استراتژی را آزمایش کن."
                    )

                    with st.expander("مشاهده کندل‌های اخیر"):
                        st.dataframe(df.tail(20), use_container_width=True)

        except Exception as exc:
            st.error(f"خطا در دریافت یا تحلیل داده: {exc}")

st.divider()
st.caption(
    "نسخه آزمایشی 1.0 | فاقد اتصال مستقیم به حساب معاملاتی و اجرای سفارش"
)
