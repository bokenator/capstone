"""
Verified API Surface (VAS)
==========================

Defines the allowed API calls for documentation-grounded conditions.
Used by: C2 (Docs), C4 (Schema+Docs), C6 (Docs+TDD), C7 (All)
"""

VAS_DESCRIPTION = """
## Verified API Surface (VAS)

You must ONLY use APIs from this verified surface. Any API call not in this list will be flagged as invalid.

### pandas
- `pd.Series` - constructor
- `pd.Series.rolling` - rolling window calculations
- `pd.Series.shift` - shift values
- `pd.Series.diff` - difference between consecutive values
- `pd.Series.fillna` - fill NA values
- `pd.Series.ffill` - forward fill
- `pd.Series.bfill` - backward fill
- `pd.Series.dropna` - drop NA values
- `pd.Series.values` - get numpy array
- `pd.Series.index` - get index
- `pd.DataFrame` - constructor
- `pd.DataFrame.rolling` - rolling window calculations
- `pd.DataFrame.shift` - shift values
- `pd.DataFrame.fillna` - fill NA values

### numpy
- `np.array` - create array
- `np.zeros` - create zeros array
- `np.ones` - create ones array
- `np.full` - create array filled with value
- `np.empty` - create empty array
- `np.nan` - NaN constant
- `np.isnan` - check for NaN
- `np.isfinite` - check if finite (not NaN or inf)
- `np.where` - conditional selection
- `np.mean` - mean calculation
- `np.std` - standard deviation
- `np.sum` - sum
- `np.abs` - absolute value
- `np.maximum` - element-wise maximum
- `np.minimum` - element-wise minimum
- `np.inf` - infinity constant
- `np.ndarray` - array type
- `np.float64` - float64 dtype
- `np.float32` - float32 dtype
- `np.int64` - int64 dtype
- `np.int32` - int32 dtype
- `np.int8` - int8 dtype
- `np.bool_` - boolean dtype

### vectorbt
- `vbt.MA.run` - moving average indicator
- `vbt.RSI.run` - RSI indicator
- `vbt.MACD.run` - MACD indicator
- `vbt.ATR.run` - ATR indicator
- `vbt.portfolio.nb.order_nb` - create order
- `vbt.portfolio.enums.NoOrder` - no order sentinel value
- `vbt.portfolio.enums.Direction` - order direction enum
- `vbt.portfolio.enums.SizeType` - size type enum

### scipy.stats
- `scipy.stats.linregress` - linear regression
"""

# VAS as a set for programmatic validation
VAS_SET = {
    # pandas
    "pd.Series",
    "pd.Series.rolling",
    "pd.Series.shift",
    "pd.Series.diff",
    "pd.Series.fillna",
    "pd.Series.ffill",
    "pd.Series.bfill",
    "pd.Series.dropna",
    "pd.Series.values",
    "pd.Series.index",
    "pd.DataFrame",
    "pd.DataFrame.rolling",
    "pd.DataFrame.shift",
    "pd.DataFrame.fillna",
    # numpy
    "np.array",
    "np.zeros",
    "np.ones",
    "np.full",
    "np.empty",
    "np.nan",
    "np.isnan",
    "np.isfinite",
    "np.where",
    "np.mean",
    "np.std",
    "np.sum",
    "np.abs",
    "np.maximum",
    "np.minimum",
    "np.inf",
    # numpy types
    "np.ndarray",
    "np.float64",
    "np.float32",
    "np.int64",
    "np.int32",
    "np.int8",
    "np.bool_",
    # vectorbt
    "vbt.MA.run",
    "vbt.RSI.run",
    "vbt.MACD.run",
    "vbt.ATR.run",
    "vbt.portfolio.nb.order_nb",
    "vbt.portfolio.enums.NoOrder",
    "vbt.portfolio.enums.Direction",
    "vbt.portfolio.enums.SizeType",
    # scipy
    "scipy.stats.linregress",
}
