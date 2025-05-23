import numpy as np
import pandas as pd
from fredapi import Fred


class US:
    def __init__(self):
        self.fred = Fred(api_key='ab746526067b62ac85bcf51ef80eae5a')

        ## all the info from categories you can search for
        # Stock Market Index
        self.stocks = {
            'SP500': 'S&P 500', 'DJIA': 'Dow Jones',
            'NASDAQCOM': 'NASDAQ', 'WILL5000INDFC': 'Wilshire 5000',
            'NASDAQ100': 'NASDAQ 100'
        }

        # Interest Rates
        self.rates = {
            'AMERIBOR': 'Overnight Unsecured AMERIBOR',
            'MORTGAGE30US': '30-Year Fixed Rate Mortgage Average',
            'DFEDTARU': 'Federal Funds Rate Upper'
        }

        # Macro Econ Data
        self.econ = {
            'CPIAUCSL': 'CPI', 'CORESTICKM159SFRBATL': 'CPI less food & energy',
            'PCE': 'PCE', 'EXHOSLUSM495S': 'Existing Home Sales',
            'HOSMEDUSM052N': 'Median Sales Price of Existing Homes',
            'CSUSHPINSA': 'S&P/Case-Shiller U.S. National Home Price Index',
            'PPIACO': 'PPI: All Commodities',
            'PCUOMFGOMFG': 'PPI: Total Manufacturing Industries',
            'PCUATRNWRATRNWR': 'PPI: Transportation and Warehousing Industries',
            'PCUARETTRARETTR': 'PPI: Total Retail Trade Industries',
            'UNRATE': 'Unemployment Rate',
            'TOTALSA': 'Total Vehicle Sales',
            'USHVAC': 'Home Vacancy Rate for the United States',
            'CCSA': 'Continued Claims (Insured Unemployment)',
            'M2SL': 'M2',
            'BOGZ1FL153064476Q': 'Household Held Equities as a Percentage of Total Assets',
            'T10Y2Y': '10-Year Treasury Constant Maturity Minus 2-Year Treasury Constant Maturity',
            'DGS2': 'Market Yield on U.S. Treasury Securities at 2-Year Constant Maturity, Quoted on an Investment Basis'
        }

        # US Dollar
        self.currency = {
            'DTWEXBGS': 'Nominal Broad U.S. Dollar Index',
            'DEXCHUS': 'USD/RMB', 'DEXUSEU': 'USD/Euro',
            'DEXJPUS': 'USD/Japanese Yen'
        }

    def get_data(self, index, ndays=0):
        """从 FRED API 获取指定指标数据"""
        data = pd.DataFrame(self.fred.get_series(index)).reset_index()
        data.columns = ['DateTime', index]
        return data.iloc[-ndays:,]

    def get_pct_change(self, index, ndays=0, YoY=None):
        """计算指定指标的百分比变化"""
        data = pd.DataFrame(self.fred.get_series(index)).reset_index()
        data.columns = ['DateTime', f"{index} Growth"]
        data.set_index('DateTime', inplace=True)
        pct = data.pct_change(periods=1).reset_index()
        if YoY:
            pct = data.pct_change(periods=12).reset_index()
            pct.dropna(inplace=True)
        return pct.iloc[-ndays:,]


def compute_ewma_features(df, column):
    """计算指数加权移动平均 (EWMA) 版本的特征"""
    features = pd.DataFrame()
    features['DateTime'] = df['DateTime']

    # 计算均值回报 (Mean Return)
    for w in [5, 10, 21, 120]:
        log_returns = np.log(df[column] / df[column].shift(1))
        ewma_mean_ret = log_returns.ewm(halflife=w, adjust=False).mean()
        features[f'{column}_mean_ret_{w}d'] = ewma_mean_ret

    # 计算下行标准差 (Downside Deviation)
    for w in [5, 10, 21, 120]:
        log_returns = np.log(df[column] / df[column].shift(1))
        log_returns[log_returns > 0] = np.nan  # 只保留负 log return
        downside_std = log_returns.ewm(halflife=w, adjust=False).std()
        features[f'{column}_downside_std_{w}d'] = downside_std

    # 计算 Sortino Ratio
    for w in [5, 10, 21, 120]:
        mean_ret = features[f'{column}_mean_ret_{w}d']
        downside_std = features[f'{column}_downside_std_{w}d']
        features[f'{column}_sortino_{w}d'] = mean_ret / downside_std

    return features


def generate_features(df, asset_column, api_key):
    """
    生成完整的特征：
    - 计算资产 (asset_column) 的技术特征
    - 获取并计算宏观经济特征
    """
    # 初始化数据
    us_data = US()

    # 计算资产相关特征
    df[f'{asset_column}_returns'] = np.log(df[asset_column] / df[asset_column].shift(1))
    asset_features = compute_ewma_features(df, asset_column)
    df = df.merge(asset_features, on='DateTime', how='left')

    # 获取宏观经济数据
    T10Y2Y = us_data.get_data('T10Y2Y').rename(columns={'DateTime': 'DateTime'})
    DGS2 = us_data.get_data('DGS2').rename(columns={'DateTime': 'DateTime'})

    # 计算收益率曲线斜率 EWMA
    T10Y2Y["T10Y2Y_EWMA_10"] = T10Y2Y["T10Y2Y"].ewm(halflife=10, adjust=False).mean()

    # 计算 2年期国债收益率变化 EWMA
    DGS2["DGS2_diff"] = DGS2["DGS2"].diff()
    DGS2["DGS2_diff_EWMA_21"] = DGS2["DGS2_diff"].ewm(halflife=21, adjust=False).mean()

    # 合并宏观经济特征
    df = pd.merge(df, T10Y2Y[['DateTime', 'T10Y2Y_EWMA_10']], on='DateTime', how='left')
    df = pd.merge(df, DGS2[['DateTime', 'DGS2_diff_EWMA_21']], on='DateTime', how='left')

    # 计算 VIX EWMA
    if 'VIX' in df.columns:
        df['VIX_EWMA_63'] = df['VIX'].ewm(halflife=63, adjust=False).mean()

    # 填充缺失值
    df.ffill(inplace=True)

    return df
